"""Block allocator invariants: no leaks, no double allocation, no overlap."""

from __future__ import annotations

import pytest
import torch

from mini_vllm import KVCacheManager, SamplingParams, Sequence
from tests.helpers import tiny_config


def _kv(blocks=16, block_size=4, **over):
    cfg = tiny_config(num_blocks=blocks, block_size=block_size, **over)
    return cfg, KVCacheManager(cfg, torch.device("cpu"), torch.float32)


def _seq(n_prompt, seq_id=0):
    return Sequence(seq_id, [2] * n_prompt, SamplingParams(max_tokens=8))


def test_allocation_is_rounded_up_to_blocks():
    _, kv = _kv(blocks=16, block_size=4)
    s = _seq(9)
    assert kv.allocate_slots(s, 9)
    assert len(s.block_table) == 3  # ceil(9/4)
    assert kv.num_used_blocks == 3


def test_freed_blocks_are_returned_to_the_pool():
    _, kv = _kv(blocks=8, block_size=4)
    s = _seq(10)
    kv.allocate_slots(s, 10)
    assert kv.num_free_blocks == 8 - 3
    kv.free(s)
    assert kv.num_free_blocks == 8
    assert kv.num_used_blocks == 0
    assert s.block_table == []


def test_blocks_are_never_handed_out_twice():
    _, kv = _kv(blocks=4, block_size=4)
    seqs = [_seq(4, i) for i in range(4)]
    for s in seqs:
        assert kv.allocate_slots(s, 4)
    assert kv.num_free_blocks == 0
    assert len({b for s in seqs for b in s.block_table}) == 4

    extra = _seq(1, 99)
    assert kv.allocate_slots(extra, 1) is False
    assert extra.block_table == []


def test_can_allocate_reports_before_mutating():
    _, kv = _kv(blocks=2, block_size=4)
    s = _seq(12)
    assert kv.can_allocate(s, 12) is False
    assert s.block_table == []          # nothing was consumed
    assert kv.allocate_slots(s, 12) is False
    assert kv.num_used_blocks == 0


def test_growth_is_incremental():
    _, kv = _kv(blocks=8, block_size=4)
    s = _seq(3)
    kv.allocate_slots(s, 3)
    assert len(s.block_table) == 1
    s.num_computed_tokens = 3
    kv.allocate_slots(s, 1)             # still fits in block 0
    assert len(s.block_table) == 1
    s.num_computed_tokens = 4
    kv.allocate_slots(s, 1)             # needs a second block
    assert len(s.block_table) == 2


def test_slot_mapping_is_dense_and_in_range():
    _, kv = _kv(blocks=8, block_size=4)
    s = _seq(10)
    kv.allocate_slots(s, 10)
    slots = kv.slot_mapping(s, 0, 10)
    assert len(set(slots)) == 10          # no slot handed out twice
    assert all(0 <= x < 8 * 4 for x in slots)
    # Position p must land in block_table[p // block_size].
    for p, slot in enumerate(slots):
        assert slot // 4 == s.block_table[p // 4]
        assert slot % 4 == p % 4


def test_slot_mapping_tracks_a_chunked_write():
    _, kv = _kv(blocks=8, block_size=4)
    s = _seq(6)
    kv.allocate_slots(s, 6)
    first = kv.slot_mapping(s, 0, 4)
    second = kv.slot_mapping(s, 4, 2)
    assert len(first) == 4 and len(second) == 2
    assert not set(first) & set(second)
    assert first + second == kv.slot_mapping(s, 0, 6)


def test_slot_mapping_rejects_overrun():
    _, kv = _kv(blocks=8, block_size=4)
    s = _seq(2)
    kv.allocate_slots(s, 2)
    with pytest.raises(ValueError):
        kv.slot_mapping(s, 0, 5)


def test_fragmentation_recovers():
    """Free everything and the pool must be exactly whole again."""
    _, kv = _kv(blocks=32, block_size=8)
    seqs = [_seq(8 * (i % 5 + 1), i) for i in range(10)]
    for s in seqs:
        kv.allocate_slots(s, len(s.prompt_token_ids))
    for s in seqs:
        kv.free(s)
    assert kv.num_free_blocks == 32
    assert sorted(kv.free_blocks) == list(range(32))


def test_stale_block_table_is_caught():
    """``free`` must refuse a table that still references an already-free block."""
    _, kv = _kv(blocks=4, block_size=4)
    s = _seq(4)
    kv.allocate_slots(s, 4)
    block = s.block_table[0]
    kv.free(s)

    s.block_table = [block]  # simulate a bookkeeping bug elsewhere
    with pytest.raises(RuntimeError):
        kv.free(s)
