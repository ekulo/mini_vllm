"""Scheduler behaviour: continuous batching, budgets, preemption, drain."""

from __future__ import annotations

import torch

from mini_vllm import KVCacheManager, Scheduler, SamplingParams, Sequence
from mini_vllm.sequence import SequenceStatus
from tests.helpers import tiny_config


def _setup(**over):
    cfg = tiny_config(**over)
    kv = KVCacheManager(cfg, torch.device("cpu"), torch.float32)
    return cfg, kv, Scheduler(cfg, kv)


def _add(sched, n_prompt, seq_id, max_tokens=8):
    s = Sequence(seq_id, [2] * n_prompt, SamplingParams(max_tokens=max_tokens))
    sched.add(s)
    return s


def _finish_prefill(seq):
    """Pretend the prefill forward pass just ran for this sequence."""
    seq.num_computed_tokens = len(seq.get_token_ids())
    seq.num_scheduled_tokens = 0


def test_respects_max_num_seqs():
    _, _, sched = _setup(max_num_seqs=2)
    for i in range(5):
        _add(sched, 4, i)
    out = sched.schedule()
    assert len(out.prefills) == 2
    assert len(sched.running) == 2
    assert len(sched.waiting) == 3


def test_decode_is_scheduled_for_every_resident_sequence():
    _, _, sched = _setup(max_num_seqs=4)
    seqs = [_add(sched, 4, i) for i in range(3)]
    first = sched.schedule()
    assert len(first.prefills) == 3

    for s in seqs:
        _finish_prefill(s)

    second = sched.schedule()
    assert second.prefills == []
    assert len(second.decodes) == 3
    assert all(s.num_scheduled_tokens == 1 for s in second.decodes)


def test_prefill_and_decode_share_one_step():
    """Continuous batching: an old sequence decodes while a new one prefills."""
    _, _, sched = _setup(max_num_seqs=4)
    old = _add(sched, 4, 0)
    sched.schedule()
    _finish_prefill(old)

    _add(sched, 6, 1)
    out = sched.schedule()
    assert [s.seq_id for s in out.prefills] == [1]
    assert [s.seq_id for s in out.decodes] == [0]


def test_token_budget_chunks_a_long_prefill():
    _, _, sched = _setup(max_num_batched_tokens=5, chunked_prefill=True)
    s = _add(sched, 12, 0)

    out = sched.schedule()
    assert s.num_scheduled_tokens == 5
    assert out.prefills == [s]
    assert out.decodes == []  # nothing to sample from a half-done prompt

    s.num_computed_tokens += s.num_scheduled_tokens
    s.num_scheduled_tokens = 0
    sched.schedule()
    assert s.num_scheduled_tokens == 5


def test_unchunked_prefill_uses_the_whole_prompt():
    _, _, sched = _setup(max_num_batched_tokens=100, chunked_prefill=False)
    s = _add(sched, 12, 0)
    out = sched.schedule()
    assert s.num_scheduled_tokens == 12
    assert out.prefills == [s]


def test_preemption_recomputes_a_decoding_sequence():
    # Two blocks of four tokens, two prompt-only sequences: the pool is exactly
    # full, so whichever sequence needs a *new* block must be evicted.
    _, kv, sched = _setup(
        num_blocks=2, block_size=4, max_num_seqs=4, chunked_prefill=False
    )
    a = _add(sched, 3, 0)   # 3 tokens -> block 0, position 3 is still in it
    b = _add(sched, 4, 1)   # 4 tokens -> block 1, position 4 needs a new block

    out = sched.schedule()
    assert {s.seq_id for s in out.prefills} == {0, 1}
    for s in (a, b):
        _finish_prefill(s)
    assert kv.num_free_blocks == 0

    out = sched.schedule()
    assert [s.seq_id for s in out.decodes] == [0]          # a still fits
    assert [s.seq_id for s in out.preempted] == [1]        # b does not

    assert b in sched.waiting
    assert b.num_computed_tokens == 0, "a preempted sequence must recompute"
    assert b.block_table == []
    assert b.status is SequenceStatus.WAITING
    assert b not in sched.running
    assert kv.num_used_blocks == 1


def test_preempted_sequence_is_readmitted_and_finishes():
    """Drain the whole engine even when it has to preempt along the way."""
    cfg, kv, sched = _setup(num_blocks=3, block_size=4, max_num_seqs=3, chunked_prefill=False)
    for i in range(3):
        _add(sched, 4, i, max_tokens=3)

    steps = 0
    preemptions = 0
    produced: dict[int, int] = {}
    while sched.has_unfinished():
        out = sched.schedule()
        preemptions += len(out.preempted)
        for seq in out.prefills + out.decodes:
            seq.num_computed_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.num_computed_tokens < len(seq.get_token_ids()):
                continue
            seq.append_token(2)
            produced[seq.seq_id] = produced.get(seq.seq_id, 0) + 1
            if produced[seq.seq_id] >= seq.sampling_params.max_tokens:
                sched.free_finished(seq)
        steps += 1
        assert steps < 200, "scheduler failed to make progress"

    assert preemptions > 0, "this configuration was supposed to force preemption"
    assert produced == {0: 3, 1: 3, 2: 3}
    assert kv.num_free_blocks == kv.num_blocks, "blocks leaked"


def test_abort_removes_and_frees():
    _, kv, sched = _setup(max_num_seqs=4, chunked_prefill=False)
    a = _add(sched, 4, 0)
    b = _add(sched, 4, 1)

    assert sched.abort(0) is True
    assert a not in sched.waiting
    assert sched.abort(0) is False  # already gone

    out = sched.schedule()
    assert [s.seq_id for s in out.prefills] == [1]

    assert sched.abort(1) is True
    assert b.block_table == []
    assert kv.num_free_blocks == kv.num_blocks


def test_free_finished_returns_blocks():
    _, kv, sched = _setup(max_num_seqs=4, chunked_prefill=False)
    a = _add(sched, 4, 0)
    sched.schedule()
    assert kv.num_used_blocks > 0

    sched.free_finished(a)
    assert kv.num_used_blocks == 0
    assert a in sched.finished
    assert not sched.has_unfinished()


def test_has_unfinished_tracks_both_queues():
    _, _, sched = _setup(max_num_seqs=1)
    _add(sched, 4, 0)
    _add(sched, 4, 1)
    assert sched.has_unfinished()

    sched.schedule()
    assert sched.has_unfinished()  # seq 1 is still waiting
    sched.waiting.clear()
    sched.running.clear()
    assert not sched.has_unfinished()
