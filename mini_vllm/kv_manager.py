"""Paged KV-cache manager.

This is the piece that makes it an "engine" rather than a for-loop: instead of
one growing ``torch.cat`` per sequence, the cache is a fixed pool of blocks and
every sequence holds a *block table* (the list of block ids it owns).

Layout of the single cache tensor::

    (2, num_layers, num_blocks, block_size, num_heads, head_dim)
     ^  ^           ^           ^
     |  |           |           +-- tokens inside one block (contiguous)
     |  |           +-------------- block pool
     |  +-------------------------- layer index
     +----------------------------- 0 = K, 1 = V

A token at logical position ``p`` of a sequence lives at slot
``block_table[p // block_size] * block_size + (p % block_size)``.

No copy-on-write / prefix sharing here -- one block belongs to exactly one
sequence -- which keeps the allocator to ~60 lines. Because a block newly
handed to a sequence may still contain another sequence's numbers, correctness
relies on a simple invariant: every slot below a sequence's length is written
by that sequence before it is ever read, and slots at or above its length are
masked out of attention. Hence no cache zeroing on reallocation.
"""

from __future__ import annotations

import math
from typing import List

import torch

from .config import EngineConfig
from .sequence import Sequence


class KVCacheManager:
    def __init__(self, config: EngineConfig, device: torch.device, dtype: torch.dtype) -> None:
        self.block_size = config.block_size
        self.num_blocks = config.num_blocks
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.max_model_len = config.max_model_len
        self.device = device
        self.dtype = dtype

        self.cache = torch.zeros(
            2, self.num_layers, self.num_blocks, self.block_size,
            self.num_heads, self.head_dim,
            dtype=dtype, device=device,
        )
        # Free list as a plain list used as a stack; ``set`` mirror makes
        # membership checks O(1) for the assertions/tests.
        self.free_blocks: List[int] = list(range(self.num_blocks))
        self._free = set(self.free_blocks)
        self.num_used_blocks = 0

    # ----------------------------------------------------------- accounting
    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def usage(self) -> float:
        return self.num_used_blocks / self.num_blocks if self.num_blocks else 0.0

    def max_slots_for(self, seq: Sequence) -> int:
        """How many more tokens this sequence could hold with today's free blocks."""
        start = seq.num_computed_tokens
        have = len(seq.block_table) * self.block_size - start
        return max(0, have + self.num_free_blocks * self.block_size)

    def blocks_needed(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.block_size)

    def can_allocate(self, seq: Sequence, num_tokens: int) -> bool:
        """Can ``seq`` hold ``num_computed_tokens + num_tokens`` tokens?"""
        need = self.blocks_needed(seq.num_computed_tokens + num_tokens) - len(seq.block_table)
        return need <= self.num_free_blocks

    # ------------------------------------------------------------ mutation
    def allocate_slots(self, seq: Sequence, num_tokens: int) -> bool:
        """Grow ``seq``'s block table so the next ``num_tokens`` tokens fit."""
        target = self.blocks_needed(seq.num_computed_tokens + num_tokens)
        need = target - len(seq.block_table)
        if need > self.num_free_blocks:
            return False
        for _ in range(need):
            block = self.free_blocks.pop()
            self._free.discard(block)
            seq.block_table.append(block)
        self.num_used_blocks += need
        return True

    def free(self, seq: Sequence) -> None:
        """Return every block the sequence owns to the pool."""
        for block in seq.block_table:
            if block in self._free:  # pragma: no cover - guards double free
                raise RuntimeError(f"block {block} freed twice")
            self.free_blocks.append(block)
            self._free.add(block)
        self.num_used_blocks -= len(seq.block_table)
        seq.block_table = []

    # ------------------------------------------------------------ helpers
    def slot_mapping(self, seq: Sequence, start: int, num_tokens: int) -> List[int]:
        """Flat cache slots for logical positions ``[start, start + num_tokens)``."""
        if start + num_tokens > len(seq.block_table) * self.block_size:
            raise ValueError("slot_mapping past the end of the block table")
        return [
            seq.block_table[p // self.block_size] * self.block_size + (p % self.block_size)
            for p in range(start, start + num_tokens)
        ]

    def stats(self) -> dict:
        return {
            "blocks_total": self.num_blocks,
            "blocks_used": self.num_used_blocks,
            "blocks_free": self.num_free_blocks,
            "block_size": self.block_size,
            "usage": round(self.usage, 4),
            "cache_mib": round(self.cache.numel() * self.cache.element_size() / 2**20, 1),
        }
