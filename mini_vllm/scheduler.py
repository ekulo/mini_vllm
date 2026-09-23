"""Continuous-batching scheduler.

One engine step is: *admit some new prefills, then advance every resident
sequence by one token*. That mix is what "continuous batching" means -- a long
generation never blocks a new request from starting, unlike static batching.

Policy: FCFS with a token budget and a sequence-count cap.
Preemption: recompute. If the block pool cannot give a decode sequence another
slot, that sequence is evicted (blocks returned, ``num_computed_tokens`` reset
to 0) and pushed back to the *front* of the waiting queue, so its whole token
list is re-run later. Correct, simple, and cheap for a model this small.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List

from .config import EngineConfig
from .kv_manager import KVCacheManager
from .sequence import Sequence, SequenceStatus


@dataclass
class SchedulerOutput:
    prefills: List[Sequence] = field(default_factory=list)
    decodes: List[Sequence] = field(default_factory=list)
    preempted: List[Sequence] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.prefills and not self.decodes

    def num_prefill_tokens(self) -> int:
        return sum(s.num_scheduled_tokens for s in self.prefills)

    def num_decode_tokens(self) -> int:
        return sum(s.num_scheduled_tokens for s in self.decodes)


class Scheduler:
    def __init__(self, config: EngineConfig, kv_manager: KVCacheManager) -> None:
        self.config = config
        self.kv_manager = kv_manager
        self.waiting: deque[Sequence] = deque()
        self.running: List[Sequence] = []
        self.finished: List[Sequence] = []

    # ------------------------------------------------------------- lifecycle
    def add(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def abort(self, seq_id: int) -> bool:
        for queue in (self.waiting, self.running):
            for seq in list(queue):
                if seq.seq_id == seq_id:
                    queue.remove(seq)
                    self.kv_manager.free(seq)
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.finished.append(seq)
                    return True
        return False

    def free_finished(self, seq: Sequence) -> None:
        """Retire a sequence that hit EOS / max_tokens."""
        self.kv_manager.free(seq)
        if seq in self.running:
            self.running.remove(seq)
        self.finished.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    def _preempt(self, seq: Sequence) -> None:
        self.running.remove(seq)
        self.kv_manager.free(seq)
        seq.reset_for_recompute()
        self.waiting.appendleft(seq)  # newest victim retries first

    # -------------------------------------------------------------- schedule
    def _prefill_tokens(self, seq: Sequence, budget: int) -> int:
        """How many not-yet-cached tokens to compute for ``seq`` under ``budget``."""
        remaining = seq.num_prefill_tokens - seq.num_computed_tokens
        if remaining <= 0:
            return 0
        if not self.config.chunked_prefill:
            return remaining if remaining <= budget else 0
        return min(remaining, budget)

    def _grow_or_preempt(self, seq: Sequence, n: int, out: SchedulerOutput) -> int:
        """Try to reserve slots for ``n`` tokens; return the granted amount (0 = failed).

        A sequence that cannot grow is preempted, not stalled: recomputing it
        later is always correct, and it lets the pool move to a sequence that
        does fit.
        """
        if self.kv_manager.allocate_slots(seq, n):
            return n
        if self.config.chunked_prefill:
            shrunk = min(n, self.kv_manager.max_slots_for(seq))
            if shrunk > 0 and self.kv_manager.allocate_slots(seq, shrunk):
                return shrunk
        self._preempt(seq)
        out.preempted.append(seq)
        return 0

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        cfg = self.config
        budget = cfg.max_num_batched_tokens
        scheduled: set[int] = set()

        # ---- 1. admit new requests (prefill, optionally chunked) ----------
        while self.waiting and len(self.running) < cfg.max_num_seqs and budget > 0:
            seq = self.waiting[0]
            n = self._prefill_tokens(seq, budget)
            if n <= 0:
                break  # needs a whole step's budget; try again next step

            if self.kv_manager.allocate_slots(seq, n):
                pass
            elif cfg.chunked_prefill:
                n = min(n, self.kv_manager.max_slots_for(seq))
                if n <= 0 or not self.kv_manager.allocate_slots(seq, n):
                    break  # pool exhausted -> a resident sequence must finish
            else:
                break

            self.waiting.popleft()
            seq.num_scheduled_tokens = n
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            out.prefills.append(seq)
            scheduled.add(seq.seq_id)
            budget -= n

        # ---- 2. finish chunks of sequences already mid-prefill ------------
        # Without this pass a chunked prefill admitted above would be mistaken
        # for a decode next step and would emit a token from a half-filled cache.
        for seq in list(self.running):
            if seq.seq_id in scheduled or budget <= 0:
                continue
            if not seq.is_prefilling:
                continue
            n = self._prefill_tokens(seq, budget)
            if n <= 0:
                break
            granted = self._grow_or_preempt(seq, n, out)
            if granted <= 0:
                continue
            seq.num_scheduled_tokens = granted
            out.prefills.append(seq)
            scheduled.add(seq.seq_id)
            budget -= granted

        # ---- 3. one decode token for everyone else ------------------------
        for seq in list(self.running):
            if seq.seq_id in scheduled:
                continue  # prefilling this step; decode starts next step
            if budget <= 0:
                break
            if self.kv_manager.allocate_slots(seq, 1):
                seq.num_scheduled_tokens = 1
                out.decodes.append(seq)
                budget -= 1
            else:
                self._preempt(seq)
                out.preempted.append(seq)

        return out

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "finished": len(self.finished),
        }
