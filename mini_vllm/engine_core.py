"""Engine core: the step loop that drives scheduler -> runner -> kv manager.

    +-----------+     schedule()      +-----------+
    | Scheduler | ------------------> | prefills  |  (new requests, chunked)
    |           |                     | decodes   |  (one token each)
    +-----------+                     +-----------+
          |                                 |
          v allocate_slots                  v run()
    +-------------+                  +--------------+
    | KVCacheMgr  | <--------------- | ModelRunner  |  writes K/V into blocks
    | (blocks)    |   kv_cache       | (model fwd)  |
    +-------------+                  +--------------+
                                            |
                                            v sample
                                      next token ids

One ``step()`` == one iteration of the engine loop == at most two forward
passes (one for the prefill group, one for the decode group). Splitting them is
deliberate: the decode group is always ``query_len == 1`` for every sequence,
so it runs with zero padding waste, and it makes the prefill/decode cost
difference directly measurable.
"""

from __future__ import annotations

import time
from typing import List

import torch

from .config import EngineConfig
from .kv_manager import KVCacheManager
from .model_runner import ModelRunner
from .scheduler import Scheduler, SchedulerOutput
from .sequence import RequestOutput, SamplingParams, Sequence, SequenceStatus


class EngineCore:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.device = config.resolve_device()
        model_dtype = config.resolve_dtype(self.device)
        kv_dtype = config.resolve_kv_dtype(self.device, model_dtype)

        self.kv_manager = KVCacheManager(config, self.device, kv_dtype)
        self.model_runner = ModelRunner(config, self.kv_manager)
        self.scheduler = Scheduler(config, self.kv_manager)

        self.eos_token_id = config.eos_token_id
        self.next_seq_id = 0
        self._seqs: dict[int, Sequence] = {}
        # Aborts arrive from the HTTP thread while ``step()`` runs on the engine
        # thread. Freeing blocks immediately would race with a forward pass that
        # already scheduled the sequence, so requests are queued and applied at
        # the top of the next step.
        self._pending_aborts: set[int] = set()

        # Cost accounting, surfaced by /stats. This is what shows the
        # prefill-vs-decode difference while the server is live.
        self.step_count = 0
        self.prefill_steps = 0
        self.decode_steps = 0
        self.prefill_tokens = 0
        self.decode_tokens = 0
        self.prefill_time = 0.0
        self.decode_time = 0.0
        self.schedule_time = 0.0

    # ------------------------------------------------------------- requests
    def add_request(
        self,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
        prompt_text: str = "",
        meta: dict | None = None,
    ) -> Sequence:
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        total = len(prompt_token_ids) + sampling_params.max_tokens
        if total > self.config.max_model_len:
            raise ValueError(
                f"prompt ({len(prompt_token_ids)}) + max_tokens "
                f"({sampling_params.max_tokens}) exceeds max_model_len "
                f"({self.config.max_model_len})"
            )
        if len(self.scheduler.running) >= self.config.max_num_seqs:
            # Not fatal: it just waits. Kept as a soft signal in the logs.
            pass

        seq = Sequence(
            seq_id=self.next_seq_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            prompt_text=prompt_text,
            meta=meta,
        )
        self.next_seq_id += 1
        self._seqs[seq.seq_id] = seq
        self.scheduler.add(seq)
        return seq

    def abort_request(self, seq_id: int) -> bool:
        """Queue an abort; safe to call from any thread."""
        if seq_id not in self._seqs:
            return False
        self._pending_aborts.add(seq_id)
        return True

    def _drain_aborts(self) -> None:
        while self._pending_aborts:
            seq_id = self._pending_aborts.pop()
            self.scheduler.abort(seq_id)
            self._seqs.pop(seq_id, None)

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    def num_unfinished(self) -> int:
        return len(self.scheduler.waiting) + len(self.scheduler.running)

    # ------------------------------------------------------------------ step
    def step(self) -> List[RequestOutput]:
        """Run one engine iteration; returns the tokens produced this step."""
        self._drain_aborts()
        t0 = time.perf_counter()
        sched: SchedulerOutput = self.scheduler.schedule()
        self.schedule_time += time.perf_counter() - t0
        self.step_count += 1

        outputs: List[RequestOutput] = []

        if sched.prefills:
            t = time.perf_counter()
            logits = self.model_runner.execute(sched.prefills)
            self._sync()
            self.prefill_time += time.perf_counter() - t
            self.prefill_steps += 1
            self.prefill_tokens += sched.num_prefill_tokens()
            self._postprocess(sched.prefills, logits, outputs)

        if sched.decodes:
            t = time.perf_counter()
            logits = self.model_runner.execute(sched.decodes)
            self._sync()
            self.decode_time += time.perf_counter() - t
            self.decode_steps += 1
            self.decode_tokens += sched.num_decode_tokens()
            self._postprocess(sched.decodes, logits, outputs)

        return outputs

    def _postprocess(
        self, seqs: List[Sequence], logits: torch.Tensor, outputs: List[RequestOutput]
    ) -> None:
        next_tokens = self.model_runner.sample(logits, seqs)
        for seq, token_id in zip(seqs, next_tokens):
            seq.num_computed_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            # A chunked prefill that is not done yet produces no token.
            if seq.num_computed_tokens < len(seq.get_token_ids()):
                continue

            seq.append_token(token_id)
            reason = self._finish_reason(seq, token_id)
            if reason is not None:
                seq.status = (
                    SequenceStatus.FINISHED_STOPPED if reason == "stop"
                    else SequenceStatus.FINISHED_LENGTH
                )
                self.scheduler.free_finished(seq)
                self._seqs.pop(seq.seq_id, None)

            outputs.append(
                RequestOutput(
                    seq_id=seq.seq_id,
                    token_id=token_id,
                    output_token_ids=list(seq.output_token_ids),
                    finished=reason is not None,
                    finish_reason=reason,
                    prompt_tokens=seq.num_prompt_tokens,
                    meta=seq.meta,
                )
            )

    def _finish_reason(self, seq: Sequence, token_id: int) -> str | None:
        sp = seq.sampling_params
        if not sp.ignore_eos and self.eos_token_id is not None and token_id == self.eos_token_id:
            return "stop"
        if len(seq.output_token_ids) >= sp.max_tokens:
            return "length"
        return None

    def _sync(self) -> None:
        if self.config.time_steps and self.device.type == "cuda":
            torch.cuda.synchronize()

    # ------------------------------------------------------------- utilities
    def run_to_completion(self, max_steps: int = 100_000) -> List[RequestOutput]:
        """Drain the engine (used by tests and by the offline benchmark)."""
        all_out: List[RequestOutput] = []
        steps = 0
        while self.has_unfinished():
            got = self.step()
            all_out.extend(got)
            steps += 1
            if steps > max_steps:
                raise RuntimeError("engine did not drain")
        return all_out

    def warmup(self, tokens: int = 8) -> None:
        self.model_runner.warmup(tokens)

    def stats(self) -> dict:
        def rate(tokens: int, secs: float) -> float:
            return round(tokens / secs, 2) if secs > 0 else 0.0

        return {
            "model": self.model_runner.stats(),
            "kv_cache": self.kv_manager.stats(),
            "scheduler": self.scheduler.stats(),
            "steps": self.step_count,
            "prefill_steps": self.prefill_steps,
            "decode_steps": self.decode_steps,
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
            "prefill_seconds": round(self.prefill_time, 4),
            "decode_seconds": round(self.decode_time, 4),
            "prefill_tokens_per_s": rate(self.prefill_tokens, self.prefill_time),
            "decode_tokens_per_s": rate(self.decode_tokens, self.decode_time),
            "schedule_seconds": round(self.schedule_time, 4),
        }
