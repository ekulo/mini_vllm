"""Public entry points: a synchronous offline engine and an async serving engine.

``LLMEngine``    -- convenience wrapper: text in, text out, drains the engine.
``AsyncEngine``  -- runs the step loop on a dedicated thread and fans results out
                    to asyncio queues, one per request, so the HTTP layer can
                    stream tokens without ever calling the model itself.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, List

from .config import EngineConfig
from .engine_core import EngineCore, RequestOutput
from .sequence import SamplingParams
from .tokenizer import Tokenizer


@dataclass
class GenerationResult:
    text: str
    token_ids: List[int] = field(default_factory=list)
    prompt_tokens: int = 0
    finish_reason: str = "length"
    latency: float = 0.0
    meta: dict = field(default_factory=dict)


class LLMEngine:
    """Everything wired together: tokenizer + engine core."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.tokenizer = Tokenizer(os.path.join(self.config.model_dir, self.config.vocab_file))

        # The vocab file is the source of truth for both size and EOS.
        self.config.vocab_size = self.tokenizer.vocab_size
        if self.config.eos_token_id is None:
            self.config.eos_token_id = self.tokenizer.eos_id

        self.core = EngineCore(self.config)

    # ------------------------------------------------------------ lifecycle
    def warmup(self, tokens: int = 8) -> None:
        self.core.warmup(tokens=tokens)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        return self.tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)

    # -------------------------------------------------------------- offline
    def generate_ids(self, prompt_ids: List[int], sampling_params: SamplingParams | None = None,
                     meta: dict | None = None) -> GenerationResult:
        sp = sampling_params or SamplingParams()
        sp.validate(self.config.max_model_len)
        t0 = time.perf_counter()

        seq = self.core.add_request(prompt_ids, sp, meta=meta)
        outputs = self.core.run_to_completion()
        produced = [o for o in outputs if o.seq_id == seq.seq_id]

        finish = "length"
        if produced and produced[-1].finish_reason:
            finish = produced[-1].finish_reason
        if sp.ignore_eos and finish == "stop":
            finish = "length"
        return GenerationResult(
            text=self.tokenizer.decode(seq.output_token_ids),
            token_ids=list(seq.output_token_ids),
            prompt_tokens=seq.num_prompt_tokens,
            finish_reason=finish,
            latency=time.perf_counter() - t0,
            meta=meta or {},
        )

    def generate(self, prompt: str, sampling_params: SamplingParams | None = None,
                 **kwargs) -> GenerationResult:
        """Convenience: text prompt in, text out, engine drained to completion."""
        if kwargs:
            sampling_params = SamplingParams(**{**(vars(sampling_params) if sampling_params else {}), **kwargs})
        sp = sampling_params or SamplingParams()
        return self.generate_ids(self.encode(prompt), sp, meta={"prompt": prompt})

    def generate_batch(self, prompts: List[str], sampling_params: SamplingParams | None = None) -> List[GenerationResult]:
        """Add every prompt before stepping, so they batch together."""
        sp = sampling_params or SamplingParams()
        sp.validate(self.config.max_model_len)
        t0 = time.perf_counter()
        seqs = [self.core.add_request(self.encode(p), sp, meta={"prompt": p}) for p in prompts]
        self.core.run_to_completion()
        dt = time.perf_counter() - t0
        out = []
        for seq in seqs:
            out.append(
                GenerationResult(
                    text=self.tokenizer.decode(seq.output_token_ids),
                    token_ids=list(seq.output_token_ids),
                    prompt_tokens=seq.num_prompt_tokens,
                    finish_reason="stop" if seq.status.value == "stopped" else "length",
                    latency=dt,
                    meta=seq.meta,
                )
            )
        return out

    def stats(self) -> dict:
        return self.core.stats()

    def shutdown(self) -> None:  # pragma: no cover - symmetry for the server
        pass


class AsyncEngine:
    """Engine loop on a background thread; results delivered to asyncio queues.

    Why a thread: model forward is blocking, and the HTTP layer must stay
    responsive while other requests stream. ``call_soon_threadsafe`` is the only
    cross-thread hop needed.
    """

    def __init__(self, engine: LLMEngine) -> None:
        self.engine = engine
        self.core = engine.core
        self.tokenizer = engine.tokenizer
        self._queues: dict[int, asyncio.Queue] = {}
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.steps_run = 0

    # ---------------------------------------------------------------- thread
    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._thread is not None:
            return
        self._loop = loop
        self._thread = threading.Thread(target=self._run, name="mini-vllm-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wakeup.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        assert self._loop is not None
        while not self._stop.is_set():
            if not self.core.has_unfinished():
                # Nothing to do: park until a request arrives (or 0.2 s, so a
                # shutdown flag is noticed promptly).
                self._wakeup.wait(0.2)
                self._wakeup.clear()
                continue
            outputs = self.core.step()
            self.steps_run += 1
            for out in outputs:
                with self._lock:
                    queue = self._queues.get(out.seq_id)
                    if out.finished:
                        self._queues.pop(out.seq_id, None)
                if queue is not None:
                    self._loop.call_soon_threadsafe(queue.put_nowait, out)

    # ------------------------------------------------------------------- api
    async def stream(
        self,
        prompt_ids: List[int],
        sampling_params: SamplingParams,
        meta: dict | None = None,
    ) -> AsyncIterator[RequestOutput]:
        sampling_params.validate(self.core.config.max_model_len)
        queue: asyncio.Queue = asyncio.Queue()
        seq = self.core.add_request(prompt_ids, sampling_params, meta=meta)
        with self._lock:
            self._queues[seq.seq_id] = queue
        self._wakeup.set()
        try:
            while True:
                out: RequestOutput = await queue.get()
                yield out
                if out.finished:
                    return
        finally:
            # Client disconnected or the stream ended: make sure the sequence
            # does not keep occupying KV blocks.
            with self._lock:
                still_there = self._queues.pop(seq.seq_id, None)
            if still_there is not None:
                self.core.abort_request(seq.seq_id)
                self._wakeup.set()  # let the engine thread apply it promptly

    async def complete(
        self,
        prompt_ids: List[int],
        sampling_params: SamplingParams,
        meta: dict | None = None,
    ) -> RequestOutput | None:
        last: RequestOutput | None = None
        async for out in self.stream(prompt_ids, sampling_params, meta=meta):
            last = out
        return last
