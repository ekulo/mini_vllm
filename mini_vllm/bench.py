"""Prefill vs decode benchmark.

The interesting number for an inference engine is not one tokens/s figure but
the *gap* between the two phases:

    prefill  many tokens, one pass   -> compute bound, cheap per token
    decode   one token per sequence  -> memory bound, expensive per token
             per pass, re-reading the
             whole KV cache

This script measures them separately, with device synchronisation, and prints a
table so the gap is obvious. Everything it reports comes out of the engine's own
``step()``, so it measures the real scheduling + paged-attention path.

    python -m mini_vllm.bench
    python -m mini_vllm.bench --device cuda --batch-sizes 1,8,32 --prompt-len 128 --gen-len 32
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

from mini_vllm._env import ensure_local_libs

ensure_local_libs()

from .config import EngineConfig  # noqa: E402
from .llm_engine import LLMEngine  # noqa: E402
from .sequence import SamplingParams  # noqa: E402


# --------------------------------------------------------------------------- #
#  timing helpers
# --------------------------------------------------------------------------- #
def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@dataclass
class PhaseTiming:
    total_seconds: float = 0.0
    steps: int = 0
    tokens: int = 0
    samples: List[float] = field(default_factory=list)

    def add(self, seconds: float, tokens: int) -> None:
        self.total_seconds += seconds
        self.steps += 1
        self.tokens += tokens
        self.samples.append(seconds)

    @property
    def tokens_per_s(self) -> float:
        return self.tokens / self.total_seconds if self.total_seconds > 0 else 0.0

    @property
    def ms_per_step(self) -> float:
        return 1000.0 * self.total_seconds / self.steps if self.steps else 0.0

    @property
    def ms_per_token(self) -> float:
        return 1000.0 * self.total_seconds / self.tokens if self.tokens else 0.0

    def percentile(self, p: float) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        k = max(0, min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1)))))
        return ordered[k] * 1000.0


def make_prompt(length: int, vocab_size: int, generator: torch.Generator) -> List[int]:
    """``<BOS>`` + random real tokens (ids 4.. are the normal words)."""
    body = torch.randint(4, vocab_size, (max(0, length - 1),), generator=generator).tolist()
    return [2] + body


# --------------------------------------------------------------------------- #
#  measurements
# --------------------------------------------------------------------------- #
def measure(engine: LLMEngine, batch_size: int, prompt_len: int, gen_len: int,
            repeats: int, vocab_size: int, seed: int = 0) -> Tuple[PhaseTiming, PhaseTiming]:
    """One config: time the prefill pass and each decode pass, averaged."""
    core = engine.core
    device = core.device
    gen = torch.Generator().manual_seed(seed)

    prefill = PhaseTiming()
    decode = PhaseTiming()

    for _ in range(repeats):
        prompts = [make_prompt(prompt_len, vocab_size, gen) for _ in range(batch_size)]
        sp = SamplingParams(max_tokens=gen_len + 1, temperature=0.0, ignore_eos=True)
        for p in prompts:
            core.add_request(p, sp)

        # ---- pass 1: every prompt in one batch -> pure prefill -------------
        sync(device)
        t0 = time.perf_counter()
        core.step()
        sync(device)
        prefill.add(time.perf_counter() - t0, batch_size * prompt_len)

        # ---- passes 2..N: one token per sequence -> pure decode -----------
        for _ in range(gen_len):
            sync(device)
            t0 = time.perf_counter()
            core.step()
            sync(device)
            decode.add(time.perf_counter() - t0, batch_size)

        assert not core.has_unfinished(), "benchmark left sequences behind"

    return prefill, decode


def bench_sweep(engine: LLMEngine, batch_sizes: List[int], prompt_len: int, gen_len: int,
                repeats: int, vocab_size: int) -> Dict[int, Tuple[PhaseTiming, PhaseTiming]]:
    results: Dict[int, Tuple[PhaseTiming, PhaseTiming]] = {}
    for bs in batch_sizes:
        engine.config.max_num_seqs = max(engine.config.max_num_seqs, bs)
        engine.warmup()
        results[bs] = measure(engine, bs, prompt_len, gen_len, repeats, vocab_size)
    return results


# --------------------------------------------------------------------------- #
#  reporting
# --------------------------------------------------------------------------- #
def print_sweep(results, prompt_len: int, gen_len: int, device) -> None:
    print()
    print("=" * 96)
    print(f"  PREFILL vs DECODE   device={device}   prompt={prompt_len} tok   generate={gen_len} tok")
    print("=" * 96)
    head = (
        f"{'batch':>6} | {'prefill ms':>11} {'prefill tok/s':>14} {'ms/tok':>8} | "
        f"{'decode ms/step':>15} {'decode tok/s':>13} {'ms/tok':>8} | {'speedup':>8}"
    )
    print(head)
    print("-" * 96)
    for bs, (pre, dec) in results.items():
        speedup = (dec.ms_per_token / pre.ms_per_token) if pre.ms_per_token else 0.0
        print(
            f"{bs:>6} | {pre.ms_per_step:>11.2f} {pre.tokens_per_s:>14.1f} {pre.ms_per_token:>8.3f} | "
            f"{dec.ms_per_step:>15.2f} {dec.tokens_per_s:>13.1f} {dec.ms_per_token:>8.3f} | "
            f"{speedup:>7.2f}x"
        )
    print("-" * 96)
    print("  ms/tok  = milliseconds per token; 'speedup' = decode ms/tok / prefill ms/tok")
    print("  prefill ms is the whole batch's prefill pass, i.e. the time to first token")
    print("=" * 96)


def print_percentiles(results) -> None:
    print()
    print("  Step latency percentiles (ms)")
    print(f"  {'batch':>6} | {'prefill p50':>12} {'prefill p90':>12} | {'decode p50':>11} {'decode p90':>11} {'decode p99':>11}")
    print("  " + "-" * 76)
    for bs, (pre, dec) in results.items():
        print(
            f"  {bs:>6} | {pre.percentile(50):>12.2f} {pre.percentile(90):>12.2f} | "
            f"{dec.percentile(50):>11.2f} {dec.percentile(90):>11.2f} {dec.percentile(99):>11.2f}"
        )


def print_memory(engine: LLMEngine) -> None:
    kv = engine.core.kv_manager.stats()
    print()
    print(f"  KV cache: {kv['blocks_total']} blocks x {kv['block_size']} tokens "
          f"= {kv['cache_mib']} MiB  ({kv['blocks_total'] * kv['block_size']} token slots)")
    if engine.core.device.type == "cuda":
        print(f"  CUDA peak allocated: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB, "
              f"reserved: {torch.cuda.max_memory_reserved() / 2**20:.1f} MiB")
    m = engine.core.model_runner.stats()
    print(f"  model: {m['parameters']:,} params, dtype={m['dtype']}, weights={m['weights']}")


# --------------------------------------------------------------------------- #
#  continuous-batching load test
# --------------------------------------------------------------------------- #
def bench_continuous(engine: LLMEngine, arrivals: int, prompt_len: int, gen_len: int,
                     submit_every: int, vocab_size: int) -> None:
    """Steady-state load: new requests arrive while others are still decoding."""
    core = engine.core
    device = core.device
    gen = torch.Generator().manual_seed(7)
    sp = SamplingParams(max_tokens=gen_len, temperature=0.0, ignore_eos=True)
    prompts = [make_prompt(prompt_len, vocab_size, gen) for _ in range(arrivals)]

    completed = 0
    step_times: List[float] = []
    sync(device)
    start = time.perf_counter()
    i = 0
    while i < len(prompts) or core.has_unfinished():
        for _ in range(submit_every):
            if i < len(prompts):
                core.add_request(prompts[i], sp)
                i += 1
        sync(device)
        t0 = time.perf_counter()
        outs = core.step()
        sync(device)
        step_times.append(time.perf_counter() - t0)
        completed += sum(1 for o in outs if o.finished)
    total = time.perf_counter() - start

    generated = arrivals * gen_len
    print()
    print("=" * 96)
    print("  CONTINUOUS BATCHING (new requests arrive while others decode)")
    print("=" * 96)
    print(f"  requests            : {arrivals} (submit {submit_every}/step)")
    print(f"  prompt / generate   : {prompt_len} / {gen_len} tokens")
    print(f"  wall time           : {total:.3f} s")
    print(f"  steps               : {len(step_times)}")
    print(f"  output tokens       : {generated}")
    print(f"  throughput          : {generated / total:.1f} output tok/s")
    print(f"  total tokens        : {arrivals * (prompt_len + gen_len)} "
          f"-> {(arrivals * (prompt_len + gen_len)) / total:.1f} tok/s")
    print(f"  step latency        : mean {statistics.fmean(step_times) * 1000:.2f} ms, "
          f"p50 {statistics.median(step_times) * 1000:.2f} ms, "
          f"max {max(step_times) * 1000:.2f} ms")
    print("=" * 96)


# --------------------------------------------------------------------------- #
#  cli
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mini-vLLM prefill/decode benchmark")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--weights", default=None, help="checkpoint .pt (default: random init)")
    p.add_argument("--batch-sizes", default="1,4,16,32")
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--gen-len", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--num-blocks", type=int, default=4096)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--skip-continuous", action="store_true")
    p.add_argument("--continuous-arrivals", type=int, default=64)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]

    cfg = EngineConfig(
        device=args.device,
        dtype=args.dtype,
        weights_path=args.weights,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_num_seqs=max(args.max_num_seqs, max(batch_sizes)),
        max_model_len=max(2048, args.prompt_len + args.gen_len + 8),
        # A single prefill pass per batch keeps the prefill/decode split clean.
        chunked_prefill=False,
        max_num_batched_tokens=max(4096, max(batch_sizes) * (args.prompt_len + 8)),
        time_steps=False,  # the benchmark times things itself
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
    )
    engine = LLMEngine(cfg)
    vocab_size = engine.tokenizer.vocab_size
    device = engine.core.device

    print(f"[mini-vllm bench] device={device} dtype={engine.core.model_runner.dtype} "
          f"vocab={vocab_size}")
    engine.warmup()

    results = bench_sweep(engine, batch_sizes, args.prompt_len, args.gen_len, args.repeats, vocab_size)
    print_sweep(results, args.prompt_len, args.gen_len, device)
    print_percentiles(results)
    print_memory(engine)

    if not args.skip_continuous:
        bench_continuous(engine, args.continuous_arrivals, args.prompt_len,
                         min(args.gen_len, 16), max(1, args.continuous_arrivals // 16), vocab_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
