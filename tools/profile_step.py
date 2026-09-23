"""Where does a decode step actually spend its time?

    python tools/profile_step.py --device cuda --batch-size 8 --prompt-len 128

Prints a phase breakdown (metadata build / forward / sampling) and a torch
profiler table. Useful because for a model this small the engine's cost is
dominated by per-op dispatch and host<->device synchronisation, not by FLOPs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mini_vllm import EngineConfig, LLMEngine, SamplingParams  # noqa: E402
from mini_vllm._env import ensure_local_libs  # noqa: E402

ensure_local_libs()

import torch  # noqa: E402


def _time(fn, repeats: int, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=50)
    args = ap.parse_args()

    cfg = EngineConfig(
        device=args.device,
        num_blocks=4096,
        block_size=16,
        max_num_seqs=max(64, args.batch_size),
        max_model_len=2048,
        chunked_prefill=False,
        max_num_batched_tokens=8192,
        time_steps=False,
    )
    eng = LLMEngine(cfg)
    core = eng.core
    device = core.device

    sp = SamplingParams(max_tokens=1024, temperature=0.0, ignore_eos=True)
    for _ in range(args.batch_size):
        core.add_request([2] + list(range(4, args.prompt_len + 1)), sp)

    core.step()  # prefill
    print(f"device={device} dtype={core.model_runner.dtype} batch={args.batch_size} "
          f"prompt={args.prompt_len}")

    # Everything is decoding now. Each phase is timed from that stable state;
    # ``schedule()`` is idempotent for an unchanged sequence, so it can be
    # replayed to rebuild metadata without consuming tokens.
    seqs = list(core.scheduler.running)
    runner = core.model_runner

    def scheduled_metadata():
        core.scheduler.schedule()
        return runner.build_metadata(list(core.scheduler.running))

    meta = scheduled_metadata()
    logits = runner.model(meta, core.kv_manager.cache)

    meta_ms = _time(scheduled_metadata, args.repeats, device)
    forward_ms = _time(lambda: runner.model(meta, core.kv_manager.cache), args.repeats, device)
    sample_ms = _time(lambda: runner.sample(logits, seqs), args.repeats, device)
    step_ms = _time(core.step, args.repeats, device)

    total = step_ms
    print()
    print(f"  decode step          : {total:8.3f} ms   ({args.batch_size} tokens -> "
          f"{args.batch_size / total * 1000:.1f} tok/s)")
    print(f"    build_metadata     : {meta_ms:8.3f} ms   {meta_ms / total * 100:5.1f}%")
    print(f"    forward            : {forward_ms:8.3f} ms   {forward_ms / total * 100:5.1f}%")
    print(f"    sample             : {sample_ms:8.3f} ms   {sample_ms / total * 100:5.1f}%")
    print(f"    rest (sched + python): {total - meta_ms - forward_ms - sample_ms:7.3f} ms")

    if device.type == "cuda":
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            for _ in range(10):
                runner.model(meta, core.kv_manager.cache)
            torch.cuda.synchronize()
        print()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))

    # How many host<->device syncs happen per forward pass?
    print("Evidence for the launch/sync bound: kernels per forward pass ~")
    print(f"  {sum(1 for _ in core.model_runner.model.decoder.layers)} layers x "
          f"(qkv + 2 scatter + 2 gather + mask + sdpa + 2 mlp + 2 norm + adds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
