"""Before/after: the original hand-written generate loop vs the engine.

``手写/infra.py`` generates one request at a time and grows a *contiguous* KV
cache with ``torch.cat`` on every decode step, i.e. it reallocates and copies
the whole history once per token. This script rebuilds that loop faithfully
(same weights, same math, causal masking added so the outputs are comparable),
then runs the identical workload through the mini-vLLM engine and prints both.

    python tools/baseline_compare.py --device cuda --requests 8 --prompt-len 64 --gen-len 32
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
import torch.nn.functional as F  # noqa: E402


# --------------------------------------------------------------------------- #
#  the "before": contiguous cache, one request at a time
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _naive_layer(layer, x, cache):
    B, T, C = x.shape
    H, D = layer.atten.num_heads, layer.atten.head_dim

    h = layer.norm1(x)
    q, k, v = layer.atten.qkv(h).chunk(3, dim=-1)
    q = q.view(B, T, H, D).transpose(1, 2)
    k = k.view(B, T, H, D).transpose(1, 2)
    v = v.view(B, T, H, D).transpose(1, 2)

    if cache is not None:
        # This is the original's approach: a fresh concatenated tensor every
        # step, so the whole KV history is copied once per generated token.
        k = torch.cat([cache[0], k], dim=2)
        v = torch.cat([cache[1], v], dim=2)
    new_cache = (k, v)

    S = k.shape[2]
    scores = (q @ k.transpose(-1, -2)) * layer.atten.scale
    if T > 1:
        q_pos = torch.arange(S - T, S, device=x.device)[:, None]
        k_pos = torch.arange(S, device=x.device)[None, :]
        scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    y = (weights @ v).transpose(1, 2).reshape(B, T, H * D)

    x = x + y
    residual = x
    x = layer.li2(F.gelu(layer.li1(layer.norm2(x))))
    return x + residual, new_cache


@torch.no_grad()
def naive_generate(model, prompt_ids, max_tokens, device, eos_id=None):
    """The pre-engine loop: one sequence, growing cat-cache, greedy."""
    out: list[int] = []
    x = model.embedding(torch.tensor([prompt_ids], device=device))
    cache = None
    for _ in range(max_tokens):
        for i, layer in enumerate(model.decoder.layers):
            layer_cache = cache[i] if cache else None
            x, new_layer_cache = _naive_layer(layer, x, layer_cache)
            if cache is None:
                cache = [None] * len(model.decoder.layers)
            cache[i] = new_layer_cache
        logits = model.llm(x[:, -1, :])
        token = int(logits.argmax(dim=-1).item())
        out.append(token)
        if eos_id is not None and token == eos_id:
            break
        x = model.embedding(torch.tensor([[token]], device=device))
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--gen-len", type=int, default=32)
    ap.add_argument("--num-blocks", type=int, default=4096)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--compare-outputs", action="store_true", default=True)
    args = ap.parse_args()

    cfg = EngineConfig(
        device=args.device,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_num_seqs=max(32, args.requests),
        max_model_len=2048,
        chunked_prefill=False,
        time_steps=False,
    )
    engine = LLMEngine(cfg)
    device = engine.core.device
    model = engine.core.model_runner.model
    vocab = engine.tokenizer.vocab_size
    eng_gen = torch.Generator().manual_seed(11)
    prompts = [
        [2] + torch.randint(4, vocab, (args.prompt_len - 1,), generator=eng_gen).tolist()
        for _ in range(args.requests)
    ]
    sp = SamplingParams(max_tokens=args.gen_len, temperature=0.0, ignore_eos=True)

    print(f"mini-vLLM baseline comparison   device={device} dtype={engine.core.model_runner.dtype}")
    print(f"requests={args.requests}  prompt={args.prompt_len} tok  generate={args.gen_len} tok")

    def timed(fn):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return out, time.perf_counter() - t0

    # Warm both paths first: the very first CUDA call in a process pays for
    # context creation and cuBLAS/cuDNN initialisation, which would otherwise
    # be charged to whichever path ran first and make the comparison nonsense.
    naive_generate(model, prompts[0], 4, device)
    engine.warmup()
    if device.type == "cuda":
        torch.cuda.synchronize()

    # ---- before: one request at a time through the original-style loop ----
    baseline_out, baseline_time = timed(
        lambda: [naive_generate(model, p, args.gen_len, device) for p in prompts]
    )

    # ---- after: the same workload, continuously batched -------------------
    prompt_texts = [engine.tokenizer.decode(p[1:]) for p in prompts]
    results, engine_time = timed(lambda: engine.generate_batch(prompt_texts, sp))

    # For reference: the engine on a single stream, to expose its own
    # per-step overhead (it is higher than the naive loop's; the win comes
    # from batching, not from a cheaper single-sequence step).
    _, single_time = timed(lambda: engine.generate_batch(prompt_texts[:1], sp))

    generated = args.requests * args.gen_len
    total_tokens = args.requests * (args.prompt_len + args.gen_len)

    print()
    print("=" * 88)
    print(f"  {'':<30}{'naive loop (手写 style)':>27}{'mini-vLLM engine':>27}")
    print("-" * 88)
    print(f"  {'wall time (s)':<30}{baseline_time:>27.4f}{engine_time:>27.4f}")
    print(f"  {'output tokens/s':<30}{generated / baseline_time:>27.1f}{generated / engine_time:>27.1f}")
    print(f"  {'total tokens/s':<30}{total_tokens / baseline_time:>27.1f}{total_tokens / engine_time:>27.1f}")
    print(f"  {'per-request latency (s)':<30}{baseline_time / args.requests:>27.4f}{engine_time / args.requests:>27.4f}")
    print("-" * 88)
    print(f"  throughput speedup : {baseline_time / engine_time:.2f}x   "
          f"(KV cache: growing torch.cat -> {args.num_blocks} paged blocks x {args.block_size})")
    print(f"  engine, 1 request  : {single_time:.4f} s "
          f"({args.gen_len / single_time:.0f} tok/s) -- the batching win, not a cheaper step")
    print("=" * 88)

    if args.compare_outputs:
        mismatch = 0
        for want, got in zip(baseline_out, results):
            if want != got.token_ids[: len(want)]:
                mismatch += 1
        print(f"  output check: {args.requests - mismatch}/{args.requests} sequences match token-for-token")
        if mismatch:
            print("  (a mismatch here is a real bug -- the two paths must agree)")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
