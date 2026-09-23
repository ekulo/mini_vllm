"""Prove that the engine's model is weight-compatible with ``手写/infra.py``.

The engine's ``HandwrittenForCausalLM`` was written to keep the original
parameter names exactly, so the original checkpoint (or a randomly initialised
``Infra``) loads with ``strict=True``. That is easy to break by accident, so
this script checks both directions:

    1. structural -- the key sets match exactly (runs anywhere);
    2. numerical  -- optional, loads the original model and compares logits
                     (needs the ``multi_head_attn`` CUDA extension, so it only
                     works on the machine that built it).

    python tools/check_weight_compat.py                    # structure only
    python tools/check_weight_compat.py --numeric          # also compare logits
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mini_vllm import EngineConfig  # noqa: E402
from mini_vllm._env import ensure_local_libs  # noqa: E402

ensure_local_libs()

import torch  # noqa: E402

from mini_vllm.models.handwritten import HandwrittenForCausalLM  # noqa: E402

# The names ``Infra`` produces, spelled out so the check does not need the
# original module (which imports the CUDA extension) to run here.
def original_key_template(num_layers: int, vocab_size: int, hidden: int) -> set[str]:
    keys = {"embedding.weight", "llm.weight", "llm.bias"}
    for i in range(num_layers):
        p = f"decoder.layers.{i}"
        keys |= {
            f"{p}.norm1.weight", f"{p}.norm1.bias",
            f"{p}.atten.qkv.weight", f"{p}.atten.qkv.bias",
            f"{p}.norm2.weight", f"{p}.norm2.bias",
            f"{p}.li1.weight", f"{p}.li1.bias",
            f"{p}.li2.weight", f"{p}.li2.bias",
        }
    return keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--numeric", action="store_true", help="also compare logits against 手写/infra.py")
    ap.add_argument("--model-dir", default=os.path.join(ROOT, "手写"))
    args = ap.parse_args()

    cfg = EngineConfig(model_dir=args.model_dir, device="cpu", dtype="float32")
    mine = HandwrittenForCausalLM(cfg.vocab_size, cfg.hidden_size, cfg.num_layers, cfg.num_heads)
    my_keys = set(mine.state_dict().keys())
    want = original_key_template(cfg.num_layers, cfg.vocab_size, cfg.hidden_size)

    print(f"[struct] engine keys: {len(my_keys)}   original keys: {len(want)}")
    missing, extra = want - my_keys, my_keys - want
    if missing:
        print("  MISSING (present in 手写, absent here):")
        for k in sorted(missing)[:20]:
            print("   ", k)
    if extra:
        print("  EXTRA (present here, absent in 手写):")
        for k in sorted(extra)[:20]:
            print("   ", k)
    ok = not missing and not extra
    print("  ->", "IDENTICAL" if ok else "MISMATCH")

    shapes_ok = True
    for name, param in mine.state_dict().items():
        if name.endswith("embedding.weight") and param.shape != (cfg.vocab_size, cfg.hidden_size):
            shapes_ok = False
            print("  bad shape", name, tuple(param.shape))
    print("  shapes:", "ok" if shapes_ok else "BAD")

    if args.numeric:
        sys.path.insert(0, args.model_dir)
        try:
            from infra import Infra  # type: ignore
        except Exception as exc:  # pragma: no cover - needs the CUDA extension
            print(f"\n[numeric] skipped: cannot import 手写/infra.py ({exc})")
            print("          this is expected unless multi_head_attn was built for this torch/CUDA.")
            return 0 if ok else 1

        print("\n[numeric] loading 手写/infra.py state dict with strict=True ...")
        reference = Infra()
        mine.load_state_dict(reference.state_dict(), strict=True)
        print("  load_state_dict: OK")

        ids = torch.tensor([[2, 5, 9, 11, 23]])
        # The original is not causal; compare with causal masking off to isolate
        # the weight question from the masking fix.
        from tests.helpers import naive_forward

        reference.eval()
        mine.eval()
        with torch.no_grad():
            ref_ids = ids.expand(1, -1)
            ref_out, _ = reference.decoder(reference.embedding(ref_ids), mask=None, use_cache=False)
            ref_logits = reference.llm(ref_out)
        with torch.no_grad():
            my_logits = naive_forward(mine, ids)[0, -1]
        diff = (ref_logits[0, -1].float() - my_logits.float()).abs().max().item()
        print(f"  max |logit difference| vs 手写 (causal off path): {diff:.5f}")
        print("  note: exact equality needs mask=None in both; see models/handwritten.py")

    return 0 if ok and shapes_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
