"""Engine configuration. One flat dataclass, everything tunable in one place."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

# ``手写/`` sits next to this package unless the user points somewhere else.
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "手写")


@dataclass
class EngineConfig:
    # ---------------------------------------------------------------- model
    model_dir: str = DEFAULT_MODEL_DIR
    vocab_file: str = "vocab_infer.json"
    weights_path: str | None = None  # .pt / .safetensors; None => random init

    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 8
    vocab_size: int = 1000

    # NOTE: there is no ``causal`` switch. The original model attends with
    # mask=None (prompt tokens see the future), which makes prefill and decode
    # disagree; the engine always applies a proper causal mask.
    # See models/handwritten.py for the full list of deliberate differences.

    # ------------------------------------------------------------ kv cache
    # Paged cache: memory is split into fixed-size blocks and each sequence
    # gets a block table (block ids), exactly like vLLM's KV block manager.
    block_size: int = 16
    num_blocks: int = 2048
    kv_dtype: str = "auto"  # auto | float16 | bfloat16 | float32

    # ----------------------------------------------------------- scheduler
    max_num_seqs: int = 32       # max sequences resident at once
    max_num_batched_tokens: int = 4096  # token budget per step (chunked prefill)
    max_model_len: int = 1024
    chunked_prefill: bool = True
    eos_token_id: int | None = None  # defaults to the vocab's <EOS>

    # ------------------------------------------------------------- runtime
    device: str = "auto"  # auto | cuda | cpu
    dtype: str = "auto"   # auto | float16 | bfloat16 | float32
    seed: int = 0
    # Synchronise CUDA around each forward pass so the per-stage timings in
    # /stats are real. Costs a little latency; turn off for max throughput.
    time_steps: bool = True

    # ------------------------------------------------------------ derived
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def resolve_dtype(self, device: torch.device) -> torch.dtype:
        name = self.dtype
        if name == "auto":
            # bf16 on GPU (same tensor-core speed as fp16 on Blackwell, more
            # headroom); fp32 on CPU because fp16 matmuls there are slow.
            name = "bfloat16" if device.type == "cuda" else "float32"
        table = {
            "float16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float": torch.float32,
        }
        if name not in table:
            raise ValueError(f"unknown dtype {name!r}")
        return table[name]

    def resolve_kv_dtype(self, device: torch.device, model_dtype: torch.dtype) -> torch.dtype:
        if self.kv_dtype == "auto":
            # Keep the cache in fp32 on CPU so correctness tests stay exact.
            return model_dtype if device.type == "cuda" else torch.float32
        return self.resolve_dtype(device)

    def cache_bytes(self) -> int:
        itemsize = 2 if self.kv_dtype in ("auto", "float16", "bfloat16", "half") else 4
        return 2 * self.num_layers * self.num_blocks * self.block_size * self.num_heads * self.head_dim * itemsize
