"""Shared helpers for the test-suite.

The reference forward pass below is written from scratch with plain tensor ops
(no paged cache, no attention metadata) so it is a genuinely independent check
of the engine's incremental prefill/decode path.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from mini_vllm import EngineConfig


def tiny_config(**overrides) -> EngineConfig:
    cfg = EngineConfig(
        device="cpu",
        dtype="float32",
        num_blocks=64,
        block_size=8,
        max_num_seqs=8,
        max_num_batched_tokens=256,
        max_model_len=128,
        chunked_prefill=True,
        time_steps=False,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@torch.no_grad()
def naive_forward(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Plain causal forward over one sequence, ``(1, T) -> (1, T, vocab)``."""
    x = model.embedding(input_ids)
    B, T, _ = x.shape
    H = model.decoder.layers[0].atten.num_heads
    D = model.decoder.layers[0].atten.head_dim
    scale = 1.0 / math.sqrt(D)

    tril = torch.tril(torch.ones(T, T, dtype=torch.bool))

    for layer in model.decoder.layers:
        h = layer.norm1(x)
        q, k, v = layer.atten.qkv(h).chunk(3, dim=-1)
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        scores = (q @ k.transpose(-1, -2)) * scale
        scores = scores.masked_fill(~tril, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        y = (weights @ v).transpose(1, 2).reshape(B, T, H * D)
        x = x + y

        residual = x
        x = layer.li2(F.gelu(layer.li1(layer.norm2(x))))
        x = x + residual

    return model.llm(x)
