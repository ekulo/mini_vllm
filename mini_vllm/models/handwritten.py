"""The hand-written Transformer from ``手写/``, rewritten for paged inference.

Differences from the original, and why
--------------------------------------
1. **Attention reads a paged KV cache.** ``手写/multiheadattention.py`` keeps a
   contiguous ``(k, v)`` pair per sequence and grows it with ``torch.cat``,
   which forces a fresh allocation every decode step. Here K/V are *written
   into* and *gathered from* the block pool owned by :class:`KVCacheManager`.

2. **Causal masking.** The original always calls the attention op with
   ``mask=None``, so prompt tokens attend to future tokens. Prefill and decode
   then disagree about the same position. We build a real causal mask.

3. **The custom CUDA kernel is not used.** ``multi_head_attn.cu`` has no
   ``__syncthreads()`` between the softmax writes and reads (a race) and its
   ``out_off`` uses ``threadIdx.x % D``, which only writes the right value for
   some threads. Correctness first: this file uses
   ``torch.nn.functional.scaled_dot_product_attention``, which is also what the
   commented-out fallback in the original does.

4. **No ``torch.cat`` / ``F.normalize``.** ``手写/transform.py`` (the encoder
   variant, unused by ``Infra``) normalises activations; the decoder variant
   ``transform_mask.py`` -- which ``Infra`` actually uses -- does not, and we
   mirror that exactly.

Parameter names are identical to the original ``Infra`` model, so

    model.load_state_dict(original_infra.state_dict(), strict=True)

works unchanged (see ``tools/check_weight_compat.py``):

    embedding.weight
    decoder.layers.{i}.norm1.{weight,bias}
    decoder.layers.{i}.atten.qkv.{weight,bias}
    decoder.layers.{i}.norm2.{weight,bias}
    decoder.layers.{i}.li1.{weight,bias}
    decoder.layers.{i}.li2.{weight,bias}
    llm.{weight,bias}

The model has **no positional encoding at all** -- neither a learned embedding
nor RoPE -- so token positions never enter the computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Cache tensor indices.
K_CACHE, V_CACHE = 0, 1


@dataclass
class AttnMetadata:
    """Everything a forward pass needs, in a padded ``(batch, seq)`` layout.

    Working on padded batches (rather than vLLM's flattened/varlen layout) is
    what keeps this file short: one ``forward`` handles prefill, chunked
    prefill and decode with no branching beyond the mask.

    ``context_lens[b]``  tokens already in the cache before this step
    ``query_lens[b]``    real (non-padding) query tokens for sequence ``b``

    Everything under "derived" is computed once per step in ``__post_init__``
    and shared by every layer. That is not just tidier -- rebuilding the mask
    per layer costs four times the ops, and the ``.item()``/``.any()`` calls
    that feed it would each force a host<->device synchronisation per layer.
    """

    input_ids: torch.Tensor      # (B, L) int64
    slot_mapping: torch.Tensor   # (B, L) int64, flat KV slot per token, -1 = pad
    block_tables: torch.Tensor   # (B, NB) int64
    context_lens: torch.Tensor   # (B,) int64
    query_lens: torch.Tensor     # (B,) int64
    block_size: int

    # ---- derived ----
    attn_mask: torch.Tensor = None       # (B, 1, L, S) bool, True = attend
    num_kv_blocks: int = 0               # blocks of history to gather
    write_mask: torch.Tensor = None      # (B*L,) bool, which tokens are written
    write_slots: torch.Tensor = None     # (n,) int64, their cache slots

    def __post_init__(self) -> None:
        if self.attn_mask is not None:
            return
        device = self.input_ids.device
        B, L = self.input_ids.shape

        seq_lens = self.context_lens + self.query_lens
        self.num_kv_blocks = max(1, math.ceil(int(seq_lens.max().item()) / self.block_size))
        S = self.num_kv_blocks * self.block_size

        q_pos = torch.arange(L, device=device)[None, :, None]     # (1, L, 1)
        k_pos = torch.arange(S, device=device)[None, None, :]     # (1, 1, S)
        ctx = self.context_lens[:, None, None]                    # (B, 1, 1)
        qlen = self.query_lens[:, None, None]                     # (B, 1, 1)
        allowed = (k_pos <= ctx + q_pos) & (k_pos < ctx + qlen) & (q_pos < qlen)
        # Padding rows would otherwise softmax over an all -inf row -> NaN.
        # Give each padded row exactly one visible key instead.
        allowed = allowed | ((q_pos >= qlen) & (k_pos == 0))
        self.attn_mask = allowed[:, None]                         # (B, 1, L, S)

        self.write_mask = (self.slot_mapping >= 0).reshape(-1)
        self.write_slots = self.slot_mapping.reshape(-1)[self.write_mask]

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]

    @property
    def seq_lens(self) -> torch.Tensor:
        """Total KV length per sequence *after* this step's writes."""
        return self.context_lens + self.query_lens

    def num_tokens(self) -> int:
        return int(self.query_lens.sum().item())


class PagedAttention(nn.Module):
    """Multi-head attention over a paged KV cache. Same weights as ``手写``."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        # The CUDA kernel in 手写 does `dot / scale` with scale = sqrt(head_dim),
        # i.e. it divides. We reproduce that (scaled_dot_product_attention takes
        # a multiplier, hence the reciprocal).
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)

    def forward(
        self,
        x: torch.Tensor,
        meta: AttnMetadata,
        kv_cache: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, H, D).transpose(1, 2)   # (B, H, L, D)
        k = k.reshape(-1, H, D)                  # (B*L, H, D)
        v = v.reshape(-1, H, D)

        cache_k = kv_cache[K_CACHE, layer_idx]   # (num_blocks, block_size, H, D)
        cache_v = kv_cache[V_CACHE, layer_idx]

        # ---- 1. scatter this step's K/V into their paged slots ------------
        if meta.write_slots.numel():
            cache_k.view(-1, H, D).index_copy_(0, meta.write_slots, k[meta.write_mask])
            cache_v.view(-1, H, D).index_copy_(0, meta.write_slots, v[meta.write_mask])

        # ---- 2. gather each sequence's whole history through its blocks ---
        bt = meta.block_tables[:, : meta.num_kv_blocks]            # (B, nb)
        S = meta.num_kv_blocks * meta.block_size
        # (B, nb, block_size, H, D) -> (B, H, S, D)
        big_k = cache_k[bt].reshape(B, S, H, D).transpose(1, 2)
        big_v = cache_v[bt].reshape(B, S, H, D).transpose(1, 2)

        # ---- 3. attend (causal + padding mask built once per step) --------
        y = F.scaled_dot_product_attention(q, big_k, big_v, meta.attn_mask, scale=self.scale)
        return y.transpose(1, 2).reshape(B, L, H * D)


class HandwrittenDecoderLayer(nn.Module):
    """Pre-norm block, mirroring ``手写/transform_mask.py`` exactly."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.atten = PagedAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.li1 = nn.Linear(hidden_size, hidden_size * 2)
        self.li2 = nn.Linear(hidden_size * 2, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(
        self, x: torch.Tensor, meta: AttnMetadata, kv_cache: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        residual = x
        x = self.atten(self.norm1(x), meta, kv_cache, layer_idx)
        x = x + residual

        residual = x
        x = self.li1(self.norm2(x))
        x = F.gelu(x)
        x = self.li2(x)
        return x + residual


class HandwrittenDecoder(nn.Module):
    """Stack of decoder layers; keeps the ``layers`` ModuleList name of ``手写``."""

    def __init__(self, num_layers: int, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            HandwrittenDecoderLayer(hidden_size, num_heads) for _ in range(num_layers)
        )

    def forward(self, x: torch.Tensor, meta: AttnMetadata, kv_cache: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, meta, kv_cache, i)
        return x


class HandwrittenForCausalLM(nn.Module):
    """``embedding -> decoder -> lm head``, same shapes/names as ``Infra``."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.decoder = HandwrittenDecoder(num_layers, hidden_size, num_heads)
        self.llm = nn.Linear(hidden_size, vocab_size)

    def forward(self, meta: AttnMetadata, kv_cache: torch.Tensor) -> torch.Tensor:
        """Return logits for the last real query token of every sequence."""
        x = self.embedding(meta.input_ids)
        x = self.decoder(x, meta, kv_cache)
        # Only the last valid position per sequence is ever sampled from.
        idx = (meta.query_lens - 1).clamp_min(0)
        last = x[torch.arange(meta.batch_size, device=x.device), idx]
        return self.llm(last)

    # ------------------------------------------------------------------ util
    @torch.no_grad()
    def init_weights(self, seed: int = 0) -> None:
        """Deterministic random init, for running without a checkpoint."""
        gen = torch.Generator(device="cpu").manual_seed(seed)
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p, generator=gen)
            else:
                nn.init.zeros_(p)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
