"""Model runner: owns the model and turns sequences into logits and tokens.

This is the "modelrun" stage. It does exactly three things:

1. ``build_metadata`` -- pack the scheduled sequences into padded tensors plus
   the paged-attention bookkeeping (block tables and slot mapping).
2. ``execute``        -- one forward pass, returning ``(B, vocab)`` logits.
3. ``sample``         -- greedy / temperature / top-k / top-p, one token each.

It deliberately calls the model directly (no worker process, no IPC): the model
is tiny, so an in-process runner is both faster and far easier to read.
"""

from __future__ import annotations

import math
import os

import torch

from .config import EngineConfig
from .kv_manager import KVCacheManager
from .models.handwritten import AttnMetadata, HandwrittenForCausalLM
from .sequence import Sequence


# --------------------------------------------------------------------------- #
#  sampling
# --------------------------------------------------------------------------- #
def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filtering kept as a helper so the sampler reads linearly."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    drop = cum > top_p
    drop[1:] = drop[:-1].clone()  # keep the token that crosses the threshold
    drop[0] = False
    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
    return torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)


class Sampler:
    def __init__(self, config: EngineConfig, device: torch.device) -> None:
        self.device = device
        gen_device = "cuda" if device.type == "cuda" else "cpu"
        self.generator = torch.Generator(device=gen_device)
        self.generator.manual_seed(config.seed)

    @torch.inference_mode()
    def __call__(self, logits: torch.Tensor, seqs: list[Sequence]) -> list[int]:
        # Greedy is the common case (and the model's original behaviour): do it
        # for the whole batch in one op so there is a single device sync rather
        # than one ``.item()`` per sequence.
        if all(s.sampling_params.temperature <= 1e-5 for s in seqs):
            return logits.argmax(dim=-1).tolist()

        out: list[int] = []
        for i, seq in enumerate(seqs):
            sp = seq.sampling_params
            row = logits[i].float()
            row = row / sp.temperature
            if sp.top_k > 0:
                k = min(sp.top_k, row.numel())
                kth = torch.topk(row, k).values[-1]
                row = row.masked_fill(row < kth, float("-inf"))
            row = _top_p_filter(row, sp.top_p)

            probs = torch.softmax(row, dim=-1)
            if not bool(torch.isfinite(probs).all()) or float(probs.sum()) <= 0.0:
                out.append(int(torch.argmax(logits[i]).item()))  # degenerate guard
                continue
            tok = torch.multinomial(probs, num_samples=1, generator=self.generator)
            out.append(int(tok.item()))
        return out


# --------------------------------------------------------------------------- #
#  runner
# --------------------------------------------------------------------------- #
class ModelRunner:
    def __init__(self, config: EngineConfig, kv_manager: KVCacheManager) -> None:
        self.config = config
        self.kv_manager = kv_manager
        self.device = kv_manager.device
        self.dtype = config.resolve_dtype(self.device)
        self.block_size = config.block_size
        self.model = self._build_model()
        self.sampler = Sampler(config, self.device)

    # ----------------------------------------------------------- construction
    def _build_model(self) -> HandwrittenForCausalLM:
        cfg = self.config
        model = HandwrittenForCausalLM(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
        )
        if cfg.weights_path:
            model.load_state_dict(_load_state_dict(cfg.weights_path), strict=True)
        else:
            model.init_weights(seed=cfg.seed)
        model = model.to(device=self.device, dtype=self.dtype)
        model.eval()
        return model

    # --------------------------------------------------------- metadata build
    def build_metadata(self, seqs: list[Sequence]) -> AttnMetadata:
        """Pack ``seqs`` (all scheduled for this step) into padded tensors.

        Tensors are assembled on the CPU and moved once at the end: per-row
        ``torch.tensor(..., device="cuda")`` calls would each be a synchronous
        host->device copy, which for a batch of 32 is 60+ pipeline stalls.
        """
        if not seqs:
            raise ValueError("build_metadata called with no sequences")

        bs = self.block_size
        B = len(seqs)
        L = max(s.num_scheduled_tokens for s in seqs)
        if L <= 0:
            raise ValueError("a scheduled sequence has no tokens to compute")

        input_ids = torch.zeros(B, L, dtype=torch.long)
        slot_mapping = torch.full((B, L), -1, dtype=torch.long)
        context_lens = torch.empty(B, dtype=torch.long)
        query_lens = torch.empty(B, dtype=torch.long)

        # Attention reads ceil(seq_len / block_size) blocks per sequence; the
        # tables must be at least that wide. Rows of shorter sequences are
        # padded with block 0 and masked out on read.
        widest = max(len(s.block_table) for s in seqs)
        longest = max(s.num_computed_tokens + s.num_scheduled_tokens for s in seqs)
        width = max(widest, math.ceil(longest / bs), 1)
        block_tables = torch.zeros(B, width, dtype=torch.long)

        for i, seq in enumerate(seqs):
            # ``get_token_ids`` is prompt + generated so far. For a fresh request
            # that is just the prompt; after a preemption it re-runs both, which
            # is exactly what a recompute prefill should do.
            start = seq.num_computed_tokens
            n = seq.num_scheduled_tokens
            tokens = seq.get_token_ids()[start:start + n]

            input_ids[i, :n] = torch.tensor(tokens, dtype=torch.long)

            bt = torch.tensor(seq.block_table, dtype=torch.long)
            block_tables[i, : bt.numel()] = bt

            pos = torch.arange(start, start + n)
            slot_mapping[i, :n] = bt[pos // bs] * bs + (pos % bs)

            context_lens[i] = start
            query_lens[i] = n

        dev = self.device
        return AttnMetadata(
            input_ids=input_ids.to(dev, non_blocking=True),
            slot_mapping=slot_mapping.to(dev, non_blocking=True),
            block_tables=block_tables.to(dev, non_blocking=True),
            context_lens=context_lens.to(dev, non_blocking=True),
            query_lens=query_lens.to(dev, non_blocking=True),
            block_size=bs,
        )

    # ------------------------------------------------------------- execution
    @torch.inference_mode()
    def execute(self, seqs: list[Sequence]) -> torch.Tensor:
        """One forward pass over the scheduled sequences -> ``(B, vocab)`` logits."""
        meta = self.build_metadata(seqs)
        return self.model(meta, self.kv_manager.cache)

    @torch.inference_mode()
    def sample(self, logits: torch.Tensor, seqs: list[Sequence]) -> list[int]:
        return self.sampler(logits, seqs)

    def run(self, seqs: list[Sequence]) -> list[int]:
        """``execute`` + ``sample`` -- the call the engine core makes."""
        return self.sample(self.execute(seqs), seqs)

    # ----------------------------------------------------------------- misc
    @torch.inference_mode()
    def warmup(self, tokens: int = 8) -> None:
        """Run one tiny batch so CUDA kernels/libraries are loaded before serving."""
        from .sequence import SamplingParams

        seq = Sequence(-1, list(range(1, tokens + 1)), SamplingParams(max_tokens=1))
        self.kv_manager.allocate_slots(seq, tokens)
        seq.num_scheduled_tokens = tokens
        try:
            self.run([seq])
        finally:
            self.kv_manager.free(seq)

    def stats(self) -> dict:
        cfg = self.config
        return {
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "parameters": self.model.num_parameters(),
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "num_heads": cfg.num_heads,
            "vocab_size": cfg.vocab_size,
            "causal_attention": True,
            "weights": os.path.basename(cfg.weights_path) if cfg.weights_path else "random-init",
        }

    def to(self, *args, **kwargs):  # pragma: no cover - convenience for tests
        self.model = self.model.to(*args, **kwargs)
        return self


def _load_state_dict(path: str) -> dict:
    """Load a checkpoint, tolerating a wrapping ``{"model": ...}`` container."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    if not isinstance(obj, dict):
        raise ValueError(f"{path} does not contain a state dict")
    return obj
