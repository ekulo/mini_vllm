"""Request-level state: sampling parameters, sequence lifecycle, output record."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List


class SequenceStatus(enum.Enum):
    WAITING = "waiting"      # queued, not admitted yet
    RUNNING = "running"      # resident, owns KV blocks
    FINISHED_STOPPED = "stopped"    # hit EOS
    FINISHED_LENGTH = "length"      # hit max_tokens
    FINISHED_ABORTED = "aborted"    # client went away / explicit abort

    @property
    def is_finished(self) -> bool:
        return self in (
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH,
            SequenceStatus.FINISHED_ABORTED,
        )


@dataclass
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 0.0   # 0 => greedy, the model's original behaviour
    top_k: int = -1            # -1 => disabled
    top_p: float = 1.0         # 1.0 => disabled
    ignore_eos: bool = False
    seed: int | None = None

    def validate(self, max_model_len: int) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if self.max_tokens > max_model_len:
            raise ValueError(f"max_tokens {self.max_tokens} > max_model_len {max_model_len}")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")


class Sequence:
    """One in-flight generation.

    Bookkeeping that matters for the KV cache:

    ``num_computed_tokens``  tokens whose K/V is already written to the paged
                             cache (prompt + generated so far, minus one on
                             decode steps because the last token is fed again).
    ``num_prefill_tokens``   how many *leading* tokens must be in the cache
                             before this sequence may decode again. It starts
                             as the prompt length; preemption raises it to the
                             whole token list, because recompute has to replay
                             the generated tokens too. The sequence is
                             "prefilling" while ``num_computed_tokens`` is
                             below this.
    ``num_scheduled_tokens`` how many tokens *this* engine step will compute.
    """

    __slots__ = (
        "seq_id", "prompt_token_ids", "output_token_ids", "sampling_params",
        "status", "block_table", "num_computed_tokens", "num_prefill_tokens",
        "num_scheduled_tokens", "prompt_text", "meta",
    )

    def __init__(
        self,
        seq_id: int,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
        prompt_text: str = "",
        meta: dict | None = None,
    ) -> None:
        self.seq_id = seq_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.output_token_ids: List[int] = []
        self.sampling_params = sampling_params
        self.status = SequenceStatus.WAITING
        self.block_table: List[int] = []
        self.num_computed_tokens = 0
        self.num_prefill_tokens = len(self.prompt_token_ids)
        self.num_scheduled_tokens = 0
        self.prompt_text = prompt_text
        self.meta = meta or {}

    # ------------------------------------------------------------- helpers
    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    def get_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    def __len__(self) -> int:
        return self.num_prompt_tokens + len(self.output_token_ids)

    @property
    def is_prefilling(self) -> bool:
        """True while the cache still lacks tokens this sequence must replay."""
        return self.num_computed_tokens < self.num_prefill_tokens

    def __repr__(self) -> str:
        return (
            f"<Seq {self.seq_id} {self.status.value} len={len(self)} "
            f"computed={self.num_computed_tokens}/{self.num_prefill_tokens} "
            f"blocks={len(self.block_table)}>"
        )

    # ------------------------------------------------------------ mutation
    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def reset_for_recompute(self) -> None:
        """Preemption: the K/V is gone, so the whole token list must be re-run."""
        self.block_table = []
        self.num_computed_tokens = 0
        self.num_prefill_tokens = len(self.get_token_ids())
        self.num_scheduled_tokens = 0
        self.status = SequenceStatus.WAITING


@dataclass
class RequestOutput:
    """Incremental output pushed to a client after one engine step."""

    seq_id: int
    token_id: int
    output_token_ids: List[int] = field(default_factory=list)
    finished: bool = False
    finish_reason: str | None = None
    prompt_tokens: int = 0
    meta: dict = field(default_factory=dict)
