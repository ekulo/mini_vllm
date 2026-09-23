"""Tokenizer for the hand-written model's word-level vocabulary.

The original ``手写/vocab_tokenizer.py`` has two quirks an inference engine must
not inherit:

* its ``BOS_ID``/``EOS_ID`` constants (1 and 2) disagree with the ids actually
  written into ``vocab_infer.json`` by ``create_vocab.py`` (``<UNK>``=1,
  ``<BOS>``=2, ``<EOS>``=3), so ``encode`` prepends ``<UNK>``;
* ``encode`` pads to ``max_seq_len``. Padding a prompt is fine for a fixed-shape
  training batch but wrong for generation -- the PAD positions would be
  attended to. We never pad.

Ids are therefore always looked up by *name* in ``word2id``.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, List

PAD, UNK, BOS, EOS = "<PAD>", "<UNK>", "<BOS>", "<EOS>"


class Tokenizer:
    def __init__(self, vocab_path: str) -> None:
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"vocabulary not found: {vocab_path}")
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.word2id: dict[str, int] = data["word2id"]
        self.id2word: dict[int, str] = {int(k): v for k, v in data["id2word"].items()}
        self.path = vocab_path

        self.pad_id = self.word2id.get(PAD, 0)
        self.unk_id = self.word2id.get(UNK, 1)
        self.bos_id = self.word2id.get(BOS)
        self.eos_id = self.word2id.get(EOS)
        self.special_ids = {i for i in (self.pad_id, self.unk_id, self.bos_id, self.eos_id) if i is not None}

    @property
    def vocab_size(self) -> int:
        return len(self.word2id)

    # ----------------------------------------------------------------- encode
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
        truncate: int | None = None,
    ) -> List[int]:
        """Whitespace tokenisation, exactly like the original, without padding."""
        tokens = text.split()
        ids = [self.word2id.get(t, self.unk_id) for t in tokens]
        if add_bos and self.bos_id is not None:
            ids.insert(0, self.bos_id)
        if add_eos and self.eos_id is not None:
            ids.append(self.eos_id)
        if not ids:  # empty prompt still needs one token to run through
            ids = [self.bos_id if self.bos_id is not None else self.unk_id]
        if truncate is not None and len(ids) > truncate:
            ids = ids[:truncate]
        return ids

    # ----------------------------------------------------------------- decode
    def decode_token(self, token_id: int) -> str:
        """Text for a single generated id; ``""`` for special tokens."""
        word = self.id2word.get(int(token_id), UNK)
        if word in (PAD, BOS, EOS, UNK):
            return ""
        return word

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        words: List[str] = []
        for i in ids:
            i = int(i)
            if skip_special and i in self.special_ids:
                continue
            words.append(self.id2word.get(i, UNK))
        return " ".join(words)

    def tokens_to_text(self, tokens: Iterable[str]) -> str:
        return " ".join(tokens)

    def __repr__(self) -> str:
        return f"<Tokenizer vocab={self.vocab_size} eos={self.eos_id}>"
