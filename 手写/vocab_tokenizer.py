import random
import json
import torch
from typing import List, Union


SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN = SPECIAL_TOKENS

PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3



class TransTokenizer:
    def __init__(self, word2id: dict, id2word: dict, max_seq_len: int):
        self.word2id = word2id
        self.id2word = id2word
        self.max_seq_len = max_seq_len

    def encode(self, tokens: List[str], add_bos=True, add_eos=True) -> torch.Tensor:
        ids = []
        if add_bos:
            ids.append(BOS_ID)
        ids.extend([self.word2id.get(t, UNK_ID) for t in tokens])
        if add_eos:
            ids.append(EOS_ID)
        # 截断+padding
        if len(ids) > self.max_seq_len:
            ids = ids[:self.max_seq_len]
        ids += [PAD_ID] * (self.max_seq_len - len(ids))
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def decode(self, ids_tensor: torch.Tensor) -> str:
        ids = ids_tensor.squeeze().cpu().tolist()
        res = []
        for idx in ids:
            w = self.id2word[idx]
            if w == EOS_TOKEN:
                # 碰到结束符，终止解码
                break
            if w in (PAD_TOKEN, BOS_TOKEN):
                continue
            res.append(w)
        return " ".join(res)

    def save(self, path="vocab_infer.json"):
        json.dump({"word2id": self.word2id, "id2word": self.id2word}, open(path, "w", encoding="utf-8"), indent=2)

    @staticmethod
    def load(path="vocab_infer.json", max_seq_len=32):
        data = json.load(open(path, "r", encoding="utf-8"))
        id2word = {int(k): v for k, v in data["id2word"].items()}
        return TransTokenizer(data["word2id"],id2word , max_seq_len)