import random
import json
import torch
from typing import List, Union

# 严格按 ID 0,1,2,3 顺序定义，匹配你的encode/decode逻辑
PAD_TOKEN  = "<PAD>"
UNK_TOKEN  = "<UNK>"
BOS_TOKEN  = "<BOS>"
EOS_TOKEN  = "<EOS>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

# 固定ID，和token顺序一一对应
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

def build_random_vocab(vocab_size: int = 1000):
    # 特殊token先写入，保证id严格 0,1,2,3
    word2id = {}
    for idx, tok in enumerate(SPECIAL_TOKENS):
        word2id[tok] = idx
    
    # 普通词填充剩余词表容量
    normal_word_num = vocab_size - len(SPECIAL_TOKENS)
    normal_words = [f"word_{i}" for i in range(normal_word_num)]
    random.shuffle(normal_words)
    for w in normal_words:
        word2id[w] = len(word2id)
    
    # 反向id2word自动生成，包含0,1,2,3全部特殊id，彻底解决KeyError
    id2word = {v: k for k, v in word2id.items()}
    return word2id, id2word

# --- 保存词表的代码 ---
if __name__ == "__main__":
    # 1. 构建词表
    word2id, id2word = build_random_vocab(vocab_size=1000)
    
    # 校验特殊ID是否存在（调试用，可删除）
    print("校验特殊id映射：")
    print(f"id0: {id2word[0]}")
    print(f"id1: {id2word[1]}")
    print(f"id2: {id2word[2]}")
    print(f"id3: {id2word[3]}")

    # 2. 打包保存
    vocab_data = {
        "word2id": word2id,
        "id2word": id2word
    }
    
    # 3. 保存 JSON
    with open("vocab_infer.json", "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    print("词表已保存到 vocab_infer.json")
    print(f"词表总大小: {len(word2id)}")