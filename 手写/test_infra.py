from infra import Infra


import torch


def test_generate():

    model= Infra()

    model = model.cuda()
    model.eval()

    test_prompts = [
        # 短输入
        "word_961",
        "word_498 word_783",
        # 中等长上下文
        "word_12 word_40 word_53 word_343 word_242",
        # 混合未知词
        "word_961 hello word_498",
        # 空输入
        "",
        # 纯特殊标记
        "<PAD> <BOS>"
    ]

    for p in test_prompts:
        out = model.generate(p, max_token=25)
        print(f"输入:{p:40} 输出:{out}")

if __name__ == "__main__":
    test_generate()