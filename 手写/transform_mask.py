from multiheadattention import MultiHeadAttetion
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transform_mask(nn.Module):
    def __init__(self, hidden_size, head=8):
        super().__init__()
        self.atten = MultiHeadAttetion(hidden_size, head)
        self.li1 = nn.Linear(hidden_size, hidden_size * 2)
        self.li2 = nn.Linear(hidden_size * 2, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, src_mask=None,use_cache=False,cache=None):
        x_res = x
        x = self.norm1(x)

        x,cache = self.atten(x, mask=src_mask,use_cache=use_cache,cache=cache)
        x = x + x_res

        x_res = x
        x = self.norm2(x)
        x = self.li1(x)
        x = F.gelu(x)
        x = self.li2(x)
        x = x + x_res
        return x,cache