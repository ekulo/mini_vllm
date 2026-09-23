from transform import Transform

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, N , hidden_size=128):
        super().__init__()
        self.layers=nn.ModuleList(
            [Transform(hidden_size) for _ in range(N)]
        )
    

    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return x