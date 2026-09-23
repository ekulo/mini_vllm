from transform_mask import Transform_mask


import torch
import torch.nn as nn
import torch.nn.functional as F



class Decoder(nn.Module):
    def __init__(self, N , hidden_size=128,head=8):
        super().__init__()
        self.layers=nn.ModuleList(
            [Transform_mask(hidden_size,head) for _ in range(N)]
        )
    

    def forward(self,x,mask=None,use_cache=False,cache=None):
        for layer in self.layers:
            x,cache=layer(x,mask,use_cache,cache)
        return x,cache
    
