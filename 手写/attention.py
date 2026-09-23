import torch
import torch.nn as nn
import torch.nn.functional as F

class Attetion(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__
        self.qkv=nn.Linear(hidden_dim,hidden_dim*3)
        self.scale=hidden_dim**0.5


    def forward(self,x):
        B,T,C=x.shape
        q,k,v=self.qkv(x).chunk(3,dim=-1)

        atten=q @ k.transpose(-2,-1)*self.scale

        atte_weight=F.softmax(atten,dim=-1)

        y=atte_weight @ v
        return y
    
    