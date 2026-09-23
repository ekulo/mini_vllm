from attention import Attetion

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transform(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.atten=Attetion(hidden_size)
        self.li1=nn.Linear(hidden_size,hidden_size*2)
        self.li2=nn.Linear(hidden_size*2,hidden_size)
    

    def forward(self, x):
        x_nor=F.normalize(x,dim=-1)
        x_atten=self.atten(x_nor)
        x_li1=self.li1(x_atten)
        x_li2=self.li2(x_li1)
        y=x+F.normalize(x_li2,dim=-1)
        return y