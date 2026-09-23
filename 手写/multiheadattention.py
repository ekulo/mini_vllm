import torch
import torch.nn as nn
import torch.nn.functional as F

import multi_head_attn

class MultiHeadAttetion(nn.Module):
    def __init__(self,hidden_dim,head):
        super().__init__()
        self.head_dim=hidden_dim//head
        self.head=head
        self.qkv=nn.Linear(hidden_dim,hidden_dim*3)
        self.scale =self.head_dim**0.5

    def forward(self,x,mask=None,cache=None,use_cache=False):
        B,T,C=x.shape
    
        q,k,v=self.qkv(x).chunk(3,dim=-1)
        q,k,v=map(lambda t: t.reshape(B,T,self.head,self.head_dim).transpose(1,2),[q,k,v])

        if use_cache and cache is not None:
            k_cacahe , v_cache=cache
            k=torch.cat([k_cacahe,k],dim=2)
            v=torch.cat([v_cache,v],dim=2)
        new_cache=(k,v) if use_cache else None
        
        y=multi_head_attn(q,k,v,mask,self.scale)

        # atten_sore=q @ k.transpose(-2,-1)/self.scale 
        # if mask is not None :
        #     atten_sore=atten_sore.masked_fill(mask==0,float('inf'))
        # atten_weight=F.softmax(atten_sore,dim=-1)
        # y=atten_weight @ v
        
        return y.transpose(1,2).reshape(B,T,C),new_cache