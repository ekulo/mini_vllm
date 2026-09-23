from decoder import Decoder

import torch

decoder=Decoder(N=3,hidden_size=128,head=8)
b=2
promt=10
hidden_dim=128

x=torch.randn(b,promt,hidden_dim)

x_out,cache=decoder(x,mask=None,use_cache=True,cache=None)

gen_len=20
x=x_out[:,-1:,:]

generated_ids = [[] for _ in range(b)]
for i in range (gen_len):

    x,cache=decoder(
        x,
        mask=None,
        use_cache=True,
        cache=cache
    )
    