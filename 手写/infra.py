from decoder import Decoder
from vocab_tokenizer import TransTokenizer
from cudalinear import CudaLinear
from cudalinearfused import CudaLinearFused

import torch
import torch.nn as nn


class Infra(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder=Decoder(N=4)
        self.tokenizer=TransTokenizer.load()

        self.vocab_size=len(self.tokenizer.word2id)
        self.hidden_size=128

        self.embedding=torch.nn.Embedding(self.vocab_size,self.hidden_size)
        self.llm=torch.nn.Linear(self.hidden_size,self.vocab_size)
        #self.llm=CudaLinear(self.hidden_size,self.vocab_size)
        #self.llm=CudaLinearFused(self.hidden_size,self.vocab_size)



    def generate(self,text,max_token=20):
        tokens = text.split()
        ids=self.tokenizer.encode(tokens).to("cuda")
        
        cache=None
        for _ in range(max_token):
            if cache is None: 
                x=self.embedding(ids)
            else:
                x=self.embedding(ids[:,-1:])
            x,cache=self.decoder(x,mask=None,use_cache=True,cache=cache)

            
            logits=self.llm(x[:,-1,:])
            next_id=logits.argmax(dim=-1,keepdim=True)

            ids=torch.cat([ids,next_id],dim=-1)

            if next_id.item() == self.tokenizer.word2id["<EOS>"] :
                break
        
        ans = self.tokenizer.decode(ids)
        return ans