"""Tiny contact classifier, independent of frozen OpenVLA."""
import torch
from torch import nn

class TinyGRU(nn.Module):
    def __init__(self,input_dim,hidden_dim=32):
        super().__init__()
        self.rnn=nn.GRU(input_dim,hidden_dim,batch_first=True)
        self.head=nn.Linear(hidden_dim,1)
    def forward(self,x):
        h,_=self.rnn(x)
        return self.head(h[:,-1]).squeeze(-1)
