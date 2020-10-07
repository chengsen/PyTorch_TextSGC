import torch.nn as nn
import torch as th
from torch.nn.modules.module import Module


class SGC(Module):
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()
        self.W = nn.Linear(nfeat, nclass)
        th.nn.init.xavier_normal_(self.W.weight)

    def forward(self, x):
        return self.W(x)
