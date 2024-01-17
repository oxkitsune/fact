import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from typing import Unpack

GCNArgs = {"nfeat": 0, "nhid": 0, "dropout": 0, "nclass": 0}


class GCN(nn.Module):
    def __init__(self, **kwargs: Unpack[GCNArgs]):
        super(GCN, self).__init__()
        nfeat = kwargs["nfeat"]
        nhid = kwargs["nhid"]
        dropout = kwargs["dropout"]
        nclass = kwargs["nclass"]
        self.body = GCN_Body(nfeat, nhid, dropout)
        self.fc = nn.Linear(nhid, nclass)

    def forward(self, g, x):
        x = self.body(g, x)
        x = self.fc(x)
        return x


class GCN_Body(nn.Module):
    def __init__(self, **kwargs: Unpack[GCNArgs]):
        super(GCN_Body, self).__init__()
        nfeat = kwargs["nfeat"]
        nhid = kwargs["nhid"]
        dropout = kwargs["dropout"]
        nclass = kwargs["nclass"]
        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, nhid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x):
        x = F.relu(self.gc1(g, x))
        x = self.dropout(x)
        x = self.gc2(g, x)
        # x = self.dropout(x)
        return x
