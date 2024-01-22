import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import Callable


class HGNNAC(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: Callable,
        dropout: float,
        num_heads: int,
    ):
        super(HGNNAC, self).__init__()

        self.dropout = dropout
        self.attentions = [
            _AttentionLayer(input_dim, hidden_dim, activation, dropout)
            for _ in range(num_heads)
        ]

    def forward(self, biased_adj, emb_dest, emb_src, feature_src):
        biased_adj = F.dropout(biased_adj, self.dropout, training=self.training)
        x = torch.cat(
            [
                att(biased_adj, emb_dest, emb_src, feature_src).unsqueeze(0)
                for att in self.attentions
            ],
            dim=0,
        )

        return torch.mean(x, dim=0, keepdim=False)


class _AttentionLayer(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, activation: Callable, dropout: float
    ):
        super(_AttentionLayer, self).__init__()
        self.dropout = dropout
        self.activation = activation

        self.W = nn.Parameter(
            nn.init.xavier_normal_(
                torch.Tensor(input_dim, hidden_dim).type(torch.float32),
                gain=np.sqrt(2.0),
            ),
            requires_grad=True,
        )
        self.W2 = nn.Parameter(
            nn.init.xavier_normal_(
                torch.Tensor(hidden_dim, hidden_dim).type(torch.float32),
                gain=np.sqrt(2.0),
            ),
            requires_grad=True,
        )

        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, biased_adj, emb_dest, emb_src, feature_src):
        h_1 = torch.mm(emb_src, self.W)
        h_2 = torch.mm(emb_dest, self.W)

        e = self.leakyrelu(torch.mm(torch.mm(h_2, self.W2), h_1.t()))
        zero_vec = -9e15 * torch.ones_like(e)

        attention = torch.where(biased_adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, feature_src)

        return self.activation(h_prime)
