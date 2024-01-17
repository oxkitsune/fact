from _gat import GAT_Body
from _gcn import GCN_Body
from _sage import SAGE_Body

import torch
from torch import nn
from typing import Union, Literal

GNNKind = Literal["GCN", "GAT", "SAGE"]


def _create_gnn(gnn_type: GNNKind, input_dim: int, hidden_dim: int):
    """
    Create a GNN model based on the given type.
    """
    match gnn_type:
        case "GCN":
            return GCN_Body()
        case "GAT":
            return GAT_Body(input_dim, hidden_dim)
        case "SAGE":
            return SAGE_Body(input_dim, hidden_dim)
        case _:
            raise ValueError(f"Unknown GNN type: {gnn_type}")


# taken from `FairAC.py`
class WrappedGNN(nn.Module):
    """
    Wrapper class for GNN models.
    This class introduces a [TODO: what kind] classifier on top of the GNN model.
    """

    def __init__(self, input_dim, hidden_dim, gnn_type: GNNKind):
        super(WrappedGNN, self).__init__()
        nhid = args.num_hidden
        self.GNN = _create_gnn(gnn_type, input_dim, hidden_dim)
        self.classifier = nn.Linear(nhid, 1)
        G_params = list(self.GNN.parameters()) + list(self.classifier.parameters())
        self.optimizer_G = torch.optim.Adam(
            G_params, lr=args.lr, weight_decay=args.weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, g, x):
        z = self.GNN(g, x)
        y = self.classifier(z)
        return z, y
