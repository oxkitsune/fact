from _gat import GATBody
from _gcn import GCNBody
from _sage import SAGEBody

import torch
from torch import nn
from typing import Literal

GNNKind = Literal["GCN", "GAT", "SAGE"]


def _create_gnn(gnn_type: GNNKind, **kwargs):
    """
    Create a GNN model based on the given type.
    """
    match gnn_type:
        case "GCN":
            return GCNBody(**kwargs)
        case "GAT":
            return GATBody(**kwargs)
        case "SAGE":
            return SAGEBody(**kwargs)
        case _:
            raise ValueError(f"Unknown GNN type: {gnn_type}")


# taken from `FairAC.py`
class WrappedGNN(nn.Module):
    """
    Wrapper class for GNN models.
    This class introduces a sensitivy classifier on top of the GNN model.

    Note: It's important that the GNN model's output is the logits for the classifier, e.g. before the model applies it's own classifier!
          The forward pass of this model will return the logits of the GNN, alongside the output of the classifier.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        gnn_type: GNNKind,
        lr: float,
        weight_decay: float,
    ):
        """A wrapper class for GNN models.

        Args:
            input_dim (int): The dimensionality of the input features
            hidden_dim (int): The dimensionality of the hidden layers
            gnn_type (GNNKind): The type of GNN to use
            lr (float): The learning rate
            weight_decay (float): The weight decay
        """
        super(WrappedGNN, self).__init__()
        self.gnn = _create_gnn(gnn_type, input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.optimizer_G = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, g, x):
        z = self._gnn(g, x)
        y = self.classifier(z)
        return z, y
