from ._gat import GATBody
from ._gcn import GCNBody, GCN
from ._sage import SAGEBody

from itertools import chain
from dataclasses import dataclass

import torch
from torch import nn
from typing import Literal

GNNKind = Literal["GCN", "GAT", "SAGE"]


def _create_gnn(gnn_type: GNNKind, **kwargs):
    """
    Create a GNN model based on the given type.
    """
    if gnn_type == "GCN":
        return GCNBody(**kwargs)
    elif gnn_type == "GAT":
        return GATBody(**kwargs)
    elif gnn_type == "SAGE":
        return SAGEBody(**kwargs)
    else:
        raise ValueError(f"Unknown GNN type: {gnn_type}")


@dataclass
class WrappedGNNConfig:
    """
    Configuration for wrapped GNN models.
    """

    hidden_dim: int
    kind: GNNKind
    lr: float
    weight_decay: float
    # additional kwargs for the GNN model
    kwargs: dict


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
        config: WrappedGNNConfig,
    ):
        """A wrapper class for GNN models.

        Args:
            input_dim (int): Input dimension of the GNN.
            config (WrappedGNNConfig): Configuration for the GNN model.
        """
        super(WrappedGNN, self).__init__()
        self.gnn = _create_gnn(
            config.kind,
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            **config.kwargs,
        )
        self.classifier = nn.Linear(config.hidden_dim, 1)

        gnn_params = chain(self.gnn.parameters(), self.classifier.parameters())
        self.optimizer = torch.optim.Adam(
            gnn_params, lr=config.lr, weight_decay=config.weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, g, x):
        z = self.gnn(g, x)
        y = self.classifier(z)
        return z, y
