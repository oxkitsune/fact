import torch
import torch.nn as nn

from models.gnn import WrappedGNN, GNNKind, GCN


class FairGNN(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_hidden: int = 128,
        dropout: float = 0.5,
        alpha: float = 10,
        beta: float = 1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        gnn_kind: GNNKind = "GCN",
        gnn_hidden_dim: int = 64,
        gnn_args: dict = {"dropout": 0.5},
    ):
        super(FairGNN, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.estimator = GCN(num_features, num_hidden, dropout, 1)
        self.gnn = WrappedGNN(
            input_dim=num_features,
            hidden_dim=gnn_hidden_dim,
            gnn_type=gnn_kind,
            lr=lr,
            weight_decay=weight_decay,
            **gnn_args,
        )

        self.adv = nn.Linear(gnn_hidden_dim, 1)
        self.gnn_args = gnn_args

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, adj, x):
        s = self.estimator(adj, x)
        _, y = self.gnn(adj, x)

        return y, s

    def gnn_loss(self, y_score, y_pred, y_labels, s_g, s_score):
        cls_loss = self.criterion(y_pred, y_labels)
        adv_loss = self.criterion(s_score, s_g)
        cov = torch.abs(
            torch.mean(s_score - torch.mean(s_score))
            * torch.mean(y_score - torch.mean(y_score))
        )

        return cls_loss + self.alpha * cov - self.beta * adv_loss

    def adv_loss(self, h, s_score):
        s_g = self.adv(h)
        return self.criterion(s_score, s_g)
