import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
    def __init__(
        self, feature_dim: int, hidden_dim: int, dropout: float, num_classes: int
    ):
        """The GCN model.

        Args:
            feature_dim (int): The dimensionality of the input features
            hidden_dim (int): The dimensionality of the hidden layers
            dropout (float): The dropout rate for the input features
            num_classes (int): The number of classes to predict
        """
        super(GCN, self).__init__()
        self.body = GCNBody(feature_dim, hidden_dim, dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, g, x):
        x = self.body(g, x)
        x = self.fc(x)
        return x


class GCNBody(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float):
        """The body of the GCN model.

        Args:
            feature_dim (int): The dimensionality of the input features
            hidden_dim (int): The dimensionality of the hidden layers
            dropout (float): The dropout rate for the input features
        """
        super(GCNBody, self).__init__()
        self.gc1 = GraphConv(feature_dim, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x):
        x = F.relu(self.gc1(g, x))
        x = self.dropout(x)
        x = self.gc2(g, x)
        # x = self.dropout(x)
        return x
