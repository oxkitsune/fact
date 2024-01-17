import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv


class SAGE(nn.Module):
    def __init__(
        self, feature_dim: int, hidden_dim: int, num_classes: int, dropout: float
    ):
        """The SAGE model.

        Args:
            feature_dim (int): The dimensionality of the input features
            hidden_dim (int): The dimensionality of the hidden layers
            num_classes (int): The number of classes
            dropout (float): The dropout rate
        """
        super(SAGE, self).__init__()
        self.body = SAGEBody(feature_dim, hidden_dim, dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, g, x):
        x = self.body(g, x)
        x = self.fc(x)
        return x


class SAGEBody(nn.Module):
    def __init__(
        self, feature_dim: int, hidden_dim: int, num_classes: int, dropout: float
    ):
        """The body of the SAGE model.

        Args:
            feature_dim (int): The dimensionality of the input features
            hidden_dim (int): The dimensionality of the hidden layers
            num_classes (int): The number of classes
            dropout (float): The dropout rate
        """
        super(SAGEBody, self).__init__()

        self.gc1 = SAGEConv(feature_dim, hidden_dim, "mean")
        self.gc2 = SAGEConv(hidden_dim, hidden_dim, "mean")
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x):
        x = F.relu(self.gc1(g, x))
        x = self.dropout(x)
        x = self.gc2(g, x)
        # x = self.dropout(x)
        return x
