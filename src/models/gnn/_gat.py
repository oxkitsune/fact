import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from typing import List, Optional


class GAT(nn.Module):
    def __init__(
        self,
        num_layers: int,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int,
        heads: Optional[List[int]],
        num_heads: Optional[int],
        num_out_heads: Optional[int],
        feat_drop: float,
        attn_drop: float,
        negative_slope: float,
        residual: bool,
    ):
        """A GAT model.

        Args:
            num_layers (int): The number of GATConv layers in this model
            feature_dim (int): The dimensionality of the input features
            hidden_dim (int): The dimensionality of the hidden layers
            num_classes (int): The number of classes to predict
            heads (List[int] | None): The number of attention heads in each layer, if specified this overrides `num_heads` and `num_out_heads`
            num_heads (int | None): The number of attention heads in the hidden layers
            num_out_heads (int | None): The number of attention heads in the output layer
            feat_drop (float): The dropout rate for the input features
            attn_drop (float): The dropout rate for the attention weights
            negative_slope (float): The negative slope for the LeakyReLU activation function
            residual (bool): Whether to use residual connections

        Raises:
            ValueError: If `heads` are specified, `num_heads` and `num_out_heads` must not be specified.
        """
        super(GAT, self).__init__()
        if heads is None:
            assert (
                num_heads is not None
            ), "If no `heads` are specified, `num_heads` must be specified."
            assert (
                num_out_heads is not None
            ), "If no `heads` are specified, `num_out_heads` must be specified."
            heads = [num_heads] * num_layers + [num_out_heads]
        elif num_heads is not None or num_out_heads is not None:
            raise ValueError(
                "If `heads` are specified, `num_heads` and `num_out_heads` must not be specified."
            )

        self.body = GATBody(
            num_layers,
            feature_dim,
            hidden_dim,
            heads,
            feat_drop,
            attn_drop,
            negative_slope,
            residual,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, g, inputs):
        logits = self.body(g, inputs)
        logits = self.fc(logits)

        return logits


class GATBody(nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        heads: List[int],
        feat_drop: float,
        attn_drop: float,
        negative_slope: float,
        residual: bool,
    ):
        """The GAT body.

        Args:
            num_layers (int): The number of GATConv layers in this model
            input_dim (int): The dimensionality of the input features
            hidden_dim (int): The dimensionality of the hidden layers
            heads (List[int]): The number of attention heads in each layer
            feat_drop (float): The dropout rate for the input features
            attn_drop (float): THe dropout rate for the attention weights
            negative_slope (float): The negative slope for the LeakyReLU activation function
            residual (bool): Whether to use residual connections
        """
        super(GATBody, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = F.elu

        # input projection (no residual)
        self.gat_layers.append(
            GATConv(
                input_dim,
                hidden_dim,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
            )
        )

        # hidden layers
        for i in range(1, num_layers):
            # due to multi-head, the input_dim = num_hidden * num_heads of last layer
            self.gat_layers.append(
                GATConv(
                    hidden_dim * heads[i - 1],
                    hidden_dim,
                    heads[i],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                )
            )

        # output projection
        self.gat_layers.append(
            GATConv(
                hidden_dim * heads[-2],
                hidden_dim,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
            )
        )

    def forward(self, g, inputs):
        h = inputs
        # TODO: Perhaps we can just use `nn.Sequential` here?
        for i in range(self.num_layers):
            h = self.gat_layers[i](g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)

        return logits
