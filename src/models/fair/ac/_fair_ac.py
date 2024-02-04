import torch.nn as nn
import torch.nn.functional as F

from ._hggn import HGNNAC


class FairAC(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        transformed_feature_dim: int,
        emb_dim: int,
        attn_vec_dim: int,
        attn_num_heads: int,
        dropout: float,
        num_sensitive_classes: int,
    ):
        """The Fair AC model.

        Args:
            feature_dim (int): The dimension of the input features.
            transformed_feature_dim (int): The dimension of the transformed features.
            emb_dim (int): The dimension of the embeddings.
            attn_vec_dim (int): The dimension of the attention vectors.
            attn_num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
            num_sensitive_classes (int): The number of sensitive classes.
        """
        super(FairAC, self).__init__()
        self.ae = FairACAutoEncoder(feature_dim, transformed_feature_dim, emb_dim)
        self.hgnn_ac = HGNNAC(
            input_dim=emb_dim,
            hidden_dim=attn_vec_dim,
            dropout=dropout,
            num_heads=attn_num_heads,
            activation=F.elu,
        )
        self.sensitive_classifier = nn.Linear(
            transformed_feature_dim, num_sensitive_classes
        )

    def forward(self, biased_adj, emb_dest, emb_src, feature_src):
        transformed_features, feature_hat = self.ae(feature_src)
        feature_src_re = self.hgnn_ac(
            biased_adj, emb_dest, emb_src, transformed_features
        )

        return feature_src_re, feature_hat, transformed_features

    def sensitive_pred(self, transformed_features):
        return self.sensitive_classifier(transformed_features)

    def encode_feature(self, features):
        self.ae.encoder(features)

    def decode_feature(self, transformed_features):
        self.ae.decoder(transformed_features)

    def loss(self, origin_feature, ac_feature):
        return F.pairwise_distance(
            self.ae.encoder(origin_feature).detach(), ac_feature, 2
        ).mean()


class FairACAutoEncoder(nn.Module):
    def __init__(self, feature_dim: int, transformed_feature_dim: int, emb_dim: int):
        """The Fair AC autoencoder.

        Args:
            feature_dim (int): The dimension of the input features.
            transformed_feature_dim (int): The dimension of the transformed features.
            emb_dim (int): The dimension of the embeddings.
        """
        super(FairACAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 2 * transformed_feature_dim),
            nn.ReLU(),
            nn.Linear(2 * transformed_feature_dim, transformed_feature_dim),
        )

        nn.init.xavier_normal_(self.encoder[0].weight, gain=1.414)
        nn.init.xavier_normal_(self.encoder[2].weight, gain=1.414)

        self.decoder = nn.Sequential(
            nn.Linear(transformed_feature_dim, 2 * transformed_feature_dim),
            nn.ReLU(),
            nn.Linear(2 * transformed_feature_dim, feature_dim),
        )

        nn.init.xavier_normal_(self.decoder[0].weight, gain=1.414)
        nn.init.xavier_normal_(self.decoder[2].weight, gain=1.414)

    def forward(self, x):
        h = self.encoder(x)
        h_hat = self.decoder(h)

        return h, h_hat
