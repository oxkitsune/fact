import torch
from torch import nn


class FairAC(nn.Module):
    def __init__(self, feature_dim: int, transformed_feature_dim: int, emb_dim: int):
        super(FairAC, self).__init__()
        # TODO: figure out why they multiply by 2
        self.ae = FairACAutoEncoder(feature_dim, transformed_feature_dim, emb_dim)


class FairACAutoEncoder(nn.Module):
    def __init__(self, feature_dim: int, transformed_feature_dim: int, emb_dim: int):
        super(FairACAutoEncoder, self).__init__()
        # TODO: figure out why they multiply by 2
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 2 * transformed_feature_dim),
            nn.ReLU(),
            nn.Linear(2 * transformed_feature_dim, transformed_feature_dim),
        )

        # TODO: figure out why they set gain to 1.414 (sqrt(2) perhaps?)
        nn.init.xavier_normal_(self.encoder.Linear1.weight, gain=1.414)
        nn.init.xavier_normal_(self.encoder.Linear2.weight, gain=1.414)

        self.decoder = nn.Sequential(
            nn.Linear(transformed_feature_dim, 2 * transformed_feature_dim),
            nn.ReLU(),
            nn.Linear(2 * transformed_feature_dim, feature_dim),
        )

        # TODO: figure out why they set gain to 1.414 (sqrt(2) perhaps?)
        nn.init.xavier_normal_(self.decoder.Linear1.weight, gain=1.414)
        nn.init.xavier_normal_(self.decoder.Linear2.weight, gain=1.414)

    def forward(self, x):
        h = self.encoder(x)
        h_hat = self.decoder(x)

        return h, h_hat
