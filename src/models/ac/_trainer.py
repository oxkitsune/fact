import torch
import torch.nn.functional as F


class Trainer:
    def __init__(
        self, ac_model, dataset, lr: float = 0.01, weight_decay: float = 0.001
    ):
        self.ac_model = ac_model
        self.dataset = dataset

        self.ac_optimizer = torch.optim.Adam(
            self.ac_model.parameters(), lr=1e-3, weight_decay=1e-5
        )

    def pretrain(self, epochs: int = 200):
        for epoch in range(epochs):
            self.ac_model.train()
            self.ac_optimizer.zero_grad()
            (
                training_adj,
                embeddings,
                kept_embeddings,
                kept_features,
                dropped_features,
                sens,
            ) = self.dataset.sample_ac()

            feature_src_re2, features_hat, transformed_feature = self.ac_model(
                training_adj,
                embeddings,
                kept_embeddings,
                kept_features,
            )

            loss_ac = self.ac_model.loss(dropped_features, feature_src_re2)
            loss_reconstruction = F.pairwise_distance(
                features_hat, kept_features, 2
            ).mean()

            print(
                "Epoch: {:04d}, loss_ac: {:.4f},loss_reconstruction: {:.4f}".format(
                    epoch, loss_ac.item(), loss_reconstruction.item()
                )
            )

            total_loss = loss_ac + loss_reconstruction
            total_loss.backward()
            self.ac_optimizer.step()

    def train(self, acc, fairness):
        raise NotImplementedError
