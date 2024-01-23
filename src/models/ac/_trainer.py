from itertools import chain
import torch
import torch.nn.functional as F
from tqdm import trange, tqdm


class Trainer:
    def __init__(
        self,
        ac_model,
        dataset,
        lambda1=1,
        lambda2=1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        self.ac_model = ac_model
        self.dataset = dataset

        self.lambda1 = lambda1
        self.lambda2 = lambda2

        ac_params = chain(
            self.ac_model.ae.encoder.parameters(),
            self.ac_model.ae.decoder.parameters(),
            self.ac_model.hgnn_ac.parameters(),
        )
        self.ac_optimizer = torch.optim.Adam(
            ac_params, lr=lr, weight_decay=weight_decay
        )
        self.c_sen_optimizer = torch.optim.Adam(
            self.ac_model.sensitive_classifier.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def pretrain(self, epochs: int = 200, progress_bar=True):
        pbar = trange(epochs, disable=not progress_bar)
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}")

            self.ac_model.train()
            self.ac_optimizer.zero_grad()

            (
                train_adj,
                embeddings,
                features,
                _sens,
                keep_indices,
                drop_indices,
            ) = self.dataset.sample_ac()

            kept_embeddings = embeddings[keep_indices]
            kept_features = features[keep_indices]
            dropped_features = features[drop_indices]

            feature_src_re2, features_hat, _transformed_feature = self.ac_model(
                train_adj,
                embeddings,
                kept_embeddings,
                kept_features,
            )

            loss_ac = self.ac_model.loss(
                dropped_features, feature_src_re2[drop_indices]
            )

            loss_reconstruction = F.pairwise_distance(
                features_hat, kept_features, 2
            ).mean()

            pbar.set_postfix_str(
                f"Loss AC: {loss_ac.item():.4f}, Loss Reconstruction: {loss_reconstruction.item():.4f}",
            )

            total_loss = loss_ac + loss_reconstruction
            total_loss.backward()
            self.ac_optimizer.step()

    def train(self, epochs, acc, fairness, progress_bar=True):
        pbar = trange(epochs, disable=not progress_bar)
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}")

            self.ac_model.train()
            self.ac_model.zero_grad()

            (
                train_adj,
                embeddings,
                features,
                train_sens,
                keep_indices,
                drop_indices,
            ) = self.dataset.sample_ac()

            kept_embeddings = embeddings[keep_indices]
            kept_features = features[keep_indices]
            dropped_features = features[drop_indices]

            feature_src_re2, features_hat, transformed_feature = self.ac_model(
                train_adj,
                embeddings,
                kept_embeddings,
                kept_features,
            )

            loss_ac = self.ac_model.loss(
                dropped_features, feature_src_re2[drop_indices]
            )

            loss_reconstruction = F.pairwise_distance(
                features_hat, kept_features, 2
            ).mean()

            # mitigate unfairness loss
            sens_prediction_detach = self.ac_model.sensitive_pred(
                transformed_feature.detach()
            )
            criterion = torch.nn.BCEWithLogitsLoss()
            # only update sensitive classifier
            Csen_loss = criterion(
                sens_prediction_detach, train_sens[keep_indices].unsqueeze(1).float()
            )

            # sensitive optimizer.step
            Csen_loss.backward()
            self.c_sen_optimizer.step()

            feature_src_re2[keep_indices] = transformed_feature
            sens_prediction = self.ac_model.sensitive_pred(
                feature_src_re2[drop_indices]
            )
            sens_confusion = (
                torch.ones(
                    sens_prediction.shape,
                    device=sens_prediction.device,
                    dtype=torch.float32,
                )
                / 2
            )
            Csen_adv_loss = criterion(sens_prediction, sens_confusion)

            sens_prediction_keep = self.ac_model.sensitive_pred(transformed_feature)

            Csen_loss = criterion(
                sens_prediction_keep, train_sens[keep_indices].unsqueeze(1).float()
            )

            total_loss = (
                self.lambda2 * (Csen_adv_loss - Csen_loss)
                + loss_reconstruction
                + self.lambda1 * loss_ac
            )
            total_loss.backward()
            self.ac_optimizer.step()

            pbar.set_postfix_str(
                f"Loss AC: {loss_ac.item():.4f}, Loss Reconstruction: {loss_reconstruction.item():.4f}, Loss Sensitive: {(Csen_adv_loss - Csen_loss):.4f}",
            )
