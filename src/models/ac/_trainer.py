from itertools import chain
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import trange
from models.gnn import GNNKind, WrappedGNN

from sklearn.metrics import roc_auc_score
from ._metric import fair_metric, accuracy, Metrics

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BestMetrics:
    best_fair: Metrics
    acc: Metrics
    auc: Metrics
    ar: Metrics


class Trainer:
    def __init__(
        self,
        ac_model,
        dataset,
        gnn_kind: GNNKind,
        gnn_hidden_dim: int,
        gnn_lr: float,
        gnn_weight_decay: float,
        gnn_args: dict,
        log_dir: Path,
        min_acc: float,
        min_roc: float,
        lambda1=1,
        lambda2=1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        self.ac_model = ac_model
        self.dataset = dataset
        self.gnn_kind = gnn_kind
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_lr = gnn_lr
        self.gnn_weight_decay = gnn_weight_decay
        self.gnn_args = gnn_args
        self.log_dir = log_dir
        self.best_metrics = None
        self.min_acc = min_acc
        self.min_roc = min_roc

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

    def train(
        self,
        epochs: int,
        val_start_epoch,
        val_epoch_interval: int = 100,
        progress_bar: bool = True,
    ):
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

            # print("train embeddings shape", embeddings.shape)
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

            if (
                epoch > val_start_epoch and val_epoch_interval % 100 == 0
            ) or epoch == epochs - 1:
                self._eval_with_gnn(epoch)

        if self.best_metrics is None:
            print("Please set smaller acc/roc thresholds!")
        else:
            print("Finished training!")
            print()
            print("Best fair model:")
            print(f"\tacc: {self.best_metrics.best_fair.acc}")
            print(f"\troc: {self.best_metrics.best_fair.roc}")
            print(f"\tparity: {self.best_metrics.best_fair.parity}")
            print(f"\tequality: {self.best_metrics.best_fair.equality}")

    def _eval_with_gnn(self, curr_epoch, epochs=1000):
        features_embedding = self._get_feature_embeddings()
        features_embedding_exclude_test = features_embedding[self.dataset.mask]

        y_idx, labels = self.dataset.inside_labels()

        gnn_model = WrappedGNN(
            input_dim=features_embedding.shape[1],
            hidden_dim=self.gnn_hidden_dim,
            gnn_type=self.gnn_kind,
            lr=self.gnn_lr,
            weight_decay=self.gnn_weight_decay,
            **self.gnn_args,
        )

        gnn_model.train()
        pbar = trange(epochs, leave=False)
        for epoch in pbar:
            pbar.set_description(f"Sub-epoch {epoch}")

            gnn_model.zero_grad()
            _feat_emb, y_hat = gnn_model(
                self.dataset.train_sub_graph, features_embedding_exclude_test
            )

            cy_loss = gnn_model.criterion(y_hat[y_idx], labels.unsqueeze(1).float())
            cy_loss.backward()

            gnn_model.optimizer.step()

            pbar.set_postfix_str(
                f"Loss AC: {cy_loss.item():.04d}",
            )

        gnn_model.eval()

        with torch.no_grad():
            print("Mask:", self.dataset.mask)
            print("Inverted mask:", self.dataset.mask.invert())

            mask_test = self.dataset.mask.invert()

            _, output = gnn_model(self.dataset.train_sub_graph, features_embedding)
            acc_test = accuracy(output[mask_test], labels[mask_test])
            roc_test = roc_auc_score(
                labels[mask_test].cpu().numpy(),
                output[mask_test].detach().cpu().numpy(),
            )
            parity, equality = fair_metric(output, mask_test, labels, self.dataset.sens)

            result = Metrics(acc_test.item(), roc_test, parity, equality)

            if result.acc > self.best_metrics.acc or self.best_metrics.acc is None:
                self.best_metrics.acc = result

            if result.roc > self.best_metrics.auc or self.best_metrics.auc is None:
                self.best_metrics.auc = result

            if (
                result.acc + result.roc > self.best_metrics.ar
                or self.best_metrics.ar is None
            ):
                self.best_metrics.ar = result

            best_fair = (
                result.parity + result.equality
                < self.best_metrics.best_fair.parity
                + self.best_metrics.best_fair.equality
            )
            if (
                best_fair and result.acc > self.min_acc and result.roc > self.min_roc
            ) or self.best_metrics.best_fair is None:
                self.best_metrics.best_fair = result

                torch.save(gnn_model, self.log_dir / f"gnn_epoch{curr_epoch:04d}.pt")
                torch.save(self.ac_model, self.log_dir / f"ac_epoch{curr_epoch:04d}.pt")

    def _get_feature_embeddings(self):
        features_embedding = None
        loader = DataLoader(self.dataset, batch_size=None)
        with torch.no_grad():
            # ############# Attribute completion over graph######################
            for (
                sub_adj,
                sub_node,
                embeddings,
                features,
                keep_indices,
                drop_indices,
            ) in loader:
                feature_src_ac, _features_hat, transformed_feature = self.ac_model(
                    sub_adj,
                    embeddings,
                    embeddings[keep_indices],
                    features[keep_indices],
                )

                # I want to cry ^_^
                if features_embedding is None:
                    features_embedding = torch.zeros(
                        (self.dataset.features.shape[0], transformed_feature.shape[1]),
                    )

                features_embedding[sub_node[drop_indices]] = feature_src_ac[
                    drop_indices
                ]
                features_embedding[sub_node[keep_indices]] = transformed_feature
        # 😿
        return features_embedding
