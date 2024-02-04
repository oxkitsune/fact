from itertools import chain
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import trange
from sklearn.metrics import roc_auc_score
from dataclasses import asdict
from pathlib import Path
import json

from dataset import FairACDataset
from metrics import fair_metric, accuracy, Metrics, BestMetrics, consistency_metric
from models.gnn import WrappedGNN, WrappedGNNConfig
from models.fair.ac import FairAC


class Trainer:
    def __init__(
        self,
        ac_model: FairAC,
        dataset: FairACDataset,
        device: torch.device,
        gnn_config: WrappedGNNConfig,
        log_dir: Path,
        min_acc: float,
        min_roc: float,
        lambda1=1,
        lambda2=1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        """Generate a new Trainer for the FairAC model.

        Args:
            ac_model (FairAC): The FairAC model to train.
            dataset (FairACDataset): The dataset to train on.
            device (torch.device): The device to run the training on.
            gnn_config (WrappedGNNConfig): The configuration for the GNN model.
            log_dir (Path): The directory to save the logs and models to.
            min_acc (float): The minimum accuracy threshold.
            min_roc (float): The minimum roc threshold.
            lambda1 (int, optional): The lambda1 hyperparameter. Defaults to 1.
            lambda2 (int, optional): The lambda2 hyperparameter. Defaults to 1.
            lr (float, optional): The learning rate for both the AC model and the gnn that's evaluated. Defaults to 1e-3.
            weight_decay (float, optional) The weight decay value for the Adam optimizer. Defaults to 1e-5.
        """

        self.ac_model = ac_model.to(device)
        self.dataset = dataset
        self.device = device
        self.gnn_config = gnn_config
        self.log_dir = log_dir
        self.best_metrics = BestMetrics(None, None, None, None)
        self.best_epoch = None
        self.best_output = None
        self.min_acc = min_acc
        self.min_roc = min_roc

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lr = lr
        self.weight_decay = weight_decay

        ac_params = chain(
            self.ac_model.ae.encoder.parameters(),
            self.ac_model.ae.decoder.parameters(),
            self.ac_model.hgnn_ac.parameters(),
        )
        self.ac_optimizer = torch.optim.Adam(
            ac_params, lr=self.lr, weight_decay=self.weight_decay
        )
        self.c_sen_optimizer = torch.optim.Adam(
            self.ac_model.sensitive_classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # save configuration in log dir
        self._save_config()

    def pretrain(self, epochs: int = 200, progress_bar=True):
        pbar = trange(epochs, disable=not progress_bar)
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}")

            self.ac_model.train()
            self.ac_optimizer.zero_grad()
            self.c_sen_optimizer.zero_grad()

            (
                train_adj,
                embeddings,
                features,
                _sens,
                keep_indices,
                drop_indices,
            ) = self.dataset.sample_fairac()

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
                dropped_features, feature_src_re2[drop_indices, :]
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
            self.ac_optimizer.zero_grad()
            self.c_sen_optimizer.zero_grad()

            (
                train_adj,
                embeddings,
                features,
                train_sens,
                keep_indices,
                drop_indices,
            ) = self.dataset.sample_fairac()

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
                dropped_features, feature_src_re2[drop_indices, :]
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
                sens_prediction_detach,
                train_sens[keep_indices].unsqueeze(1).to(dtype=torch.float32),
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
                    device=self.device,
                    dtype=torch.float32,
                )
                / 2
            )
            Csen_adv_loss = criterion(sens_prediction, sens_confusion)

            sens_prediction_keep = self.ac_model.sensitive_pred(transformed_feature)
            Csen_loss = criterion(
                sens_prediction_keep,
                train_sens[keep_indices].unsqueeze(1).to(dtype=torch.float32),
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
                epoch > val_start_epoch and epoch % val_epoch_interval == 0
            ) or epoch == epochs - 1:
                self._eval_with_gnn(epoch, progress_bar=progress_bar)

        if (
            self.best_metrics.fair.acc < self.min_acc
            or self.best_metrics.fair.roc < self.min_roc
        ):
            print("Please set smaller acc/roc thresholds!")
        else:
            print("Finished training!")

        print(f"Best epoch: {self.best_epoch}")

        self.best_metrics.fair.consistency = consistency_metric(
            self.dataset.sparse_adj, self.dataset.test_idx, self.best_output
        )

        for metric, name in zip(
            [self.best_metrics.fair, self.best_metrics.acc, self.best_metrics.auc],
            ["fair", "acc", "auc"],
        ):
            print()
            Trainer._print_metric(metric, name)

        with open(self.log_dir / "best_metrics.json", "w+") as f:
            best_metrics = asdict(self.best_metrics)
            best_metrics["best_epoch"] = self.best_epoch
            best_metrics["best_gnn_model"] = f"gnn_epoch{self.best_epoch}.pt"
            best_metrics["best_ac_model"] = f"ac_epoch{self.best_epoch}.pt"
            json.dump(best_metrics, f, indent=4)

    def _print_metric(metric, name):
        print(f"Best {name} model:")
        print(f"\tacc: {metric.acc:.04f}")
        print(f"\troc: {metric.roc:.04f}")
        print(f"\tparity: {metric.parity:.04f}")
        print(f"\tequality: {metric.equality:.04f}")

        if metric.consistency is not None:
            print(f"\tconsistency: {metric.consistency:.04f}")

    def _eval_with_gnn(self, curr_epoch, epochs=1000, progress_bar: bool = True):
        features_embedding = self._get_feature_embeddings()
        features_embedding_exclude_test = features_embedding[self.dataset.mask].detach()
        y_idx, train_idx, labels = self.dataset.inside_labels()

        gnn_model = WrappedGNN(features_embedding.shape[1], self.gnn_config).to(
            self.device
        )

        pbar = trange(epochs, leave=False, disable=not progress_bar)
        for epoch in pbar:
            gnn_model.train()
            pbar.set_description(f"Sub-epoch {epoch}")

            gnn_model.optimizer.zero_grad()
            _feat_emb, y_hat = gnn_model(
                self.dataset.train_sub_graph, features_embedding_exclude_test
            )

            cy_loss = gnn_model.criterion(
                y_hat[y_idx], labels[train_idx].unsqueeze(1).to(dtype=torch.float32)
            )
            cy_loss.backward()

            gnn_model.optimizer.step()

            gnn_model.eval()
            self.ac_model.eval()
            with torch.no_grad():
                test_idx = self.dataset.test_idx

                _, output = gnn_model(self.dataset.graph, features_embedding)
                acc_test = accuracy(output[test_idx], labels[test_idx])

                roc_test = roc_auc_score(
                    labels[test_idx].cpu().numpy(),
                    output[test_idx].detach().cpu().numpy(),
                )

                parity, equality = fair_metric(
                    output, test_idx, labels, self.dataset.sens
                )

                result = Metrics(
                    curr_epoch, acc_test.item(), roc_test, parity, equality, None
                )

                pbar.set_postfix_str(
                    f"Loss: {cy_loss.item():.04f}, Acc: {acc_test.item():.04f}, Roc: {roc_test:.04f}, Fair: {(parity + equality):.04f}"
                )

                if self.best_metrics.update_metrics(result, self.min_acc, self.min_roc):
                    pbar.write(
                        f"[{curr_epoch}:{epoch}] Found new best fairness of {result.parity + result.equality:.6f}"
                    )

                    self.best_output = output
                    self.best_epoch = curr_epoch

                    torch.save(gnn_model, self.log_dir / f"gnn_epoch{curr_epoch}.pt")
                    torch.save(self.ac_model, self.log_dir / f"ac_epoch{curr_epoch}.pt")

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

                if features_embedding is None:
                    features_embedding = torch.zeros(
                        (self.dataset.features.shape[0], transformed_feature.shape[1]),
                    ).to(self.device)

                features_embedding[sub_node[drop_indices]] = feature_src_ac[
                    drop_indices
                ]
                features_embedding[sub_node[keep_indices]] = transformed_feature

        return features_embedding

    def _save_config(self):
        # open file in mode that truncates first
        with open(self.log_dir / "hparams.json", "w+") as f:
            hparams = {
                "min_acc": self.min_acc,
                "min_roc": self.min_roc,
                "lambda1": self.lambda1,
                "lambda2": self.lambda2,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
            }

            json.dump(hparams, f, indent=4)

        with open(self.log_dir / "gnn_config.json", "w+") as f:
            json.dump(asdict(self.gnn_config), f, indent=4)
