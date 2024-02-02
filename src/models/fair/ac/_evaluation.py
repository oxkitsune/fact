from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.auto import trange
import json

from dataset import NBA
from metrics import BestMetrics, Metrics, accuracy, consistency_metric, fair_metric
from models.gnn import WrappedGNN, WrappedGNNConfig


class Evaluation:
    def __init__(self, dataset: NBA, device: torch.device):
        self.dataset = dataset
        self.device = device

        self.gnn_config = None
        self.model = None
        self.min_acc = None
        self.min_roc = None

    def load_model(self, path: Path):
        best_metrics = json.load((path / "best_metrics.json").open("r"))
        print("Loaded best metrics", best_metrics)

        self.model = torch.load(path / best_metrics["best_ac_model"]).to(self.device)
        self.model.eval()
        print("Loaded model", self.model)

        self.best_gnn = torch.load(path / best_metrics["best_gnn_model"]).to(
            self.device
        )
        self.best_gnn.eval()

        self.hparams = json.load((path / "hparams.json").open("r"))

        print("Loaded gnn_model", self.best_gnn)
        self.gnn_config = json.load((path / "gnn_config.json").open("r"))
        self.gnn_config = WrappedGNNConfig(**self.gnn_config)

    def evaluate_best_gnn(self):
        assert self.best_gnn is not None, "Best GNN Model not loaded"
        assert self.model is not None, "AC Model not loaded"

        features_embedding = self._get_feature_embeddings()
        y_idx, train_idx, labels = self.dataset.inside_labels()

        with torch.no_grad():
            test_idx = self.dataset.test_idx

            _, output = self.best_gnn(self.dataset.graph, features_embedding)
            acc_test = accuracy(output[test_idx], labels[test_idx])

            roc_test = roc_auc_score(
                labels[test_idx].cpu().numpy(),
                output[test_idx].detach().cpu().numpy(),
            )

            parity, equality = fair_metric(output, test_idx, labels, self.dataset.sens)

            consistency = consistency_metric(self.dataset.sparse_adj, test_idx, output)

            result = Metrics(None, acc_test.item(), roc_test, parity, equality, None)

            print("Best metrics:")
            print(f"\tacc: {result.acc:.4f}")
            print(f"\troc: {result.roc:.4f}")
            print(
                f"\tparity: {result.parity:.4f}, equality: {result.equality:.4f}, sum: {(result.parity + result.equality):.4f}"
            )
            print(f"\tconsistency: {consistency:.4f}")

    def evaluate(self, epochs: int = 1000, progress_bar: bool = True):
        features_embedding = self._get_feature_embeddings()
        features_embedding_exclude_test = features_embedding[self.dataset.mask].detach()

        y_idx, train_idx, labels = self.dataset.inside_labels()

        gnn_model = WrappedGNN(
            input_dim=features_embedding.shape[1],
            config=self.gnn_config,
        ).to(self.device)

        print(gnn_model)
        best_metrics = BestMetrics(None, None, None, None)
        best_epoch = -1

        pbar = trange(epochs, leave=False, disable=not progress_bar)
        for epoch in pbar:
            gnn_model.train()
            pbar.set_description(f"Epoch {epoch}")

            gnn_model.optimizer.zero_grad()
            _feat_emb, y_hat = gnn_model(
                self.dataset.train_sub_graph, features_embedding_exclude_test
            )

            cy_loss = gnn_model.criterion(
                y_hat[y_idx], labels[train_idx].unsqueeze(1).to(dtype=torch.float32)
            )
            cy_loss.backward()

            gnn_model.optimizer.step()

            pbar.set_postfix_str(
                f"Loss: {cy_loss.item():.04f}",
            )

            gnn_model.eval()
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

                result = Metrics(epoch, acc_test.item(), roc_test, parity, equality, None)

                if best_metrics.update_metrics(
                    result, self.hparams["min_acc"], self.hparams["min_roc"]
                ):
                    best_epoch = epoch
                    print("Best metrics updated, new fairness:", best_metrics.fair.parity + best_metrics.fair.equality)
                    best_metrics.fair.consistency = consistency_metric(
                        self.dataset.sparse_adj, test_idx, output
                    )

        print("Best metrics:")
        print(f"\tepoch: {best_epoch}")
        print(f"\tacc: {best_metrics.fair.acc:.4f}")
        print(f"\troc: {best_metrics.fair.roc:.4f}")
        print(
            f"\tparity: {best_metrics.fair.parity:.4f}, equality: {best_metrics.fair.equality:.4f}, sum: {(best_metrics.fair.parity + best_metrics.fair.equality):.4f}"
        )
        print(f"\tconsistency: {best_metrics.fair.consistency:.4f}")

    def _get_feature_embeddings(self):
        assert self.model is not None, "Model not loaded"
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
                feature_src_ac, _features_hat, transformed_feature = self.model(
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
