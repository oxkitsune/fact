from dataclasses import asdict
from itertools import chain
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score

import torch
import torch.nn.functional as F
from tqdm.auto import trange

from models.fair.gnn import FairGNN
from metrics import Metrics, accuracy, BestMetrics, fair_metric, consistency_metric
import dgl


class Trainer:
    def __init__(
        self,
        dataset,
        device,
        fair_gnn: FairGNN,
        log_dir: Path,
        min_acc: float,
        min_roc: float,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        self.dataset = dataset
        self.device = device
        self.fair_gnn = fair_gnn.to(device)
        self.log_dir = log_dir
        self.best_fair = None
        self.best_output = None
        self.best_epoch = None
        self.best_val_metrics = BestMetrics(None, None, None, None)
        self.min_acc = min_acc
        self.min_roc = min_roc

        gnn_params = chain(
            self.fair_gnn.gnn.parameters(), self.fair_gnn.estimator.parameters()
        )
        self.gnn_optimizer = torch.optim.Adam(
            gnn_params, lr=lr, weight_decay=weight_decay
        )
        self.adv_optimizer = torch.optim.Adam(
            self.fair_gnn.adv.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train(
        self,
        epochs: int,
        progress_bar: bool = True,
    ):
        (
            adj,
            _,
            features,
            sens,
            keep_indices,
            drop_indices,
        ) = self.dataset.sample_full()
        adj = self.dataset.graph
        # average mean features for dropped nodes
        kept = features[keep_indices]
        mean = kept.mean(dim=0)

        features[drop_indices] = mean

        pbar = trange(epochs, disable=not progress_bar)
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}")
            self.optimize(
                adj,
                features,
                sens,
            )
            self.eval(pbar, epoch, adj, features, sens)

        if self.best_fair is None:
            print("Please set smaller acc/roc thresholds!")
        else:
            consistency = consistency_metric(
                self.dataset.sparse_adj, self.dataset.test_idx, self.best_output
            )
            self.best_fair.consistency = consistency

            print("Finished training!")
            Trainer._print_metric(self.best_fair, "fair")

            with open(self.log_dir / "best_metrics.json", "a") as f:
                best_metrics = asdict(self.best_fair)
                best_metrics["best_epoch"] = self.best_epoch
                best_metrics["best_gnn_model"] = f"gnn_epoch{self.best_epoch}.pt"
                json.dump(asdict(self.best_fair), f, indent=4)

    def optimize(self, adj, features, sens):
        self.fair_gnn.train()

        sens_train_idx = self.dataset.sens_train_idx
        y_idx, train_idx, labels = self.dataset.inside_labels()

        sens = sens.to(dtype=torch.float32, device=self.device)

        self.fair_gnn.adv.requires_grad_(False)
        self.gnn_optimizer.zero_grad()

        s = self.fair_gnn.estimator(adj, features)
        h, y = self.fair_gnn.gnn(adj, features)

        s_g = self.fair_gnn.adv(h)

        s_score = torch.sigmoid(s.detach())
        s_score[sens_train_idx] = (
            sens[sens_train_idx]
            .unsqueeze(1)
            .to(dtype=torch.float32, device=self.device)
        )
        y_score = torch.sigmoid(y)

        gnn_loss = self.fair_gnn.gnn_loss(
            y_score,
            y[train_idx],
            labels[train_idx].unsqueeze(1).to(dtype=torch.float32, device=self.device),
            s_g,
            s_score,
        )
        gnn_loss.backward()
        self.gnn_optimizer.step()

        ## update Adv
        self.fair_gnn.adv.requires_grad_(True)
        self.adv_optimizer.zero_grad()

        adv_loss = self.fair_gnn.adv_loss(h.detach(), s_score)
        adv_loss.backward()
        self.adv_optimizer.step()

    def eval(self, pbar, curr_epoch, adj, features, sens):
        self.fair_gnn.eval()
        val_idx = self.dataset.val_idx
        test_idx = self.dataset.test_idx
        y_idx, train_idx, labels = self.dataset.inside_labels()

        output, s = self.fair_gnn(adj, features)
        acc_val = accuracy(output[val_idx], labels[val_idx])
        roc_val = roc_auc_score(
            labels[val_idx].cpu().numpy(), output[val_idx].detach().cpu().numpy()
        )
        parity_val, equality_val = fair_metric(
            output, val_idx, labels=labels, sens=sens
        )

        val_result = Metrics(
            curr_epoch,
            acc_val.item(),
            roc_val,
            parity_val,
            equality_val,
            None,
        )

        acc_test = accuracy(output[test_idx], labels[test_idx])
        roc_test = roc_auc_score(
            labels[test_idx].cpu().numpy(), output[test_idx].detach().cpu().numpy()
        )
        parity, equality = fair_metric(output, test_idx, labels=labels, sens=sens)

        pbar.set_postfix_str(
            f"Acc: {acc_test.item():.4f}, Roc: {roc_test:.4f}, Parity: {parity:.4f}, Equality: {equality:.4f}",
        )

        result = Metrics(curr_epoch, acc_test.item(), roc_test, parity, equality, None)

        if (
            (
                self.best_val_metrics.fair is None
                or val_result.parity + val_result.equality
                < self.best_val_metrics.fair.parity
                + self.best_val_metrics.fair.equality
            )
            and val_result.acc >= self.min_acc
            and val_result.roc >= self.min_roc
        ):
            torch.save(self.fair_gnn, self.log_dir / f"gnn_epoch{curr_epoch}.pt")

        if self.best_val_metrics.update_metrics(val_result, self.min_acc, self.min_roc):
            self.best_fair = result
            self.best_output = output
            self.best_epoch = curr_epoch
            pbar.write(f"[{curr_epoch}] updated to: {self.best_fair}")

    def _print_metric(metric, name):
        print(f"Best {name} model:")
        print(f"\tacc: {metric.acc:.04f}")
        print(f"\troc: {metric.roc:.04f}")
        print(f"\tparity: {metric.parity:.04f}")
        print(f"\tequality: {metric.equality:.04f}")

        if metric.consistency is not None:
            print(f"\tconsistency: {metric.consistency:.04f}")
