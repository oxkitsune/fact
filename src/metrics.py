import numpy as np
from dataclasses import dataclass

from typing import Optional
from utils import calculate_similarity_matrix

import torch


def consistency_metric_og(adj, test_idx: list, y_pred_test):
    test_idx = test_idx
    y_pred_test = y_pred_test
    x_similarity = calculate_similarity_matrix(adj, metric="cosine")
    x_similarity = x_similarity.toarray()

    numerator = 0
    denominator = 0
    for i in test_idx:
        for j in test_idx:
            denominator += x_similarity[i][j]
            indicator = 0 if y_pred_test[i] == y_pred_test[j] else 1
            numerator += indicator * x_similarity[i][j]
    consistency = 1 - numerator / denominator
    return consistency


def consistency_metric(adj, test_idx, y_pred_test):
    x_similarity = torch.tensor(
        calculate_similarity_matrix(adj, metric="cosine").toarray()
    ).cpu()
    sim_scores = x_similarity[:, test_idx][test_idx].cpu()
    indicators = (
        (y_pred_test[test_idx].unsqueeze(1).ne(y_pred_test[test_idx].unsqueeze(0)))
        .float()
        .squeeze(2)
    ).cpu()
    numerator = (indicators * sim_scores).sum().cpu()
    denominator = sim_scores.sum().cpu()

    consistency = 1 - numerator / denominator
    return consistency.item()


def fair_metric(output, idx, labels, sens):
    val_y = labels[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()] == 0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()] == 1

    idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)

    pred_y = (output[idx].squeeze() > 0).type_as(labels).cpu().numpy()
    parity = abs(sum(pred_y[idx_s0]) / sum(idx_s0) - sum(pred_y[idx_s1]) / sum(idx_s1))
    equality = abs(
        sum(pred_y[idx_s0_y1]) / sum(idx_s0_y1)
        - sum(pred_y[idx_s1_y1]) / sum(idx_s1_y1)
    )

    return parity, equality


def accuracy(output, labels):
    output = output.squeeze()
    preds = (output > 0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


@dataclass
class Metrics:
    epoch: int
    acc: float
    roc: float
    parity: float
    equality: float
    # since consistency is slow to calculate, we only do it on the best model
    consistency: Optional[float]


@dataclass
class BestMetrics:
    fair: Optional[Metrics]
    acc: Optional[Metrics]
    auc: Optional[Metrics]
    ar: Optional[Metrics]

    def update_metrics(self, metrics: Metrics, min_acc: float, min_roc: float):
        best_fair_has_changed = False
        if self.acc is None or metrics.acc > self.acc.acc:
            self.acc = metrics

        if self.auc is None or metrics.roc > self.auc.roc:
            self.auc = metrics

        if self.ar is None or metrics.acc + metrics.roc > self.ar.acc + self.ar.roc:
            self.ar = metrics

        if (
            (
                self.fair is None
                or metrics.parity + metrics.equality
                < self.fair.parity + self.fair.equality
            )
            and metrics.acc > min_acc
            and metrics.roc > min_roc
        ):
            self.fair = metrics
            best_fair_has_changed = True
        return best_fair_has_changed
