import numpy as np
from dataclasses import dataclass

from typing import Optional


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
    acc: float
    roc: float
    parity: float
    equality: float


@dataclass
class BestMetrics:
    best_fair: Optional[Metrics]
    acc: Optional[Metrics]
    auc: Optional[Metrics]
    ar: Optional[Metrics]

    def update_metrics(self, metrics: Metrics, min_acc: float, min_roc: float):
        if self.acc is None or metrics.acc > self.acc.acc:
            self.acc = metrics

        if self.auc is None or metrics.roc > self.auc.roc:
            self.auc = metrics

        if self.ar is None or metrics.acc + metrics.roc > self.ar.acc + self.ar.roc:
            self.ar = metrics

        if (
            (
                self.best_fair is None
                or metrics.parity + metrics.equality
                < self.best_fair.parity + self.best_fair.equality
            )
            and metrics.acc >= min_acc
            and metrics.roc >= min_roc
        ):
            self.best_fair = metrics
