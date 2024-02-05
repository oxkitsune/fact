import numpy as np
from dataclasses import dataclass
from typing import Optional
import torch
from scipy.sparse.csgraph import laplacian
import networkx as nx
import sklearn.preprocessing as skpp



def consistency_metric(adj, test_idx, y_pred_test):
    """Computes consistency metric"""
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
    """Computes dSP and dEO metrics"""
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
    """Computes accuracy"""
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

    def update_metrics(self, metrics: Metrics, min_acc: float, min_roc: float) -> bool:
        """
        Updates best metrics

        Returns true if the best fair has been updated
        """
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

def calculate_similarity_matrix(
    adj, metric=None, filterSigma=None, normalize=None, largestComponent=False
):
    if metric in ["cosine", "jaccard"]:
        # build similarity matrix
        if largestComponent:
            graph = nx.from_scipy_sparse_matrix(adj)
            lcc = max(
                nx.connected_components(graph), key=len
            )  # take largest connected components
            adj = nx.to_scipy_sparse_matrix(
                graph, nodelist=lcc, dtype="float", format="csc"
            )
        sim = get_similarity_matrix(adj, metric=metric)
        if filterSigma:
            sim = filter_similarity_matrix(sim, sigma=filterSigma)
        if normalize:
            sim = symmetric_normalize(sim)
    return sim


def get_similarity_matrix(mat, metric=None):
    """
    get similarity matrix of nodes in specified metric
    :param mat: scipy.sparse matrix (csc, csr or coo)
    :param metric: similarity metric
    :return: similarity matrix of nodes
    """
    if metric == "jaccard":
        return jaccard_similarity(mat.tocsc())
    elif metric == "cosine":
        return cosine_similarity(mat.tocsc())
    else:
        raise ValueError("Please specify the type of similarity metric.")


def filter_similarity_matrix(sim, sigma):
    """
    filter value by threshold = mean(sim) + sigma * std(sim)
    :param sim: similarity matrix
    :param sigma: hyperparameter for filtering values
    :return: filtered similarity matrix
    """
    sim_mean = np.mean(sim.data)
    sim_std = np.std(sim.data)
    threshold = sim_mean + sigma * sim_std
    sim.data *= sim.data >= threshold  # filter values by threshold
    sim.eliminate_zeros()
    return sim


def symmetric_normalize(mat):
    """
    symmetrically normalize a matrix
    :param mat: scipy.sparse matrix (csc, csr or coo)
    :return: symmetrically normalized matrix
    """
    degrees = np.asarray(mat.sum(axis=0).flatten())
    degrees = np.divide(1, degrees, out=np.zeros_like(degrees), where=degrees != 0)
    degrees = np.diags(np.asarray(degrees)[0, :])
    degrees.data = np.sqrt(degrees.data)
    return degrees @ mat @ degrees


def jaccard_similarity(mat):
    """
    get jaccard similarity matrix
    :param mat: scipy.sparse.csc_matrix
    :return: similarity matrix of nodes
    """
    # make it a binary matrix
    mat_bin = mat.copy()
    mat_bin.data[:] = 1

    col_sum = mat_bin.getnnz(axis=0)
    ab = mat_bin.dot(mat_bin.T)
    aa = np.repeat(col_sum, ab.getnnz(axis=0))
    bb = col_sum[ab.indices]
    sim = ab.copy()
    sim.data /= aa + bb - ab.data
    return sim


def cosine_similarity(mat):
    """
    get cosine similarity matrix
    :param mat: scipy.sparse.csc_matrix
    :return: similarity matrix of nodes
    """
    mat_row_norm = skpp.normalize(mat, axis=1)
    sim = mat_row_norm.dot(mat_row_norm.T)
    return sim


def calculate_group_lap(sim, sens):
    unique_sens = [int(x) for x in sens.unique(sorted=True).tolist()]
    num_unique_sens = sens.unique().shape[0]
    sens = [int(x) for x in sens.tolist()]
    m_list = [0] * num_unique_sens
    avgSimD_list = [[] for i in range(num_unique_sens)]
    sim_list = [sim.copy() for i in range(num_unique_sens)]

    for row, col in zip(*sim.nonzero()):
        sensRow = unique_sens[sens[row]]
        sensCol = unique_sens[sens[col]]
        if sensRow == sensCol:
            sim_list[sensRow][row, col] = 2 * sim_list[sensRow][row, col]
            sim_to_zero_list = [x for x in unique_sens if x != sensRow]
            for sim_to_zero in sim_to_zero_list:
                sim_list[sim_to_zero][row, col] = 0
            m_list[sensRow] += 1
        else:
            m_list[sensRow] += 0.5
            m_list[sensRow] += 0.5

    lap = laplacian(sim)
    lap = lap.tocsr()
    for i in range(lap.shape[0]):
        sen_label = sens[i]
        avgSimD_list[sen_label].append(lap[i, i])
    avgSimD_list = [np.mean(s) for s in avgSimD_list]

    lap_list = [laplacian(sim) for sim in sim_list]

    return lap_list, m_list, avgSimD_list