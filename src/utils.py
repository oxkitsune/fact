# %%
import numpy as np
import scipy.sparse as sp
import torch
import os
import pandas as pd
import dgl
from scipy.sparse.csgraph import laplacian
import networkx as nx
import sklearn.preprocessing as skpp


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


# %%


# %%
def load_data(path="../dataset/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print("Loading {} dataset...".format(dataset))

    idx_features_labels = np.genfromtxt(
        "{}{}.content".format(path, dataset), dtype=np.dtype(str)
    )
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    print(labels)

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
    ).reshape(edges_unordered.shape)
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def load_pokec(
    dataset,
    sens_attr,
    predict_attr,
    path="../dataset/pokec/",
    label_number=1000,
    sens_number=500,
    seed=19,
    test_idx=False,
):
    """Load data"""
    print("Loading {} dataset from {}".format(dataset, path))

    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=int
    )

    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=int
    ).reshape(edges_unordered.shape)
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    import random

    random.seed(seed)
    label_idx = np.where(labels >= 0)[0]
    random.shuffle(label_idx)

    idx_train = label_idx[: min(int(0.5 * len(label_idx)), label_number)]
    idx_val = label_idx[int(0.5 * len(label_idx)) : int(0.75 * len(label_idx))]
    if test_idx:
        idx_test = label_idx[label_number:]
        idx_val = idx_test
    else:
        idx_test = label_idx[int(0.75 * len(label_idx)) :]

    sens = idx_features_labels[sens_attr].values

    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # random.shuffle(sens_idx)

    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]

    return 2 * (features - min_values).div(max_values - min_values) - 1


def accuracy(output, labels):
    output = output.squeeze()
    preds = (output > 0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_softmax(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# %%


# %%
def load_pokec_emb(
    dataset,
    sens_attr,
    predict_attr,
    path="../dataset/pokec/",
    label_number=1000,
    sens_number=500,
    seed=19,
    test_idx=False,
):
    print("Loading {} dataset from {}".format(dataset, path))

    graph_embedding = np.genfromtxt(
        os.path.join(path, "{}.embedding".format(dataset)), skip_header=1, dtype=float
    )
    embedding_df = pd.DataFrame(graph_embedding)
    embedding_df[0] = embedding_df[0].astype(int)
    embedding_df = embedding_df.rename(index=int, columns={0: "user_id"})

    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    idx_features_labels = pd.merge(
        idx_features_labels, embedding_df, how="left", on="user_id"
    )
    idx_features_labels = idx_features_labels.fillna(0)
    # %%

    header = list(idx_features_labels.columns)
    header.remove("user_id")

    header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    # %%
    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=int
    )

    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
    ).reshape(edges_unordered.shape)
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random

    random.seed(seed)
    label_idx = np.where(labels >= 0)[0]
    random.shuffle(label_idx)

    idx_train = label_idx[: min(int(0.5 * len(label_idx)), label_number)]
    idx_val = label_idx[int(0.5 * len(label_idx)) : int(0.75 * len(label_idx))]
    if test_idx:
        idx_test = label_idx[label_number:]
    else:
        idx_test = label_idx[int(0.75 * len(label_idx)) :]

    sens = idx_features_labels[sens_attr].values

    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train


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
