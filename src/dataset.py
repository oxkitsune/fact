import torch
from torch.utils.data import Dataset

import numpy as np
import scipy.sparse as sp
import torch
import os
import pandas as pd
import dgl

from pathlib import Path

# if args.dataset != "nba":
#     # if args.dataset == "pokec_z":
#     #     dataset = "region_job"
#     #     embedding = np.load(
#     #         "pokec_z_embedding10.npy"
#     #     )  # embedding is produced by Deep Walk
#     #     embedding = torch.tensor(embedding)
#     #     sens_attr = args.sens_attr_pokec
#     # else:
#     #     dataset = "region_job_2"
#     #     embedding = np.load(
#     #         "pokec_n_embedding10.npy"
#     #     )  # embedding is produced by Deep Walk
#     #     embedding = torch.tensor(embedding)
#     #     sens_attr = args.sens_attr_pokec
#     # predict_attr = "I_am_working_in_field"
#     # label_number = args.label_number
#     # sens_number = args.sens_number
#     # seed = 20
#     # path = "../dataset/pokec/"
#     # test_idx = False
# else:
#     # dataset = "nba"
#     sens_attr = args.sens_attr_nba
#     predict_attr = "SALARY"
#     label_number = 100
#     sens_number = 50
#     seed = 42
#     path = "../dataset/NBA"
#     test_idx = True
#     embedding = np.load("nba_embedding10.npy")  # embedding is produced by Deep Walk
#     embedding = torch.tensor(embedding)
# print(dataset)

# adj, features, labels, idx_train, _, idx_test, sens, _ = load_pokec(
#     dataset,
#     sens_attr,
#     predict_attr,
#     path=path,
#     label_number=label_number,
#     sens_number=sens_number,
#     seed=seed,
#     test_idx=test_idx,
# )

# # remove idx_test adj, features
# exclude_test = torch.ones(adj.shape[1]).bool()  # indices after removing idx_test
# exclude_test[idx_test] = False
# sub_adj = adj[exclude_test][:, exclude_test]
# indices = []
# counter = 0
# for e in exclude_test:
#     indices.append(counter)
#     if e:
#         counter += 1
# indices = torch.LongTensor(indices)
# y_idx = indices[idx_train]
# # ################ modification on dataset idx######################
# print(len(idx_test))

# from utils import feature_norm

# # G = dgl.DGLGraph()
# G = dgl.from_scipy(adj, device="cuda:0")
# subG = dgl.from_scipy(sub_adj, device="cuda:0")

# if dataset == "nba":
#     features = feature_norm(features)

# labels[labels > 1] = 1
# if sens_attr:
#     sens[sens > 0] = 1


class NBADataset(Dataset):
    def __init__(
        self,
        sens_attr="country",
        predict_attr="SALARY",
        label_number=100,
        sens_number=50,
        seed=42,
    ):
        pass


class PokecNDataset(Dataset):
    def __init__(self, sens_attr="region", predict_attr="SALARY"):
        pass


class PokecZDataset(Dataset):
    pass


def load(
    nodes_path: Path,
    edges_path: Path,
    sens_attr: str,
    predict_attr: str,
    label_number=1000,
    sens_number=500,
    seed=19,
    test_idx=False,
):
    """Load data"""
    idx_features_labels = pd.read_csv(nodes_path)

    header = list(idx_features_labels.columns)
    header.remove("user_id")
    header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(edges_path), dtype=int)

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

    adj = adj + sp.eye(adj.shape[0])
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    label_idx = np.where(labels >= 0)[0]

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

    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train
