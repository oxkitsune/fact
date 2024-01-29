import torch

from sklearn.model_selection import train_test_split

import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader
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


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]

    return 2 * (features - min_values).div(max_values - min_values) - 1


class FairACDataset(Dataset):
    def __init__(
        self,
        nodes_path: Path,
        edges_path: Path,
        embedding_path: Path,
        feat_drop_rate: float,
        sens_attr: str,
        predict_attr: str,
        label_number: int,
        sens_number: int,
        # number of samples for ac training
        sample_number: int,
        test_idx: bool,
        normalize_features: bool,
        data_seed: int,
    ):
        self.feat_drop_rate = feat_drop_rate
        self.sample_number = sample_number
        self.embeddings = torch.tensor(np.load(embedding_path))

        adj, features, labels, sens, train_idx, test_idx = load(
            nodes_path,
            edges_path,
            sens_attr,
            predict_attr,
            label_number,
            sens_number,
            test_idx,
            # TODO: fact check
            # we want to do shuffling in the dataloader
            # shuffle=False,
            data_seed=data_seed,
        )

        if normalize_features:
            features = feature_norm(features)

        labels[labels > 1] = 1
        if sens_attr:
            sens[sens > 0] = 1

        self.adj = torch.tensor(adj.toarray(), dtype=torch.float)
        self.sub_nodes = list(torch.chunk(torch.arange(features.shape[0]), 4))

        # create fair subgraph adj for each graph
        self.sub_adjs = []
        self.sub_keep_indices = []
        self.sub_drop_indices = []
        for sub_node in self.sub_nodes:
            keep_indices, drop_indices = train_test_split(
                np.arange(len(sub_node)), test_size=feat_drop_rate
            )
            self.sub_keep_indices.append(keep_indices)
            self.sub_drop_indices.append(drop_indices)
            self.sub_adjs.append(self.adj[sub_node][:, sub_node][:, keep_indices])

        mask = torch.zeros(adj.shape[1]).bool()
        mask[train_idx] = True

        # mask for removing test values
        self.mask = mask

        # get y_idx indices
        indices = []
        counter = 0
        for e in mask:
            indices.append(counter)
            if e:
                counter += 1
        indices = torch.tensor(indices)

        self.y_idx = indices[train_idx]
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.labels = labels

        self.graph = dgl.from_scipy(adj)
        self.train_sub_graph = dgl.from_scipy(adj[mask][:, mask])
        self.features = features
        self.sens = sens

    def __len__(self):
        return len(self.sub_nodes)

    # gets a sub node, can be enumerated through with a dataloader
    def __getitem__(self, index):
        sub_adj = self.sub_adjs[index]
        sub_node = self.sub_nodes[index]
        embeddings = self.embeddings[sub_node]
        features = self.features[sub_node]
        keep_indices = self.sub_keep_indices[index]
        drop_indices = self.sub_drop_indices[index]

        return sub_adj, sub_node, embeddings, features, keep_indices, drop_indices

    def sample_ac(self):
        # sub_nodes[0][keep] is fully labeled
        ac_train_indices = self.sub_nodes[0][self.sub_keep_indices[0]][
            : self.sample_number
        ]
        keep_indices, drop_indices = train_test_split(
            np.arange(ac_train_indices.shape[0]), test_size=self.feat_drop_rate
        )

        train_adj = self.adj[ac_train_indices][:, ac_train_indices][:, keep_indices]
        embeddings = self.embeddings[ac_train_indices]
        features = self.features[ac_train_indices]
        sens = self.sens[ac_train_indices]

        return train_adj, embeddings, features, sens, keep_indices, drop_indices

    def inside_labels(self):
        return (
            self.y_idx,
            self.train_idx,
            self.labels,
        )


class NBA(FairACDataset):
    def __init__(
        self,
        nodes_path: Path,
        edges_path: Path,
        embedding_path: Path,
        feat_drop_rate: float,
        sens_attr="country",
        predict_attr="SALARY",
        label_number=100,
        sens_number=50,
        sample_number=1000,
        test_idx=True,
        normalize_features=True,
        data_seed=42,
    ):
        super().__init__(
            nodes_path,
            edges_path,
            embedding_path,
            feat_drop_rate,
            sens_attr,
            predict_attr,
            label_number,
            sens_number,
            sample_number,
            test_idx,
            normalize_features,
            data_seed,
        )


class PokecN(FairACDataset):
    def __init__(
        self,
        nodes_path: Path,
        edges_path: Path,
        embedding_path: Path,
        feat_drop_rate: float,
        sens_attr="region",
        predict_attr="I_am_working_in_field",
        label_number=100,
        sens_number=50,
        sample_number=1000,
        test_idx=False,
        normalize_features=True,
        data_seed=20,
    ):
        super().__init__(
            nodes_path,
            edges_path,
            embedding_path,
            feat_drop_rate,
            sens_attr,
            predict_attr,
            label_number,
            sens_number,
            sample_number,
            test_idx,
            normalize_features,
            data_seed,
        )


class PokecZ(FairACDataset):
    def __init__(
        self,
        nodes_path: Path,
        edges_path: Path,
        embedding_path: Path,
        feat_drop_rate: float,
        sens_attr="region",
        predict_attr="I_am_working_in_field",
        label_number=100,
        sens_number=50,
        sample_number=1000,
        test_idx=False,
        normalize_features=True,
        data_seed=20,
    ):
        super().__init__(
            nodes_path,
            edges_path,
            embedding_path,
            feat_drop_rate,
            sens_attr,
            predict_attr,
            label_number,
            sens_number,
            sample_number,
            test_idx,
            normalize_features,
            data_seed,
        )


def load(
    nodes_path: Path,
    edges_path: Path,
    sens_attr: str,
    predict_attr: str,
    label_number=1000,
    sens_number=500,
    test_idx=False,
    # shuffle=True,
    data_seed=19,
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

    features = torch.tensor(np.array(features.todense()))
    labels = torch.tensor(labels)

    import random

    random.seed(data_seed)
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
    sens = torch.tensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))

    random.seed(data_seed)
    random.shuffle(idx_sens_train)

    idx_sens_train = torch.tensor(idx_sens_train[:sens_number])

    idx_train = torch.tensor(idx_train)
    idx_val = torch.tensor(idx_val)
    idx_test = torch.tensor(idx_test)

    return adj, features, labels, sens, idx_train, idx_test


if __name__ == "__main__":
    # dataset = NBA(
    #     "./dataset/NBA/nba.csv",
    #     "./dataset/NBA/nba_relationship.txt",
    #     "./src/nba_embedding10.npy",
    #     feat_drop_rate=0.3,
    # )

    dataset = PokecZ(
        "./dataset/pokec/region_job.csv",
        "./dataset/pokec/region_job_relationship.txt",
        "./src/pokec_z_embedding10.npy",
        feat_drop_rate=0.3,
    )

    loader = DataLoader(dataset)

    for i, (sub_node, embeddings, features, keep_indices, drop_indices) in enumerate(
        loader
    ):
        print(sub_node)

    print(dataset.sample_ac())
    print(dataset.inside_labels())
