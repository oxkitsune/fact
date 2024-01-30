import torch

from sklearn.model_selection import train_test_split

import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import dgl

from pathlib import Path

import requests

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
        device: str,
        remove_user_id=False,
    ):
        self.feat_drop_rate = feat_drop_rate
        self.sample_number = sample_number
        self.embeddings = torch.tensor(np.load(embedding_path), device=device)
        self.device = device

        adj, features, labels, sens, train_idx, test_idx, idx_sens_train = load(
            nodes_path,
            edges_path,
            sens_attr,
            predict_attr,
            device,
            label_number,
            sens_number,
            # shuffle=False,
            data_seed=data_seed,
            remove_user_id=remove_user_id,
        )

        if normalize_features:
            features = feature_norm(features)

        labels[labels > 1] = 1
        if sens_attr:
            sens[sens > 0] = 1

        self.adj = torch.tensor(adj.toarray(), dtype=torch.float).to(device)
        self.sub_nodes = list(torch.chunk(torch.arange(features.shape[0]), 4))

        # create fair subgraph adj for each graph
        self.sub_adjs = []
        self.sub_keep_indices = []
        self.sub_drop_indices = []
        for sub_node in self.sub_nodes:
            keep_indices, drop_indices = train_test_split(
                np.arange(len(sub_node)), test_size=feat_drop_rate
            )
            self.sub_keep_indices.append(torch.tensor(keep_indices, device=device))
            self.sub_drop_indices.append(torch.tensor(drop_indices, device=device))
            self.sub_adjs.append(
                self.adj[sub_node][:, sub_node][:, keep_indices].clone().detach().cpu()
            )

        mask = torch.zeros(adj.shape[1]).to(device=device, dtype=torch.bool)
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
        indices = torch.tensor(indices, device=device, dtype=torch.long)

        self.y_idx = indices[train_idx]
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.sens_train_idx = idx_sens_train
        self.labels = labels

        self.graph = dgl.from_scipy(adj, device=device)
        self.train_sub_graph = dgl.from_scipy(
            adj[mask.cpu()][:, mask.cpu()], device=device
        )
        self.features = features
        self.sens = sens

    def __len__(self):
        return len(self.sub_nodes)

    # gets a sub node, can be enumerated through with a dataloader
    def __getitem__(self, index):
        sub_node = self.sub_nodes[index]
        keep_indices = self.sub_keep_indices[index]
        drop_indices = self.sub_drop_indices[index]

        sub_adj = self.sub_adjs[index].to(self.device)
        embeddings = self.embeddings[sub_node]
        features = self.features[sub_node]

        return sub_adj, sub_node, embeddings, features, keep_indices, drop_indices

    def sample_fairac(self):
        # sub_nodes[0][keep] is fully labeled
        ac_train_indices = self.sub_nodes[0][self.sub_keep_indices[0]][
            : self.sample_number
        ]

        keep_indices, drop_indices = train_test_split(
            np.arange(ac_train_indices.shape[0]), test_size=self.feat_drop_rate
        )

        keep_indices = torch.tensor(keep_indices, device=self.device)
        drop_indices = torch.tensor(drop_indices, device=self.device)

        train_adj = self.adj[ac_train_indices][:, ac_train_indices][:, keep_indices]
        embeddings = self.embeddings[ac_train_indices]
        features = self.features[ac_train_indices]
        sens = self.sens[ac_train_indices]

        return train_adj, embeddings, features, sens, keep_indices, drop_indices

    def sample_full(self):
        keep_indices = torch.cat(self.sub_keep_indices)
        drop_indices = torch.cat(self.sub_drop_indices)

        return (
            self.adj,
            self.embeddings,
            self.features,
            self.sens,
            keep_indices,
            drop_indices,
        )

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
        device,
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
            device=device,
            remove_user_id=True,
        )


class PokecN(FairACDataset):
    def __init__(
        self,
        nodes_path: Path,
        edges_path: Path,
        embedding_path: Path,
        feat_drop_rate: float,
        device,
        sens_attr="region",
        predict_attr="I_am_working_in_field",
        label_number=100,
        sens_number=50,
        sample_number=1000,
        test_idx=False,
        normalize_features=False,
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
            device=device,
            remove_user_id=True,
        )


class PokecZ(FairACDataset):
    def __init__(
        self,
        nodes_path: Path,
        edges_path: Path,
        embedding_path: Path,
        feat_drop_rate: float,
        device,
        sens_attr="region",
        predict_attr="I_am_working_in_field",
        label_number=100,
        sens_number=50,
        sample_number=1000,
        test_idx=False,
        normalize_features=False,
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
            device=device,
            remove_user_id=True,
        )


class Recidivism(FairACDataset):
    def __init__(
        self,
        feat_drop_rate: float,
        device,
        embedding_path: Path,
        download: bool = True,
        nodes_path: Path = None,
        edges_path: Path = None,
        sens_attr="WHITE",
        predict_attr="RECID",
        label_number=100,
        sens_number=50,
        sample_number=1000,
        test_idx=False,
        normalize_features=False,
        data_seed=20,
    ):
        if download:
            dataset_dir = Path("./dataset/bail/")

            if not dataset_dir.exists():
                dataset_dir.mkdir(parents=True, exist_ok=True)

            nodes_name = "bail.csv"
            edges_name = "bail_edges.txt"

            nodes_path = dataset_dir / nodes_name
            edges_path = dataset_dir / edges_name

            # check if the dataset exists
            if not nodes_path.exists():
                url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/bail/bail.csv"
                download_dataset(dataset_dir, url, nodes_name)

            if not edges_path.exists():
                url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/bail/bail_edges.txt"
                download_dataset(dataset_dir, url, edges_name)

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
            device=device,
        )


class Credit(FairACDataset):
    def __init__(
        self,
        feat_drop_rate: float,
        device,
        embedding_path: Path,
        download: bool = True,
        nodes_path: Path = None,
        edges_path: Path = None,
        sens_attr="Age",
        predict_attr="NoDefaultNextMonth",
        label_number=6000,
        sens_number=500,
        sample_number=1000,
        test_idx=False,
        normalize_features=False,
        data_seed=20,
    ):
        if download:
            dataset_dir = Path("./dataset/credit/")

            if not dataset_dir.exists():
                dataset_dir.mkdir(parents=True, exist_ok=True)

            nodes_name = "credit.csv"
            edges_name = "credit_edges.txt"

            nodes_path = dataset_dir / nodes_name
            edges_path = dataset_dir / edges_name

            # check if the dataset exists
            if not nodes_path.exists():
                url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/credit/credit.csv"
                download_dataset(dataset_dir, url, nodes_name)

            if not edges_path.exists():
                url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/bail/bail_edges.txt"
                download_dataset(dataset_dir, url, edges_name)

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
            device=device,
        )


def download_dataset(path: Path, url: str, filename: str):
    """Download file from url to path/filename"""
    r = requests.get(url)
    assert r.status_code == 200
    with open(path / filename, "wb") as f:
        f.write(r.content)

    print(f"Downloaded {filename}")


def load(
    nodes_path: Path,
    edges_path: Path,
    sens_attr: str,
    predict_attr: str,
    device,
    label_number=1000,
    sens_number=500,
    test_idx=False,
    # shuffle=True,
    data_seed=19,
    remove_user_id=False,
):
    """Load data"""
    idx_features_labels = pd.read_csv(nodes_path)

    header = list(idx_features_labels.columns)
    if remove_user_id:
        header.remove("user_id")

    header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    # build graph
    if remove_user_id:
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(os.path.join(edges_path), dtype=int)
        edges = (
            np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int)
            # .astype(int)
            .reshape(edges_unordered.shape)
        )
    else:
        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(os.path.join(edges_path)).astype(int)
        edges = (
            np.array(list(map(idx_map.get, edges_unordered.flatten())))
            .astype(int)
            .reshape(edges_unordered.shape)
        )

    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.tensor(np.array(features.todense()), device=device)
    labels = torch.tensor(labels, device=device)

    import random

    random.seed(data_seed)
    label_idx = np.where(labels.cpu() >= 0)[0]
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
    sens = torch.tensor(sens, device=device)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))

    random.seed(data_seed)
    random.shuffle(idx_sens_train)

    idx_sens_train = torch.tensor(idx_sens_train[:sens_number], device=device)

    idx_train = torch.tensor(idx_train, device=device)
    idx_val = torch.tensor(idx_val, device=device)
    idx_test = torch.tensor(idx_test, device=device)

    return adj, features, labels, sens, idx_train, idx_test, idx_sens_train


if __name__ == "__main__":
    # dataset = NBA(
    #     "./dataset/NBA/nba.csv",
    #     "./dataset/NBA/nba_relationship.txt",
    #     "./src/nba_embedding10.npy",
    #     feat_drop_rate=0.3,
    #     device="cpu",
    # )

    # dataset = PokecN(
    #     "./dataset/pokec/region_job_2.csv",
    #     "./dataset/pokec/region_job_2_relationship.txt",
    #     "./src/pokec_n_embedding10.npy",
    #     feat_drop_rate=0.3,
    #     device="cpu",
    # )

    # dataset = PokecZ(
    #     "./dataset/pokec/region_job.csv",
    #     "./dataset/pokec/region_job_relationship.txt",
    #     "./src/pokec_z_embedding10.npy",
    #     feat_drop_rate=0.3,
    #     device="cpu",
    # )

    # dataset = Recidivism(
    #     feat_drop_rate=0.3,
    #     embedding_path="./dataset/bail/deepwalk_emb-20240115-155136-wl=100-dim=64-ep=10.npy",
    #     device="cpu",
    # )

    # dataset = Credit(
    #     feat_drop_rate=0.3,
    #     embedding_path="./dataset/credit/deepwalk_emb-20240125-114643-wl=100-dim=64-ep=10.npy",
    #     device="cpu",
    # )

    # loader = DataLoader(dataset)

    # for i, sub_node in enumerate(loader):
    #     print(i, sub_node)

    # print(dataset.sample_ac())
    # print(dataset.inside_labels())
    pass
