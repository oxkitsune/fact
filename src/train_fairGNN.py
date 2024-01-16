# %%
import time
import datetime
import os
import json
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split

from utils import load_data, load_pokec
from models.FairGNN import FairGNN

from metrics import accuracy

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="Disables CUDA training."
)
parser.add_argument(
    "--fastmode",
    action="store_true",
    default=False,
    help="Validate during training pass.",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--epochs", type=int, default=2000, help="Number of epochs to train."
)
parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate.")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=1e-5,
    help="Weight decay (L2 loss on parameters).",
)
parser.add_argument(
    "--hidden",
    type=int,
    default=128,
    help="Number of hidden units of the sensitive attribute estimator",
)
parser.add_argument(
    "--dropout", type=float, default=0.5, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--alpha", type=float, default=4, help="The hyperparameter of alpha"
)
parser.add_argument(
    "--beta", type=float, default=0.01, help="The hyperparameter of beta"
)
parser.add_argument(
    "--model", type=str, default="GAT", help="the type of model GCN/GAT"
)
parser.add_argument(
    "--dataset", type=str, default="pokec_n", choices=["pokec_z", "pokec_n", "nba"]
)
parser.add_argument(
    "--num-hidden", type=int, default=64, help="Number of hidden units of classifier."
)
parser.add_argument(
    "--num-heads", type=int, default=1, help="number of hidden attention heads"
)
parser.add_argument(
    "--num-out-heads", type=int, default=1, help="number of output attention heads"
)
parser.add_argument("--num-layers", type=int, default=1, help="number of hidden layers")
parser.add_argument(
    "--residual", action="store_true", default=False, help="use residual connection"
)
parser.add_argument("--attn-drop", type=float, default=0.0, help="attention dropout")
parser.add_argument(
    "--negative-slope", type=float, default=0.2, help="the negative slope of leaky relu"
)
parser.add_argument(
    "--acc",
    type=float,
    default=0.688,
    help="the selected FairGNN accuracy on val would be at least this high",
)
parser.add_argument(
    "--roc",
    type=float,
    default=0.745,
    help="the selected FairGNN ROC score on val would be at least this high",
)
parser.add_argument(
    "--sens_number", type=int, default=200, help="the number of sensitive attributes"
)
parser.add_argument(
    "--label_number", type=int, default=500, help="the number of labels"
)
parser.add_argument("--save_dir", type=str, default="./logs", help="save directory")
parser.add_argument(
    "--feat_drop_rate", type=float, default=0.3, help="feature dropout rate"
)

args = parser.parse_known_args()[0]

print(f"Feature drop rate: {args.feat_drop_rate}")

save_dir = args.save_dir
save_folder = datetime.datetime.now()
save_path = f"{save_dir}/{save_folder}/"
test_log_file = os.path.join(save_path, "testlog.txt")

if not os.path.exists(save_path):
    os.makedirs(save_path)

with open(os.path.join(save_path, "hparams.json"), "w") as f:
    json.dump(vars(args), f, indent=4)

args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)
# %%
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
print(args.dataset)

if args.dataset != "nba":
    if args.dataset == "pokec_z":
        dataset = "region_job"
    else:
        dataset = "region_job_2"
    sens_attr = "region"
    predict_attr = "I_am_working_in_field"
    label_number = args.label_number
    sens_number = args.sens_number
    seed = 20
    path = "../dataset/pokec/"
    test_idx = False
else:
    dataset = "nba"
    sens_attr = "country"
    predict_attr = "SALARY"
    label_number = 100
    sens_number = 50
    seed = 20
    path = "../dataset/NBA"
    test_idx = True
print(dataset)

adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_pokec(
    dataset,
    sens_attr,
    predict_attr,
    path=path,
    label_number=label_number,
    sens_number=sens_number,
    seed=seed,
    test_idx=test_idx,
)
print(len(idx_test))

# %%
import dgl
from utils import feature_norm

# exclude_test = torch.ones(adj.shape[1]).bool()
# exclude_test[idx_test] = False
# sub_adj = adj[exclude_test][:, exclude_test]
adj_mat = adj.toarray()
adjTensor = torch.FloatTensor(adj_mat)
sub_nodes = np.array_split(range(features.shape[0]), 4)
sub_nodes = [torch.tensor(s).cuda() for s in sub_nodes]

G = dgl.from_scipy(adj, device="cuda:0")
if dataset == "nba":
    features = feature_norm(features)


def fair_metric(output, idx):
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


# %%
labels[labels > 1] = 1
if sens_attr:
    sens[sens > 0] = 1
# Model and optimizer

model = FairGNN(nfeat=features.shape[1], args=args)
model.estimator.load_state_dict(
    torch.load("./checkpoint/GCN_sens_{}_ns_{}".format(dataset, sens_number))
)
if args.cuda:
    model.cuda()
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    sens = sens.cuda()
    idx_sens_train = idx_sens_train.cuda()


# Generate keep/drop sets
subgraph_adj_list = []
feat_keep_idx_sub_list = []
feat_drop_idx_sub_list = []

# print(f"Drop rate: {args.feat_drop_rate}")
# print(f"Amount of sub nodes: {len(sub_nodes)}")

for sub_node in sub_nodes:
    # print(f"Length sub node: {len(sub_node)}")
    feat_keep_idx_sub, feat_drop_idx_sub = train_test_split(
        np.arange(len(sub_node)), test_size=args.feat_drop_rate
    )
    feat_keep_idx_sub_list.append(feat_keep_idx_sub)
    feat_drop_idx_sub_list.append(feat_drop_idx_sub)
    subgraph_adj = adjTensor[sub_node][:, sub_node][:, feat_keep_idx_sub]
    subgraph_adj_list.append(subgraph_adj)

    # print(f"Length keep: {len(feat_keep_idx_sub)}")
    # print(f"Length drop: {len(feat_drop_idx_sub)}")

kept = [
    features[sub_node[feat_keep]]
    for sub_node, feat_keep in zip(sub_nodes, feat_keep_idx_sub_list)
]
kept = torch.cat(kept)
mean = kept.mean(dim=0)

for sub_node, feat_drop in zip(sub_nodes, feat_drop_idx_sub_list):
    features[sub_node[feat_drop]] = mean

from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score


# Train model
t_total = time.time()
best_result = {}
best_fair = 100

for epoch in range(args.epochs):
    t = time.time()
    model.train()
    model.optimize(G, features, labels, idx_train, sens, idx_sens_train)
    cov = model.cov
    cls_loss = model.cls_loss
    adv_loss = model.adv_loss
    model.eval()
    output, s = model(G, features)
    acc_val = accuracy(output[idx_val], labels[idx_val])
    roc_val = roc_auc_score(
        labels[idx_val].cpu().numpy(), output[idx_val].detach().cpu().numpy()
    )

    acc_sens = accuracy(s[idx_test], sens[idx_test])

    parity_val, equality_val = fair_metric(output, idx_val)

    acc_test = accuracy(output[idx_test], labels[idx_test])
    roc_test = roc_auc_score(
        labels[idx_test].cpu().numpy(), output[idx_test].detach().cpu().numpy()
    )
    parity, equality = fair_metric(output, idx_test)
    if acc_val > args.acc and roc_val > args.roc:
        if best_fair > parity_val + equality_val:
            best_fair = parity_val + equality_val

            best_result["acc"] = acc_test.item()
            best_result["roc"] = roc_test
            best_result["parity"] = parity
            best_result["equality"] = equality

        print("=================================")

        print(
            "Epoch: {:04d}".format(epoch + 1),
            "cov: {:.4f}".format(cov.item()),
            "cls: {:.4f}".format(cls_loss.item()),
            "adv: {:.4f}".format(adv_loss.item()),
            "acc_val: {:.4f}".format(acc_val.item()),
            "roc_val: {:.4f}".format(roc_val),
            "parity_val: {:.4f}".format(parity_val),
            "equality: {:.4f}".format(equality_val),
        )
        print(
            "Test:",
            "accuracy: {:.4f}".format(acc_test.item()),
            "roc: {:.4f}".format(roc_test),
            "acc_sens: {:.4f}".format(acc_sens),
            "parity: {:.4f}".format(parity),
            "equality: {:.4f}".format(equality),
        )

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

print("============performace on test set=============")
if len(best_result) > 0:
    print(
        "Test:",
        "accuracy: {:.4f}".format(best_result["acc"]),
        "roc: {:.4f}".format(best_result["roc"]),
        "acc_sens: {:.4f}".format(acc_sens),
        "parity: {:.4f}".format(best_result["parity"]),
        "equality: {:.4f}".format(best_result["equality"]),
    )
    with open(test_log_file, "w") as f:
        serialized = json.dumps(best_result, indent=4)
        f.write(serialized)
else:
    print("Please set smaller acc/roc thresholds")
