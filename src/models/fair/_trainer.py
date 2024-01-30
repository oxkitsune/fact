from dataclasses import asdict
from itertools import chain
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score

import torch
import torch.nn.functional as F
from tqdm import trange

from models.gnn import GNNKind
from models.fair import FairGNN
from metrics import Metrics, accuracy, BestMetrics, fair_metric
import dgl


class FairGNNTrainer:
    def __init__(
        self,
        dataset,
        device,
        fair_gnn: FairGNN,
        log_dir: Path,
        min_acc: float,
        min_roc: float,
        alpha=1,
        beta=1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        self.dataset = dataset
        self.device = device
        self.fair_gnn = fair_gnn
        self.log_dir = log_dir
        self.best_metrics = BestMetrics(None, None, None, None)
        self.min_acc = min_acc
        self.min_roc = min_roc

        self.alpha = alpha
        self.beta = beta

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
        kept = features[keep_indices].mean(dim=0)
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

        if (
            self.best_metrics.best_fair is None
            or self.best_metrics.best_fair.acc < self.min_acc
            or self.best_metrics.best_fair.roc < self.min_roc
        ):
            print("Please set smaller acc/roc thresholds!")
        else:
            print("Finished training!")

        print()
        print("Best fair model:")
        print(f"\tacc: {self.best_metrics.best_fair.acc:.04f}")
        print(f"\troc: {self.best_metrics.best_fair.roc:.04f}")
        print(f"\tparity: {self.best_metrics.best_fair.parity:.04f}")
        print(f"\tequality: {self.best_metrics.best_fair.equality:.04f}")

        print()
        print("Best acc model:")
        print(f"\tacc: {self.best_metrics.acc.acc:.04f}")
        print(f"\troc: {self.best_metrics.acc.roc:.04f}")
        print(f"\tparity: {self.best_metrics.acc.parity:.04f}")
        print(f"\tequality: {self.best_metrics.acc.equality:.04f}")

        print()
        print("Best auc model:")
        print(f"\tacc: {self.best_metrics.auc.acc:.04f}")
        print(f"\troc: {self.best_metrics.auc.roc:.04f}")
        print(f"\tparity: {self.best_metrics.auc.parity:.04f}")
        print(f"\tequality: {self.best_metrics.auc.equality:.04f}")

        with open(self.log_dir / "best_metrics.json", "a") as f:
            json.dump(asdict(self.best_metrics), f, indent=4)

    def optimize(self, adj, features, sens):
        self.fair_gnn.train()

        sens_train_idx = self.dataset.sens_train_idx
        y_idx, train_idx, labels = self.dataset.inside_labels()

        self.fair_gnn.adv.requires_grad_(False)
        self.gnn_optimizer.zero_grad()

        s = self.fair_gnn.estimator(adj, features)
        h, y = self.fair_gnn.gnn(adj, features)

        s_g = self.fair_gnn.adv(h)

        s_score = torch.sigmoid(s.detach())
        s_score[sens_train_idx] = sens[sens_train_idx].unsqueeze(1).float()
        y_score = torch.sigmoid(y)

        gnn_loss = self.fair_gnn.gnn_loss(
            y_score, y[train_idx], labels[train_idx].unsqueeze(1).float(), s_g, s_score
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
        val_idx = self.dataset.val_idx
        test_idx = self.dataset.test_idx
        y_idx, train_idx, labels = self.dataset.inside_labels()

        output, s = self.fair_gnn(adj, features)
        # acc_val = accuracy(output[val_idx], labels[val_idx])
        # roc_val = roc_auc_score(
        #     labels[val_idx].cpu().numpy(), output[val_idx].detach().cpu().numpy()
        # )

        # acc_sens = accuracy(s[test_idx], sens[test_idx])

        parity_val, equality_val = fair_metric(
            output, val_idx, labels=labels, sens=sens
        )

        acc_test = accuracy(output[test_idx], labels[test_idx])
        roc_test = roc_auc_score(
            labels[test_idx].cpu().numpy(), output[test_idx].detach().cpu().numpy()
        )
        parity, equality = fair_metric(output, test_idx, labels=labels, sens=sens)

        pbar.set_postfix_str(
            f"Acc: {acc_test.item():.4f}, Roc: {roc_test:.4f}, Partity: {parity:.4f}, Equality: {equality:.4f}",
        )

        result = Metrics(acc_test.item(), roc_test, parity, equality)

        if (
            (
                self.best_metrics.best_fair is None
                or result.parity + result.equality
                < self.best_metrics.best_fair.parity
                + self.best_metrics.best_fair.equality
            )
            and result.acc >= self.min_acc
            and result.roc >= self.min_roc
        ):
            torch.save(self.fair_gnn, self.log_dir / f"gnn_epoch{curr_epoch:04d}.pt")

        self.best_metrics.update_metrics(result, self.min_acc, self.min_roc)


# # Model and optimizer
# model = GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=1, dropout=args.dropout)
# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# if args.cuda:
#     model.cuda()
#     features = features.cuda()
#     # adj = adj.cuda()
#     sens = sens.cuda()
#     # idx_sens_train = idx_sens_train.cuda()
#     # idx_val = idx_val.cuda()
#     idx_test = idx_test.cuda()
#     sens = sens.cuda()
#     # idx_sens_train = torch.Tensor(idx_sens_train).cuda()

# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# # Train model
# t_total = time.time()
# best_acc = 0.0
# best_test = 0.0
# for epoch in range(args.epochs + 1):
#     t = time.time()
#     model.train()
#     optimizer.zero_grad()
#     features = features.cuda()
#     output = model(G, features)
#     loss_train = F.binary_cross_entropy_with_logits(
#         output[idx_sens_train], sens[idx_sens_train].unsqueeze(1).float()
#     )
#     acc_train = accuracy(output[idx_sens_train], sens[idx_sens_train])
#     loss_train.backward()
#     optimizer.step()

#     if not args.fastmode:
#         # Evaluate validation set performance separately,
#         # deactivates dropout during validation run.
#         model.eval()
#         output = model(G, features)
#     if epoch % 10 == 0:
#         acc_val = accuracy(output[idx_val], sens[idx_val])
#         acc_test = accuracy(output[idx_test], sens[idx_test])
#         print(
#             "Epoch [{}] Test set results:".format(epoch),
#             "acc_test= {:.4f}".format(acc_test.item()),
#             "acc_val: {:.4f}".format(acc_val.item()),
#         )
#         if acc_val > best_acc:
#             best_acc = acc_val
#             best_test = acc_test
#             torch.save(
#                 model.state_dict(),
#                 "./checkpoint/GCN_sens_{}_ns_{}".format(dataset, sens_number),
#             )
# print("The accuracy of estimator: {:.4f}".format(best_test))

# print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
