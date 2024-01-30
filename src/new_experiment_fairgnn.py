from models.gnn import FairGNN, FairGNNTrainer
from dataset import NBA

from pathlib import Path

import torch
import numpy as np


SEED = 20
np.random.seed(SEED)
torch.manual_seed(SEED)

data_path = Path("./dataset/NBA/")
log_dir = Path("./logs/test_run")

dataset = NBA(
    nodes_path=data_path / "nba.csv",
    edges_path=data_path / "nba_relationship.txt",
    embedding_path=data_path / "nba_embedding10.npy",
    feat_drop_rate=0.3,
)

fair_gnn = FairGNN(
    num_features=dataset.features.shape[1],
)

fair_gnn.estimator.load_state_dict(torch.load("./checkpoint/GCN_sens_nba_ns_50"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainer = FairGNNTrainer(
    dataset=dataset,
    fair_gnn=fair_gnn,
    device=device,
    log_dir=log_dir,
    min_acc=0.65,
    min_roc=0.69,
)

trainer.pretrain(epochs=200)
trainer.train(val_start_epoch=800, val_epoch_interval=200, epochs=2800)
