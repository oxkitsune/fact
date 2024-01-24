from models.ac import FairAC, Trainer
from dataset import NBA

from pathlib import Path

import torch
import numpy as np

# seed = 19
# parity = 0.0005
# equality = 0.0024

# seed = 20
# parity = 0.0006
# equality = 0.0015

# seed = 40
# parity = 0.0010
# equality = 0.0019

# seed = 41
# parity = 0.0013
# equality = 0.0024

# seed = 42
# parity = 0.0511
# equality = 0.0920

# seed = 43
# parity = 0.0010
# equality = 0.0019

# seed = 44
# parity = 0.0010
# equality = 0.0019

# seed = 45
# parity = 0.0010
# equality = 0.0019

# seed = 46
# parity = 0.0043
# equality = 0.0024


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

fair_ac = FairAC(
    feature_dim=dataset.features.shape[1],
    transformed_feature_dim=128,
    emb_dim=dataset.embeddings.shape[1],
    attn_vec_dim=128,
    attn_num_heads=1,
    dropout=0.5,
    num_sensitive_classes=1,
)

trainer = Trainer(
    ac_model=fair_ac,
    lambda1=1.0,
    lambda2=1.0,
    dataset=dataset,
    gnn_kind="GCN",
    gnn_hidden_dim=128,
    gnn_lr=1e-3,
    gnn_weight_decay=1e-5,
    gnn_args={"dropout": 0.5},
    log_dir=log_dir,
    min_acc=0.65,
    min_roc=0.69,
)

trainer.pretrain(epochs=200)
trainer.train(val_start_epoch=800, val_epoch_interval=200, epochs=2800)
