from models.ac import FairAC, Trainer
from dataset import NBA

from pathlib import Path

import torch
import numpy as np

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

data_path = Path("./dataset/NBA/")

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

trainer = Trainer(ac_model=fair_ac, lambda2=0.7, dataset=dataset)
trainer.pretrain(epochs=200)
trainer.train(epochs=3000, acc=0.7, fairness=0.7)
