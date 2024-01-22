from models.ac import FairAC, Trainer
from dataset import NBADataset

from pathlib import Path

dataset = NBADataset(
    nodes_path=Path("dataset/NBA/nba.csv"),
    edges_path=Path("dataset/NBA/nba_relationship.txt"),
    embedding_path=Path("dataset/NBA/nba.embedding"),
    feat_drop_rate=0.2,
    train=True,
)

fair_ac = FairAC(
    feature_dim=len(dataset.embeddings),
    transformed_feature_dim=128,
    emb_dim=len(dataset.embeddings),
    attn_vec_dim=128,
    attn_num_heads=1,
    dropout=0.5,
    num_sensitive_classes=1,
)

trainer = Trainer(fair_ac, dataset="nba")
trainer.pretrain(epochs=200)
# trainer.train(acc=0.7, fairness=0.7)
