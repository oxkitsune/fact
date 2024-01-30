from models.fair import FairGNN, FairGNNTrainer
from dataset import NBA, PokecZ

from pathlib import Path

import torch
import numpy as np


SEED = 29
np.random.seed(SEED)
torch.manual_seed(SEED)

data_path = Path("./dataset/pokec/")
log_dir = Path("./logs/test_run")



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device:", device)

# dataset = NBA(
#     nodes_path=data_path / "nba.csv",
#     edges_path=data_path / "nba_relationship.txt",
#     embedding_path=data_path / "nba_embedding10.npy",
#     feat_drop_rate=0.3,
#     device=device
# )

dataset = PokecZ(
    nodes_path=data_path / "region_job.csv",
    edges_path=data_path / "region_job_relationship.txt",
    embedding_path=data_path / "pokec_z_embedding10.npy",
    feat_drop_rate=0.3,
    device=device
    
)

fair_gnn = FairGNN(
    num_features=dataset.features.shape[1],
).to(device)
fair_gnn.estimator.load_state_dict(torch.load("./src/checkpoint/GCN_sens_region_job__ns_200"))

trainer = FairGNNTrainer(
    dataset=dataset,
    fair_gnn=fair_gnn,
    device=device,
    log_dir=log_dir,
    min_acc=0.65,
    min_roc=0.69,
)

trainer.train(epochs=3000)
