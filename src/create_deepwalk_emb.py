import networkx as nx
import utils_moredatasets

from karateclub import DeepWalk
import time
import datetime
import numpy as np
import random
import torch

# set seeds
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

def create_deepwalk_emb(adj, save_dir, walk_length=100, dimensions=64, epochs=10):
    """
    Function to create a deepwalk embedding which is necessary for the FairAC framework
    
    adj: adjacency matrix of the graph dataset
    save_dir: path to directory where embedding should be saved
    walk_length: length of the random walk
    dimensions: dimension of the embedding
    epochs: number of epochs to train the deepwalk model
    """
    # create a graph from adj
    print("Creating graph from adj...")
    graph = nx.to_networkx_graph(adj)
    
    # create deepwalk embedding
    print("Fitting deepwalk embedding...")
    model = DeepWalk(walk_length=walk_length, dimensions=dimensions, epochs=epochs)
    
    start = time.time()
    model.fit(graph)
    end = time.time()
    
    diff = end-start
    print("Time taken to fit (end-start): ", diff)
    
    print("Getting embedding from fitted DeepWalk...")
    embedding = model.get_embedding()
    print("Embedding\n: ", embedding)

    # write embeddings to npy file
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print("Writing embeddings to file...")
    np.save(f"{save_dir}/deepwalk_emb-{date}-wl={WALK_LENGTH}-dim={DIMENSIONS}-ep={EPOCHS}.npy", emb)
    
    return embedding, diff

print("Loading datasets...")

adj_credit, features, labels, edges, sens, idx_train, idx_val, idx_test, _ = utils_moredatasets.load_credit()

print("Adjacency matrix shape: ", adj_credit.shape)

"""
Looks like:
(18875, 17965)        1.0
  (18875, 18139)        1.0
  (18875, 18510)        1.0
  (18875, 18619)        1.0
  (18875, 18703)        1.0
  (18875, 18842)        1.0
  (18875, 18861)        1.0
  (18875, 18875)        1.0"""

adj_embs = {}
WALK_LENGTH = 100
DIMENSIONS = 64
EPOCHS = 10
emb, diff = create_deepwalk_emb(adj_credit, "./dataset/credit", walk_length=WALK_LENGTH, dimensions=DIMENSIONS, epochs=EPOCHS)