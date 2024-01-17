from dataset import NBADataset
from metrics import fair_metric


def pretrain(fair_ac, dataset, num_epochs: int = 200):
    pass


def train(fair_ac, dataset):
    pass


def evaluate(fair_ac, dataset, gnn_factory):
    pass


def main():
    dataset = NBADataset()
    dataset.load_data()
    dataset.preprocess_data()
    dataset.train_model()


if __name__ == "__main__":
    main()
