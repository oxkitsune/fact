class FairAC:
    ae = None
    ac = None
    cs = None

    gnn_factory = None

    def samples(self):
        # bias = training_adj
        # emb_dest = embedding[ac_train_idx]
        # emb_src = embedding[ac_train_idx][feat_keep_idx],
        # feature_src = features_train[feat_keep_idx]

        # return bias, emb_dest, emb_src, feature_src
        pass

    def pretrain(self):
        self.ae.pretrain()
        self.ac.pretrain()

    def train(self):
        self.ae.train()
        self.ac.train()
        self.cs.train()
