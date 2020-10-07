import gc
import warnings
from time import time

import networkx as nx
import numpy as np
import pandas as pd
import torch as th
from sklearn.model_selection import train_test_split

from layer import SGC
from utils import accuracy
from utils import macro_f1
from utils import CudaUse
from utils import EarlyStopping
from utils import LogResult
from utils import parameter_parser
from utils import preprocess_adj
from utils import print_graph_detail
from utils import read_file
from utils import return_seed

th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")


def get_train_test(target_fn):
    train_lst = list()
    test_lst = list()
    with read_file(target_fn, mode="r") as fin:
        for indx, item in enumerate(fin):
            if item.split("\t")[1] in {"train", "training", "20news-bydate-train"}:
                train_lst.append(indx)
            else:
                test_lst.append(indx)

    return train_lst, test_lst


class PrepareData:
    def __init__(self, args):
        print("prepare data")
        self.graph_path = "data/graph"
        self.args = args

        # graph
        graph = nx.read_weighted_edgelist(f"{self.graph_path}/{args.dataset}.txt"
                                          , nodetype=int)
        print_graph_detail(graph)
        adj = nx.to_scipy_sparse_matrix(graph,
                                        nodelist=list(range(graph.number_of_nodes())),
                                        weight='weight',
                                        dtype=np.float)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        self.adj = preprocess_adj(adj, is_sparse=True)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # target

        target_fn = f"data/text_dataset/{self.args.dataset}.txt"
        target = np.array(pd.read_csv(target_fn,
                                      sep="\t",
                                      header=None)[2])
        target2id = {label: indx for indx, label in enumerate(set(target))}
        self.target = [target2id[label] for label in target]
        self.nclass = len(target2id)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # train val test split

        self.train_lst, self.test_lst = get_train_test(target_fn)


class TextSCGTrainer:
    def __init__(self, args, model, pre_data):
        self.args = args
        self.model = model
        self.device = args.device

        self.max_epoch = self.args.max_epoch
        self.set_seed()

        self.dataset = args.dataset
        self.predata = pre_data
        self.earlystopping = EarlyStopping(args.early_stopping)

    def set_seed(self):
        th.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)

    def fit(self):
        self.prepare_data()
        self.convert_tensor()

        self.model = self.model(nfeat=self.nfeat_dim,
                                nclass=self.nclass)
        self.model = self.model.to(self.device)

        self.optimizer = th.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = th.nn.CrossEntropyLoss()

        self.model_param = sum(param.numel() for param in self.model.parameters())
        print('# model parameters:', self.model_param)

        start = time()
        self.train()
        self.train_time = time() - start + self.pre_time

    @classmethod
    def set_description(cls, desc):
        string = ""
        for key, value in desc.items():
            if isinstance(value, int):
                string += f"{key}:{value} "
            else:
                string += f"{key}:{value:.4f} "
        print(string)

    def prepare_data(self):
        self.adj = self.predata.adj
        self.target = self.predata.target
        self.nclass = self.predata.nclass

        self.train_lst, self.val_lst = train_test_split(self.predata.train_lst,
                                                        test_size=self.args.val_ratio,
                                                        shuffle=True,
                                                        random_state=self.args.seed)
        self.test_lst = self.predata.test_lst

    @th.no_grad()
    def sgc_precompute(self, sp_adj, adj_dense, train_lst, val_lst, test_lst):
        start = time()

        # train
        feats = adj_dense[:, train_lst].to(self.device)
        feats = th.spmm(sp_adj, feats).t()

        train_feats_max, _ = feats.max(dim=0, keepdim=True)
        train_feats_min, _ = feats.min(dim=0, keepdim=True)

        train_feats_range = train_feats_max - train_feats_min
        useful_features_dim = train_feats_range.squeeze().gt(0).nonzero().squeeze()
        feats = feats[:, useful_features_dim]
        train_feats_range = train_feats_range[:, useful_features_dim]
        train_feats_min = train_feats_min[:, useful_features_dim]
        train_vec = ((feats - train_feats_min) / train_feats_range)

        # val
        feats = adj_dense[:, val_lst].to(self.device)
        feats = th.spmm(sp_adj, feats).t()
        feats = feats[:, useful_features_dim]
        val_vec = ((feats - train_feats_min) / train_feats_range)

        # test
        feats = adj_dense[:, test_lst].to(self.device)
        feats = th.spmm(sp_adj, feats).t()
        feats = feats[:, useful_features_dim]
        test_vec = ((feats - train_feats_min) / train_feats_range).cpu()

        print(train_vec.size())
        print(val_vec.size())
        print(test_vec.size())
        return train_vec, val_vec, test_vec, time() - start

    def convert_tensor(self):
        self.target = th.tensor(self.target).long().to(self.device)

        self.train_lst = th.tensor(self.train_lst).long().to(self.device)
        self.val_lst = th.tensor(self.val_lst).long().to(self.device)
        self.test_lst = th.tensor(self.test_lst).long()

        adj_dense = self.adj.to_dense()
        self.adj = self.adj.to(self.device)
        self.train_vec, self.val_vec, self.test_vec, self.pre_time = self.sgc_precompute(self.adj,
                                                                                         adj_dense,
                                                                                         self.train_lst,
                                                                                         self.val_lst,
                                                                                         self.test_lst)
        self.nfeat_dim = self.train_vec.size(1)

    def train(self):
        for epoch in range(self.max_epoch):
            self.model.train()
            self.optimizer.zero_grad()

            logits = self.model.forward(self.train_vec)
            loss = self.criterion(logits,
                                  self.target[self.train_lst])

            loss.backward()
            self.optimizer.step()

            val_desc = self.val(self.val_vec, self.val_lst)

            desc = dict(**{"epoch"     : epoch,
                           "train_loss": loss.item(),
                           }, **val_desc)

            self.set_description(desc)

            if self.earlystopping(val_desc["val_loss"]):
                break

    @th.no_grad()
    def val(self, feats, ind, prefix="val"):
        self.model.eval()
        with th.no_grad():
            logits = self.model.forward(feats)
            loss = self.criterion(logits,
                                  self.target[ind])
            acc = accuracy(logits,
                           self.target[ind])
            f1, precision, recall = macro_f1(logits,
                                             self.target[ind],
                                             num_classes=self.nclass)

            desc = {
                f"{prefix}_loss": loss.item(),
                "acc"           : acc,
                "macro_f1"      : f1,
                "precision"     : precision,
                "recall"        : recall,
            }
        return desc

    @th.no_grad()
    def test(self):
        test_vec = self.test_vec.to(self.device)
        test_lst = th.tensor(self.test_lst).long().to(self.device)
        test_desc = self.val(test_vec, test_lst, prefix="test")
        test_desc["train_time"] = self.train_time
        test_desc["model_param"] = self.model_param
        return test_desc


def main(dataset, times):
    args = parameter_parser()
    args.dataset = dataset

    args.device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    args.nhid = 200
    args.max_epoch = 200
    args.dropout = 0.5
    args.val_ratio = 0.1
    args.early_stopping = 10
    args.lr = 0.01
    model = SGC

    print(args)

    predata = PrepareData(args)
    cudause = CudaUse()

    record = LogResult()
    seed_lst = list()
    for ind, seed in enumerate(return_seed(times)):
        print(f"\n\n==> {ind}, seed:{seed}")
        args.seed = seed
        seed_lst.append(seed)

        framework = TextSCGTrainer(model=model, args=args, pre_data=predata)
        framework.fit()

        if th.cuda.is_available():
            gpu_mem = cudause.gpu_mem_get(_id=0)
            record.log_single(key="gpu_mem", value=gpu_mem)

        record.log(framework.test())

        del framework
        gc.collect()

        if th.cuda.is_available():
            th.cuda.empty_cache()

    print("==> seed set:")
    print(seed_lst)
    record.show_str()


if __name__ == '__main__':
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # for d in ["mr", "ohsumed", "R52", "R8", "20ng"]:
    #     main(d)
    main("R8", 1)
    # main("ohsumed")
    # main("R8", 1)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
