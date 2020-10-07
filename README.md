# Simplifying Graph Convolutional Networks in PyTorch (TextSGC)

PyTorch 1.6 and Python 3.7 implementation of Simplifying Graph Convolutional Networks [1].

Tested on the 20NG/R8/R52/Ohsumed/MR data set, the code on this repository can achieve the effect of the paper.

## Benchmark

| dataset       | 20NG | R8 | R52 | Ohsumed | MR   |
|---------------|----------|------|--------|--------|--------|
| TextGCN(official) | 0.8634    | 0.9707 | 0.9356   | 0.6836   | 0.7674   |
| This repo.    | 0.8605    | 0.9743 | 0.9384   | 0.6828  | 0.7728  |

NOTE: The result of the experiment is to repeat the run 10 times, and then take the average of accuracy.

## Requirements
* fastai==2.0.15
* PyTorch==1.6.0
* scipy==1.5.2
* pandas==1.0.1
* spacy==2.3.1
* nltk==3.5
* prettytable==1.0.0
* numpy==1.18.5
* networkx==2.5
* tqdm==4.49.0
* scikit_learn==0.23.2

## Usage
1. Process the data first, run `data_processor.py` (Already done)
2. Generate graph, run `build_graph.py` (Already done)
3. Training model, run `trainer.py`

## References
[1] [Wu, F. , Zhang, T. , Souza, A. H. D. , Fifty, C. , Yu, T. , & Weinberger, K. Q. . (2019). Simplifying graph convolutional networks.](https://arxiv.org/abs/1902.07153)
