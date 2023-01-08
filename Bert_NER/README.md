## Introduction

这次实现了BiLSTM-CRF, Bert, Bert-CRF, Bert-LSTM-CRF四种模型的中文命名实体识别(Name Entity Recognition)算法，熟悉了NER任务的基本流程，从数据标注，模型搭建到结果评估。

## Requirements

* python 3.7
* torch = 1.12.1
* numpy
* tqdm
* seqeval
* torchcrf
* transformers

## DataSet
数据集来自[CLUENER2020](https://github.com/CLUEbenchmark/CLUENER2020)。

## Train
训练模型

    python main.py
    
## Results
|Model | Val_precision | Val_recall | Val_f1 | Test_precision | Test_recall | Test_f1 | Time |
|:-----| :----- | :-----| :----- |:----- |:----- |:----- |:-----|
|BiLSTM-CRF | 0.59 | 0.57 | 0.58 | 0.59 | 0.57 | 0.58 | 10m |
|Bert | 0.65 | 0.76 | 0.70 | 0.66 | 0.77 | 0.71 | 10m |
|Bert-CRF | 0.62 | 0.76 | 0.68 | 0.64 | 0.77 | 0.70 | 18m |
|Bert-LSTM-CRF | 0.58 | 0.69 | 0.63 | 0.58 | 0.68 | 0.63 | 19m |


## Analysis

对比于BiLSTM-CRF，Bert基础模型就已经高了很多百分点，说明Bert在语言模型上非常强大。与[hemingkx](https://github.com/hemingkx/CLUENER2020)结果相比，我的结果均低了10个点左右。 
原因可能在于基础模型的选择和实验设置的不同。


## Reference
[1] [hemingkx/CLUENER2020](https://github.com/hemingkx/CLUENER2020)  
[2] [用BERT做NER？教你用PyTorch轻松入门Roberta！](https://zhuanlan.zhihu.com/p/346828049)  
