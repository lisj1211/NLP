## Introduction

基于Pytorch的CasRel关系抽取模型实现，部分参考自[https://github.com/Onion12138/CasRelPyTorch](https://github.com/Onion12138/CasRelPyTorch), 该实现基于FastNLP，为了更好学习，
便修改为完全基于Pytorch。这里附上[官方源码](https://github.com/weizhepei/CasRel)Keras版本。

## Requirements

* python 3.7
* torch = 1.12.1
* numpy
* matplotlib
* transformers

## DataSet
数据集为百度关系抽取数据集，句子最大长度为294，最小长度为5，平均长度为50，因此padding长度选择为300。

## Train
训练模型

    python main.py
    
## Results

|Model | Val_precision | Val_recall | Val_f1 | Test_precision | Test_recall | Test_f1 | Time |
|:-----| :----- | :-----| :----- |:----- |:----- |:----- |:-----|
|CasRel | 0.73 | 0.70 | 0.76 | 0.75 | 0.79 | 0.71 | 99m |

训练loss曲线  
![loss_curve](https://github.com/lisj1211/NLP/blob/main/CasRel/picture/loss1.png)  
开始loss为16是因为训练过程采用warm up，去掉第一次更新的loss后，结果为下图所示  
![loss_curve](https://github.com/lisj1211/NLP/blob/main/CasRel/picture/loss.png)
## Analysis

训练过程更新迭代200次loss就比较稳定了，可见Bert模型的强大。与[博客](https://github.com/Onion12138/CasRelPyTorch)结果相比，作者的测试集结果为F1 0.78，
precision 0.80，recall 0.76。稍微有点差距，应该是优化过程的不同。

## ToDo

之后会抽空学习SpERT，与该模型进行比较。

## Reference
[1] [A Novel Cascade Binary Tagging Framework for Relational Triple Extraction](https://arxiv.org/abs/1909.03227)  
[2] [https://github.com/Onion12138/CasRelPyTorch](https://github.com/Onion12138/CasRelPyTorch)
