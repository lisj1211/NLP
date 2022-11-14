## Introduction

选择4种经典的文本分类方法进行对比，包括LSTM，Bert，TextCNN，FastText。都是非常经典的模型，当然还有加入Attention等之后再做尝试

## Requirements

* python 3.7
* torch = 1.12.1
* numpy
* matplotlib
* sklearn
* transformers
* pkuseg

## Train
训练LSTM，TextCNN，FastText模型

    python main.py

训练Bert模型

    python bert_main.py
    
## Results
BertFreeze表示冻结Bert预训练参数，只训练分类头。BertUnfreeze表示所有参数均进行训练。

Model | Train_acc | Train_loss | Val_acc | Val_loss | Test_acc | Test_loss | time
--- | --- | --- | --- |--- |--- |--- |--- 
TextCNN | 0.9923 | 0.1131 | 0.9544 | 0.1508 | 0.969 | 0.0986 | 9m 
FastText | 0.9414 | 0.7025 | 0.9166 | 0.3223 | 0.9322 | 0.2864 | 3m 
LSTM | 0.9733 | 0.8086 | 0.9318 | 0.2561 | 0.9581 | 0.1472 | 92m 
BertFreeze | 0.9465 | 0.2647 | 0.9308 | 0.2245 | 0.9404 | 0.1865 | 189m 
BertUnfreeze | 0.9977 | 0.0252 | 0.9788 | 0.1164 | 0.9843 | 0.0594 | 369m 

训练loss曲线
![loss_curve](https://github.com/lisj1211/NLP/blob/main/TextClassification/loss_curve.png)

## Analysis

由图表可以看出BertUnfreeze模型所有指标都是最优的，但同时因为参数量大，导致训练时间较长。FastText由于模型简单，所以训练速度快，但指标稍有不足。
综合来看，TextCNN在指标上和训练时间上都比较优秀。

## TODO
* 在多个数据集上进行测试
* LSTM加入Attention机制等

## Reference
[1] [A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1510.03820.pdf)  
[2] [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)  
[3] [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)  
[4] [https://github.com/BeHappyForMe/Multi_Model_Classification](https://github.com/BeHappyForMe/Multi_Model_Classification)
