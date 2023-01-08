## Introduction
基于Skip-Gram的word embedding实现，参考了很多博客，训练数据来自THUCNews的一部分数据，共127MB。 

## Requirement
* python 3.7
* torch = 1.12.1
* numpy
* jieba
* pkuseg
* matplotlib

## Train

    python main.py

## Results
在3090单卡GPU训练了大概4-5天左右，下图为与“篮球”最相似的10个词，可以看到结果与训练数据有很大关系，总体来看效果还可以。.
![image](https://github.com/lisj1211/NLP/blob/main/Word2Vector/%E7%AF%AE%E7%90%83.jpg)

## Reference
[1] [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)
