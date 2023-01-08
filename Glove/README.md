## Introduction
基于global vector的word embedding实现，训练数据来自THUCNews的一小部分数据，随机选择了10000条进行训练。

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
训练设置参考原论文，learning rate设置为0.1，下图为与“篮球”最相似的10个词，总体来看效果还可以，增大预料效果肯定会更好。
![image](https://github.com/lisj1211/NLP/blob/main/Glove/%E7%AF%AE%E7%90%83.png)

## Reference
[1] [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162.pdf)
