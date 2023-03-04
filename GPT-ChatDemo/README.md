## Introduction

基于Pytorch的TPLinker关系抽取实现

## Requirements

* python 3.7
* torch = 1.12.1
* numpy
* tqdm
* transformers
* termcolor

## DataSet

数据集为50w中文闲聊数据集地址. 原始数据集为`./data/train.txt`

## Train

* 训练模型

```
    python train.py
```

* 测试模型

```
    python chat_demo.py
```

## Results

下图为训练50个epoch之后的对话结果 
![demo](https://github.com/lisj1211/NLP/blob/main/GPT-ChatDemo/demo.png) 

## Analysis

可以发现，人与模型可以进行简单的交互。最终效果跟训练预料和迭代次数有较大关系。因为时间关系，只训练了50个epoch，在3090单卡耗费大概2天的时间，模型仍未收敛。

## Reference

[1] [how-to-generate](https://huggingface.co/blog/how-to-generate)  
