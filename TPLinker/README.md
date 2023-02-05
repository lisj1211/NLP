## Introduction

基于Pytorch的TPLinker关系抽取实现

## Requirements

* python 3.7
* torch = 1.12.1
* numpy
* tqdm
* transformers

## DataSet

数据集为百度关系抽取数据集. 原始数据集在`./data/baidu/raw`文件夹下

## Train

* 数据预处理, bert预训练模型需要放在`bert-base-chinese`文件夹下

```
    cd ./src
    python preprocess.py
```

* 训练模型

```
    python main.py
```

## Results

无

## Analysis

在训练TPLinker模型时, 首先基于原作者的设置采用`cross-entropy`进行训练, `loss`形式采取`head2tail`, `head2head`,
`tail2tail`三个`loss`直接相加的方式. 最终结果对于实体识别, `acc`在`99%`以上, 但是对于关系抽取, 所有指标均为`0`.
经过调试发现对于关系抽取的`head2head`, `tail2tail`得分矩阵, 模型将所有`token-pair`均判为负类, 即没有任何关系.
通过查看官方`issue`,
发现有很多人都训练失败, 作者给出的方案是多训练几个epoch, 测试后仍然训练不起来. 后发现作者对于`head2tail`, `head2head`,
`tail2tail`三个`loss`进行加权, 即训练前期`head2tail`的权重大, 后期`head2head`, `tail2tail`的权重大,
个人认为前期网络的主要任务在实体的识别上,
当实体识别取得一定效果后, 增大关系抽取任务的权重, 经过实验测试后发现仍然训练不起来. 之后考虑到类别稀疏问题,
检查发现在一个batch中, 实体任务的样本数为`5050`, 但其中大部分只存在`10`个一下的正样本, 其余均为负样本,
关系任务的样本数为`num_relations * 5050`, 即`18 * 5050`, 而正样本通常只有`2-3`个, 关系任务的样本数严重不均衡,
负样本`loss`主导了梯度的主要方向, 因此将所有样本均判别为负样本, 导致模型训练失败. 考虑到样本不均衡问题, 进而采用`focal loss`来尝试解决正负样本不均衡问题, 此部分可查看`loss.py`, 试验后发现依然训练失败. 目前还没有找到有效的解决方案.   

虽然对于TPLinker模型的复现未能成功, 但复现过程中仍然取得了很多收货, 是一个非常好的工作. 训练失败问题今后需要在看看其他解决方法.

## Reference

[1] [TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking](https://arxiv.org/pdf/2010.13415.pdf)  
[2] [TPLinker官方源码](https://github.com/131250208/TPlinker-joint-extraction)
