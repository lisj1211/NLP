## Introduction

基于Pytorch的GlobalPointer实体识别实现

## Requirements

* python 3.7
* torch = 1.12.1
* numpy
* tqdm
* sklearn
* transformers

## DataSet
数据集为[CLUENER2020](https://github.com/CLUEbenchmark/CLUENER2020)。

## Train
* 数据预处理 
```
    python preprocess.py
```
* 训练模型
```
    python main.py
```
## Results
因为该数据集的测试集没有标注，所以本实验将训练集以8:2进行划分，划分出训练集和验证集，将原dev当做测试集来做。
| Model | Val_precision | Val_recall | Val_f1 | Test_precision | Test_recall | Test_f1 | Time |
|:-----| :----- | :-----| :----- |:----- |:----- |:----- |:-----|
|GlobalPointer | 0.7947 | 0.7588 | 0.7764 | 0.7995 | 0.7735 | 0.7863 | 27m |
|GlobalPointer w/o RoPE | 0.7883 | 0.7642 | 0.7760 | 0.7835 | 0.7634 | 0.7733 | 27m |


## Analysis

GlobalPointer w/o RoPE表示去除苏神提出的[旋转位置编码](https://spaces.ac.cn/archives/8265),可以发现当去除RoPE后，模型的能力并没有得到大的影响，原因应该像苏神说的，
对于长文本RoPE的提升会比较明显，而CLUENER2020训练集文本长度普遍在50以内。对比于之前基于Bert-softmax，Bert-CRF的实现，GlobalPointer提升了7-8个点以上。另外本次实验也学习到了一种新的损失函数[多标签分类](https://spaces.ac.cn/archives/7359)，这也
让我想到了最近在复现TPLinker时，实体抽取能力非常优秀，但是关系抽取一直上不去，loss也不下降，查看TPLinker的issue同样有一部分人遇到了这个问题。在学习到新的损失函数后，之后会
用于测试。GlobalPointer与TPLinker模型思路基本一致，都是以全局指针为思路进行建模。复现之后，收货颇丰，这里强烈建议学习苏神的若干博客。  

为了更好的验证RoPE的作用，以下为几个使用RoPE和不使用RoPE的预测样例：

* 样例一:   
```
    {
        "text": "主场三连胜。波尔图本赛季同样状态糟糕，葡超霸主目前只排名第6位，这是以往难以想象的。双方首回合交手"
        "GlobalPointer": {('organization', '葡超'), ('organization', '波尔图')}
        "GlobalPointer w/o RoPE":{('organization', '。波尔图'), ('organization', '波尔图'), ('organization', '胜。波尔图'), ('organization', '连胜。波尔图'), ('organization', '葡超')}
    }
 ```

  
* 样例二: 
```
    {
        "text": "诺贝尔经济学奖得主迈伦·斯科尔斯日前在北京大学发表演讲时指出，"
        "GlobalPointer": {('organization', '北京大学'), ('position', '得主'), ('name', '迈伦·斯科尔斯')}
        "GlobalPointer w/o RoPE": {('organization', '北京大学'), ('organization', '北京大学发表演讲时指'), ('name', '迈伦·斯科尔斯'), ('organization', '北京大学发表'), ('organization', '北京大学发'), ('organization', '北京大学发表演讲时指出')}
    }
 ```

可以发现，当采用RoPE时，模型可以非常严格的区分出不同实体的边界信息，而不采用RoPE，边界信息会较为混乱，这也提醒了当使用头尾指针时，其中间部分信息也同样重要。
比如CasRel模型将中间部分做平均相加，SpERT采用位置长度编码来处理中间信息。


## Reference
[1] [Global Pointer: Novel Efficient Span-based Approach for Named Entity Recognition](https://arxiv.org/pdf/2208.03054.pdf)  
[2] [https://github.com/xhw205/GlobalPointer_torch](https://github.com/xhw205/GlobalPointer_torch)
