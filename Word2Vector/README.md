# word2vector
基于Skip-Gram的word embedding实现，参考了很多博客，训练数据来自THUCNews的一部分数据，共127MB。

## 运行环境
python3.7

## 运行方式
python main.py

## 结果
在3090单卡GPU训练了大概4-5天左右，下图为与“篮球”最相似的10个词，可以看到结果与训练数据有很大关系，总体来看效果还可以。.
![image](https://github.com/lisj1211/NLP/new/main/Word2Vector/篮球.jpg)
