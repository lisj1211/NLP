## Introduction

最近学习中需要使用百度的ERNIE模型，但是ERNIE模型集成于百度PaddlePaddle平台，而我的需求是使用Huggingface的transformers库，因此需要进行跨平台的模型权重转换。虽然目前
transformers已经支持ERNIE(大佬[nghuyong](https://huggingface.co/nghuyong))，但是对于transformers库的版本有要求，低版本的无法使用。因此参考大佬的实现
[nghuyong/ERNIE-Pytorch](https://arxiv.org/abs/1909.07755)，自己进行了相关学习并完成转换过程。

## Requirements

* python 3.7
* torch = 1.12.1
* transformers = 4.23.1
* paddlepaddle = 2.4.0

## Analysis

为了能够进行模型权重的转换，首先得了解PaddlePadlle和Pytorch中模型权重保存形式是什么。  

* 对于PaddlePaddle，查看其模型权重: 
```
>>>import paddle.fluid as fluid
>>>paddle_model_path = r'C:\Users\Administrator\.paddlenlp\models\ernie-3.0-base-zh\ernie_3.0_base_zh.pdparams'
>>>paddle_params = fluid.io.load_program_state(paddle_model_path)
>>>for name, parameters in sorted(paddle_params.items()):
>>>    print(name, parameters.shape)

>>>ernie.embeddings.layer_norm.bias (768,)
>>>ernie.embeddings.layer_norm.weight (768,)
>>>ernie.embeddings.position_embeddings.weight (2048, 768)
>>>ernie.embeddings.task_type_embeddings.weight (3, 768)
>>>ernie.embeddings.token_type_embeddings.weight (4, 768)
>>>ernie.embeddings.word_embeddings.weight (40000, 768)
>>>ernie.encoder.layers.0.linear1.bias (3072,)
>>>ernie.encoder.layers.0.linear1.weight (768, 3072)
>>>ernie.encoder.layers.0.linear2.bias (768,)
>>>ernie.encoder.layers.0.linear2.weight (3072, 768)
>>>ernie.encoder.layers.0.norm1.bias (768,)
>>>ernie.encoder.layers.0.norm1.weight (768,)
>>>ernie.encoder.layers.0.norm2.bias (768,)
>>>ernie.encoder.layers.0.norm2.weight (768,)
>>>ernie.encoder.layers.0.self_attn.k_proj.bias (768,)
>>>ernie.encoder.layers.0.self_attn.k_proj.weight (768, 768)
>>>ernie.encoder.layers.0.self_attn.out_proj.bias (768,)
>>>ernie.encoder.layers.0.self_attn.out_proj.weight (768, 768)
>>>ernie.encoder.layers.0.self_attn.q_proj.bias (768,)
>>>ernie.encoder.layers.0.self_attn.q_proj.weight (768, 768)
>>>ernie.encoder.layers.0.self_attn.v_proj.bias (768,)
>>>ernie.encoder.layers.0.self_attn.v_proj.weight (768, 768)
>>>···
 ```
 
 * 对于pytorch，选择transformers中的bert-base-chinese查看其模型权重: 
```
>>>import torch
>>>pytorch_model_path = r'./bert-base-chinese/pytorch_model.bin'
>>>torch_params = torch.load(pytorch_model_path)
>>>for name, parameters in torch_params.items()::
>>>    print(name, parameters.shape)

>>>bert.embeddings.word_embeddings.weight torch.Size([21128, 768])
>>>bert.embeddings.position_embeddings.weight torch.Size([512, 768])
>>>bert.embeddings.token_type_embeddings.weight torch.Size([2, 768])
>>>bert.embeddings.LayerNorm.gamma torch.Size([768])
>>>bert.embeddings.LayerNorm.beta torch.Size([768])
>>>bert.encoder.layer.0.attention.self.query.weight torch.Size([768, 768])
>>>bert.encoder.layer.0.attention.self.query.bias torch.Size([768])
>>>bert.encoder.layer.0.attention.self.key.weight torch.Size([768, 768])
>>>bert.encoder.layer.0.attention.self.key.bias torch.Size([768])
>>>bert.encoder.layer.0.attention.self.value.weight torch.Size([768, 768])
>>>bert.encoder.layer.0.attention.self.value.bias torch.Size([768])
>>>bert.encoder.layer.0.attention.output.dense.weight torch.Size([768, 768])
>>>bert.encoder.layer.0.attention.output.dense.bias torch.Size([768])
>>>bert.encoder.layer.0.attention.output.LayerNorm.gamma torch.Size([768])
>>>bert.encoder.layer.0.attention.output.LayerNorm.beta torch.Size([768])
>>>bert.encoder.layer.0.intermediate.dense.weight torch.Size([3072, 768])
>>>bert.encoder.layer.0.intermediate.dense.bias torch.Size([3072])
>>>bert.encoder.layer.0.output.dense.weight torch.Size([768, 3072])
>>>bert.encoder.layer.0.output.dense.bias torch.Size([768])
>>>bert.encoder.layer.0.output.LayerNorm.gamma torch.Size([768])
>>>bert.encoder.layer.0.output.LayerNorm.beta torch.Size([768])
>>>···
 ```
通过对比源码发现，两者模型结构基本一致，区别在于权重名称，因此只需要建立一个名称的映射即可。此外，两者对于矩阵乘法的实现存在不同，即XW，WX。
ernie.encoder.layers.0.linear1.weight的shape为(768, 3072)，而同层的bert.encoder.layer.0.intermediate.dense.weight的shape为(3072, 768)，因此需要进行转置操作。
详细过程参见[convert.py](https://github.com/lisj1211/NLP/blob/main/ERNIE_to_Pytorch/convert.py)。由于往往只需要使用bert的前12层，所以我也只转换了前12层。
最后，为了检验正确性，与Huggingface的transformers库输出结果进行对比，见[test.py](https://github.com/lisj1211/NLP/blob/main/ERNIE_to_Pytorch/test.py)。

## Reference
[1] [nghuyong/ERNIE-Pytorch](https://arxiv.org/abs/1909.07755)  
[2] [ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](https://arxiv.org/abs/2107.02137)  
[3] [https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-3.0](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-3.0)
