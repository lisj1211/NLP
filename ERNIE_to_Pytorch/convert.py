# -*- coding: utf-8 -*-
import os
import json
from collections import OrderedDict

import paddle.fluid as fluid
import torch


def build_params_map(attention_num):
    """
    build params map from paddle-paddle's ERNIE to transformer's BERT
    :param attention_num: attention block nums
    :return:
    """
    # embeddings block map
    weight_map = OrderedDict({
        'ernie.embeddings.word_embeddings.weight': "bert.embeddings.word_embeddings.weight",
        'ernie.embeddings.position_embeddings.weight': "bert.embeddings.position_embeddings.weight",
        'ernie.embeddings.token_type_embeddings.weight': "bert.embeddings.token_type_embeddings.weight",
        'ernie.embeddings.task_type_embeddings.weight': "bert.embeddings.task_type_embeddings.weight",
        'ernie.embeddings.layer_norm.weight': 'bert.embeddings.LayerNorm.gamma',
        'ernie.embeddings.layer_norm.bias': 'bert.embeddings.LayerNorm.beta',
    })
    # attention block map
    for i in range(attention_num):
        weight_map[f'ernie.encoder.layers.{i}.self_attn.q_proj.weight'] = f'bert.encoder.layer.{i}.attention.self.query.weight'
        weight_map[f'ernie.encoder.layers.{i}.self_attn.q_proj.bias'] = f'bert.encoder.layer.{i}.attention.self.query.bias'
        weight_map[f'ernie.encoder.layers.{i}.self_attn.k_proj.weight'] = f'bert.encoder.layer.{i}.attention.self.key.weight'
        weight_map[f'ernie.encoder.layers.{i}.self_attn.k_proj.bias'] = f'bert.encoder.layer.{i}.attention.self.key.bias'
        weight_map[f'ernie.encoder.layers.{i}.self_attn.v_proj.weight'] = f'bert.encoder.layer.{i}.attention.self.value.weight'
        weight_map[f'ernie.encoder.layers.{i}.self_attn.v_proj.bias'] = f'bert.encoder.layer.{i}.attention.self.value.bias'
        weight_map[f'ernie.encoder.layers.{i}.self_attn.out_proj.weight'] = f'bert.encoder.layer.{i}.attention.output.dense.weight'
        weight_map[f'ernie.encoder.layers.{i}.self_attn.out_proj.bias'] = f'bert.encoder.layer.{i}.attention.output.dense.bias'
        weight_map[f'ernie.encoder.layers.{i}.norm1.weight'] = f'bert.encoder.layer.{i}.attention.output.LayerNorm.gamma'
        weight_map[f'ernie.encoder.layers.{i}.norm1.bias'] = f'bert.encoder.layer.{i}.attention.output.LayerNorm.beta'
        weight_map[f'ernie.encoder.layers.{i}.linear1.weight'] = f'bert.encoder.layer.{i}.intermediate.dense.weight'
        weight_map[f'ernie.encoder.layers.{i}.linear1.bias'] = f'bert.encoder.layer.{i}.intermediate.dense.bias'
        weight_map[f'ernie.encoder.layers.{i}.linear2.weight'] = f'bert.encoder.layer.{i}.output.dense.weight'
        weight_map[f'ernie.encoder.layers.{i}.linear2.bias'] = f'bert.encoder.layer.{i}.output.dense.bias'
        weight_map[f'ernie.encoder.layers.{i}.norm2.weight'] = f'bert.encoder.layer.{i}.output.LayerNorm.gamma'
        weight_map[f'ernie.encoder.layers.{i}.norm2.bias'] = f'bert.encoder.layer.{i}.output.LayerNorm.beta'

    return weight_map


def extract_and_convert(input_dir, output_dir):
    """
    :param input_dir: PaddlePaddle ERNIE dir
    :param output_dir: output dir
    :return:
    """
    os.makedirs(output_dir, exist_ok=True)
    config = json.load(open(os.path.join(input_dir, 'model_config.json'), 'r', encoding='utf-8'))
    if 'init_args' in config:
        config = config['init_args'][0]
    del config['init_class']
    config['layer_norm_eps'] = 1e-12
    config['model_type'] = 'bert'
    config['architectures'] = ["BertForMaskedLM"]
    config['intermediate_size'] = 4 * config['hidden_size']
    json.dump(config, open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8'), indent=4)

    with open(os.path.join(input_dir, 'vocab.txt'), 'rt', encoding='utf-8') as f:
        words = f.read().splitlines()
    words = [word.split('\t')[0] for word in words]
    with open(os.path.join(output_dir, 'vocab.txt'), 'wt', encoding='utf-8') as f:
        for word in words:
            f.write(word + "\n")

    state_dict = OrderedDict()
    weight_map = build_params_map(attention_num=config['num_hidden_layers'])
    paddle_paddle_params = fluid.io.load_program_state(os.path.join(input_dir, 'ernie_3.0_base_zh.pdparams'))

    for ernie_weight_name in weight_map:
        weight_value = paddle_paddle_params[ernie_weight_name]

        if 'weight' in ernie_weight_name and 'ernie.encoder' in ernie_weight_name:
            weight_value = weight_value.transpose()

        state_dict[weight_map[ernie_weight_name]] = torch.FloatTensor(weight_value)

    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))


if __name__ == '__main__':
    extract_and_convert(r'C:\Users\Administrator\.paddlenlp\models\ernie-3.0-base-zh', './convert/')
