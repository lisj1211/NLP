# -*- coding:utf-8 -*-
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import matplotlib.pyplot as plt

from src.sampling import create_train_sample, create_eval_sample, create_entity_mask


class Vocabulary:
    """实体和关系词典"""
    def __init__(self, types_path: str):
        types = json.load(open(types_path, 'r', encoding='utf-8'))
        entities = types["entities"]
        relations = types["relations"]
        self.idx2entity = {idx + 1: entity for idx, entity in enumerate(entities)}
        self.entity2idx = {entity: idx + 1 for idx, entity in enumerate(entities)}
        self.idx2entity[0] = "None"
        self.entity2idx["None"] = 0

        self.idx2rel = {idx + 1: rel for idx, rel in enumerate(relations)}
        self.rel2idx = {rel: idx + 1 for idx, rel in enumerate(relations)}
        self.idx2rel[0] = "None"
        self.rel2idx["None"] = 0

    def index_to_entity(self, idx: int):
        return self.idx2entity[idx]

    def entity_to_index(self, entity: str):
        return self.entity2idx[entity]

    def index_to_relation(self, idx: int):
        return self.idx2rel[idx]

    def relation_to_index(self, entity: str):
        return self.rel2idx[entity]

    def len_relations(self):
        return len(self.idx2rel)

    def len_entities(self):
        return len(self.idx2entity)


class MyDataSet(Dataset):
    def __init__(self, config, data_path: str, vocabulary: Vocabulary, is_test: bool):
        self.config = config
        self.entity2idx = vocabulary.entity2idx
        self.rel2idx = vocabulary.rel2idx
        self.is_test = is_test
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)

        with open(data_path, 'r', encoding='utf-8') as f:
            self.dataset = [json.loads(line) for line in f.readlines()]

    def __getitem__(self, idx):
        json_data = self.dataset[idx]
        text = json_data['text']
        spo_list = json_data['spo_list']

        token_list, text_encoding = self.parse_text(text)
        entity_dic, entity_list = self.parse_entities(spo_list, text_encoding)
        rel_list = parse_relations(spo_list)

        if not self.is_test:
            return create_train_sample(token_list, text_encoding, entity_dic, rel_list, self.rel2idx,
                                       self.config.neg_entity_count, self.config.neg_rel_count,
                                       self.config.max_span_size, len(self.rel2idx))
        else:
            return create_eval_sample(token_list, text_encoding, self.config.max_span_size, entity_list, rel_list)

    def parse_text(self, text):
        token_list = []  # text经过bert分词之后得到若干token，每个token在text_encoding中的范围
        text_encoding = [self.tokenizer.convert_tokens_to_ids('[CLS]')]

        text_tokens = self.tokenizer.tokenize(text)
        for i, token_phrase in enumerate(text_tokens):
            token_encoding = self.tokenizer.encode(token_phrase, add_special_tokens=False)
            if not token_encoding:
                token_encoding = [self.tokenizer.convert_tokens_to_ids('[UNK]')]
            span_start, span_end = len(text_encoding), len(text_encoding) + len(token_encoding)
            token_list.append({"span_start": span_start, "span_end": span_end, "phrase": token_phrase})
            text_encoding += token_encoding

        text_encoding += [self.tokenizer.convert_tokens_to_ids('[SEP]')]

        return token_list, text_encoding

    def parse_entities(self, spo_list, text_encoding):
        entity_dic = dict()
        context_size = len(text_encoding)
        entity_list = []
        for spo in spo_list:
            entity_list.append((spo["subject"], spo["subject_type"]))
            entity_list.append((spo["object"], spo["object_type"]))

        for entity, entity_type in entity_list:
            entity_encoding = self.tokenizer.encode(entity, add_special_tokens=False)
            head_idx = find_head_idx(text_encoding, entity_encoding)

            # 特殊字符的存在导致中文分词会有误差， 比如：tokenizer.tokenize("●1995年") 得到 ['●', '##19', '##95', '年']
            # 编码得到[474, 8818, 9102, 2399]，而单独对实体 tokenizer.encode("1995年") 得到 [8396, 2399]，文本中匹配不到，
            # 所以需进行判断
            if head_idx != -1:
                tail_idx = head_idx + len(entity_encoding)
                entity_dic[entity] = {"entity_span": (head_idx, tail_idx),
                                      "entity_type": self.entity2idx[entity_type],
                                      "entity_mask": create_entity_mask(head_idx, tail_idx, context_size),
                                      "entity_size": len(entity)}

        return entity_dic, entity_list

    def __len__(self):
        return len(self.dataset)


def parse_relations(spo_list):
    rel_list = []
    for spo in spo_list:
        rel_list.append((spo["subject"], spo["predicate"], spo["object"]))

    return rel_list


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            padded_batch[key] = padded_stack([s[key] for s in batch])
        else:
            padded_batch[key] = [s[key] for s in batch]

    return padded_batch


def move_dict_value_to_device(*args, device):
    for arg in args:
        for key, value in arg.items():
            if isinstance(value, torch.Tensor):
                arg[key] = value.to(device)


def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor

    return extended_tensor


def padded_stack(tensors, padding=0):
    dim_count = len(tensors[0].shape)

    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for t in tensors:
        e = extend_tensor(t, max_shape, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked


def batch_index(tensor, index, pad=False):
    """
    取出每个关系的两个实体向量
    tensor.shape： [batch_size, entity_size, bert_dim]
    index.shape: [batch_size, rel_size, 2]
    """
    if tensor.shape[0] != index.shape[0]:
        raise Exception()

    if not pad:  # 每个元素 [rel_size, 2, bert_dim]
        return torch.stack([tensor[i][index[i]] for i in range(index.shape[0])])
    else:
        return padded_stack([tensor[i][index[i]] for i in range(index.shape[0])])


def get_optimizer_params(model, weight):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_params = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    return optimizer_params


def save_loss_curve(train_loss):
    plt.figure()
    ax = plt.axes()
    ax.spines['top'].set_visible(False)  # 去除上边框
    ax.spines['right'].set_visible(False)  # 去除右边框

    plt.xlabel('iters')
    plt.ylabel('loss')

    plt.plot(range(len(train_loss)), train_loss, label="train loss")
    plt.legend()
    plt.title('Loss curve')
    plt.savefig('loss.png')
