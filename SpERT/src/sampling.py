# -*- coding:utf-8 -*-
import random
from typing import List

import torch


def create_train_sample(token_list: List[dict], text_encoding: List[int], entity_dic: dict, rel_list: List[tuple],
                        rel2idx: dict, neg_entity_count: int, neg_rel_count: int,  max_span_size: int,
                        rel_type_count: int):

    token_count = len(token_list)
    context_size = len(text_encoding)

    # ==========生成实体的正样本==========
    pos_entity_spans, pos_entity_types, pos_entity_masks, pos_entity_sizes = [], [], [], []
    for e in entity_dic:
        pos_entity_spans.append(entity_dic[e]["entity_span"])
        pos_entity_types.append(entity_dic[e]["entity_type"])
        pos_entity_masks.append(entity_dic[e]["entity_mask"])
        pos_entity_sizes.append(entity_dic[e]["entity_size"])

    # ==========生成关系字典==========
    entity_pair_relations = dict()  # {(sub, obj): [rel1, rel2], ``` , (sub, obj): [rel]}
    for rel in rel_list:
        pair = (rel[0], rel[2])
        if pair not in entity_pair_relations:
            entity_pair_relations[pair] = []
        entity_pair_relations[pair].append(rel[1])

    # ==========生成关系正样本==========
    pos_rels, pos_rel_spans, pos_rel_types, pos_rel_masks = [], [], [], []
    for pair, rels in entity_pair_relations.items():
        head_entity, tail_entity = pair
        if head_entity in entity_dic and tail_entity in entity_dic:  # 只有当两个实体均被预处理出来
            s1, s2 = entity_dic[head_entity]["entity_span"], entity_dic[tail_entity]["entity_span"]
            pos_rels.append((pos_entity_spans.index(s1), pos_entity_spans.index(s2)))  # 在实体列表中的索引
            pos_rel_spans.append((s1, s2))

            pair_rel_types = [rel2idx[r] for r in rels]
            pair_rel_types = [int(t in pair_rel_types) for t in range(1, rel_type_count)]  # one-hot向量
            pos_rel_types.append(pair_rel_types)
            pos_rel_masks.append(create_rel_mask(s1, s2, context_size))

    # ==========生成实体负样本集合==========
    neg_entity_spans, neg_entity_sizes = [], []
    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = get_neg_entity_span(i, i + size - 1, token_list)  # 得到特定大小负样本在text_encoding中的范围
            if span not in pos_entity_spans:
                neg_entity_spans.append(span)
                neg_entity_sizes.append(size)

    # ==========对实体负样本采样==========
    neg_entity_samples = random.sample(list(zip(neg_entity_spans, neg_entity_sizes)),
                                       min(len(neg_entity_spans), neg_entity_count))
    neg_entity_spans, neg_entity_sizes = zip(*neg_entity_samples) if neg_entity_samples else ([], [])

    neg_entity_masks = [create_entity_mask(*span, context_size) for span in neg_entity_spans]
    neg_entity_types = [0] * len(neg_entity_spans)

    # ==========生成关系负样本集合==========
    # 构建负样本关系时，只使用正实体样本，并且两正实体之间不存在关系，正关系的两实体调换位置也算负样本
    neg_rel_spans = []
    for i1, s1 in enumerate(pos_entity_spans):
        for i2, s2 in enumerate(pos_entity_spans):
            # 自身与自身不构成负样本
            if s1 != s2 and (s1, s2) not in pos_rel_spans:
                neg_rel_spans.append((s1, s2))

    # ==========对关系负样本采样==========
    neg_rel_spans = random.sample(neg_rel_spans, min(len(neg_rel_spans), neg_rel_count))

    neg_rels = [(pos_entity_spans.index(s1), pos_entity_spans.index(s2)) for s1, s2 in neg_rel_spans]
    neg_rel_masks = [create_rel_mask(s1, s2, context_size) for s1, s2 in neg_rel_spans]
    neg_rel_types = [[0] * (rel_type_count - 1)] * len(neg_rel_spans)  # -1因为最后算BCEloss

    # ==========合并正负样本==========
    entity_types = pos_entity_types + neg_entity_types
    entity_masks = pos_entity_masks + neg_entity_masks
    entity_sizes = pos_entity_sizes + list(neg_entity_sizes)

    rels = pos_rels + neg_rels
    rel_types = pos_rel_types + neg_rel_types
    rel_masks = pos_rel_masks + neg_rel_masks

    assert len(entity_masks) == len(entity_sizes) == len(entity_types)
    assert len(rels) == len(rel_masks) == len(rel_types)

    # ==========创建tensor==========
    encodings = torch.tensor(text_encoding, dtype=torch.long)
    context_masks = torch.ones(context_size, dtype=torch.bool)

    if entity_masks:
        entity_types = torch.tensor(entity_types, dtype=torch.long)
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool)
        # 当前样本构建出的所有正负实体mask, 值均为True,参与loss计算
    else:
        # 处理没有正负样本的边角问题
        entity_types = torch.zeros([1], dtype=torch.long)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    if rels:
        rels = torch.tensor(rels, dtype=torch.long)
        rel_masks = torch.stack(rel_masks)
        rel_types = torch.tensor(rel_types, dtype=torch.float32)
        rel_sample_masks = torch.ones([rels.shape[0]], dtype=torch.bool)
    else:
        # 处理没有正负样本的边角问题
        rels = torch.zeros([1, 2], dtype=torch.long)
        rel_types = torch.zeros([1, rel_type_count-1], dtype=torch.float32)
        rel_masks = torch.zeros([1, context_size], dtype=torch.bool)
        rel_sample_masks = torch.zeros([1], dtype=torch.bool)

    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_sizes=entity_sizes, entity_types=entity_types,
                rels=rels, rel_masks=rel_masks, rel_types=rel_types,
                entity_sample_masks=entity_sample_masks, rel_sample_masks=rel_sample_masks)


def create_eval_sample(token_list: List[dict], text_encoding: List[int], max_span_size: int, entity_list: List[tuple],
                       rel_list: List[tuple]):
    token_count = len(token_list)
    context_size = len(text_encoding)

    # ==========创建候选实体==========
    entity_spans = []
    entity_masks = []
    entity_sizes = []

    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = get_neg_entity_span(i, i + size - 1, token_list)  # 得到特定大小负样本在text_encoding中的范围
            entity_spans.append(span)
            entity_masks.append(create_entity_mask(*span, context_size))
            entity_sizes.append(size)

    encodings = torch.tensor(text_encoding, dtype=torch.long)
    context_masks = torch.ones(context_size, dtype=torch.bool)

    if entity_masks:
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_spans = torch.tensor(entity_spans, dtype=torch.long)
        entity_sample_masks = torch.tensor([1] * entity_masks.shape[0], dtype=torch.bool)
    else:
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_spans = torch.zeros([1, 2], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_sizes=entity_sizes, entity_spans=entity_spans, entity_sample_masks=entity_sample_masks,
                entity_lists=entity_list, rel_lists=rel_list)


def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask


def create_rel_mask(s1, s2, context_size):
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_entity_mask(start, end, context_size)
    return mask


def get_neg_entity_span(start, end, token_list):
    if start == end:  # size为1
        return token_list[start]["span_start"], token_list[start]["span_end"]

    return token_list[start]["span_start"], token_list[end]["span_end"]
