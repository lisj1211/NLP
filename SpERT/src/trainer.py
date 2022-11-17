# -*- coding:utf-8 -*-
import os
import time
from tqdm import tqdm

import torch
import transformers
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.utils import move_dict_value_to_device, MyDataSet, collate_fn_padding


def train(config, model, optimizer, loss_func, vocabulary):
    print("start training SpERT model")
    time_start = time.time()

    train_dataset = MyDataSet(config=config, data_path=config.train_path, vocabulary=vocabulary, is_test=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, collate_fn=collate_fn_padding,
                              shuffle=True, drop_last=True)

    updates_total = len(train_dataset) // config.batch_size
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=config.lr_warmup * updates_total,
                                                             num_training_steps=updates_total)

    model.zero_grad()

    global_step = 0
    total_loss = 0.
    logg_loss = 0.
    best_f1_score = 0.
    train_loss = []

    for epoch in range(config.max_epoch):
        model.train()
        for batch in tqdm(train_loader, desc=f'Train epoch {epoch}'):
            move_dict_value_to_device(batch, device=config.device)

            global_step += 1
            entity_logits, rel_logits = model(encodings=batch['encodings'],
                                              context_masks=batch['context_masks'],
                                              entity_masks=batch['entity_masks'],
                                              entity_sizes=batch['entity_sizes'],
                                              relations=batch['rels'],
                                              rel_masks=batch['rel_masks'])

            loss = loss_func.compute(entity_logits=entity_logits,
                                     rel_logits=rel_logits,
                                     rel_types=batch['rel_types'],
                                     entity_types=batch['entity_types'],
                                     entity_sample_masks=batch['entity_sample_masks'],
                                     rel_sample_masks=batch['rel_sample_masks'])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if global_step % config.period == 0:
                loss_scalar = (total_loss - logg_loss) / config.period
                logg_loss = total_loss
                print(f'epoch: {epoch}, iter: {global_step}, loss: {loss_scalar:.4f}')
                train_loss.append(loss_scalar)

        # eval validation sets
        entity_result, relation_result = evaluate(config, model, config.dev_path, vocabulary)
        print(f'evaluate entity result: precision: {entity_result[0]:.4f}, recall: {entity_result[1]:.4f}, '
              f'f1: {entity_result[2]:.4f}\n '
              f'evaluate relation result: precision: {relation_result[0]:.4f}, recall: {relation_result[1]:.4f}, '
              f'f1: {relation_result[2]:.4f}')

        if relation_result[2] > best_f1_score:
            best_f1_score = relation_result[2]
            print(f'Saving model, epoch: {epoch}, best f1: {relation_result[2]:.4f}')
            if not os.path.exists(config.save_weights_dir):
                os.makedirs(config.save_weights_dir)
            path = os.path.join(config.save_weights_dir, config.weights_save_name)
            torch.save(model.state_dict(), path)

    time_end = time.time()
    print(f'training SpERT model takes total {int((time_end - time_start) / 60)} m')

    return train_loss


def evaluate(config, model, eval_data_path, vocabulary, rel_filter_threshold=0.4):
    dev_dataset = MyDataSet(config=config, data_path=eval_data_path, vocabulary=vocabulary, is_test=True)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=1, collate_fn=collate_fn_padding, shuffle=False)

    correct_entity_num, predict_entity_num, gold_entity_num = 0, 0, 0
    correct_relation_num, predict_relation_num, gold_relation_num = 0, 0, 0
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)

    model.eval()
    for batch in tqdm(dev_loader, desc="evaluate"):
        with torch.no_grad():
            move_dict_value_to_device(batch, device=config.device)

            entity_clf, rel_clf, rels = model(encodings=batch['encodings'],
                                              context_masks=batch['context_masks'],
                                              entity_masks=batch['entity_masks'],
                                              entity_sizes=batch['entity_sizes'],
                                              entity_spans=batch['entity_spans'],
                                              entity_sample_masks=batch['entity_sample_masks'],
                                              inference=True)

            entity_types = entity_clf.argmax(dim=-1)  # [batch_size, entity_size]
            entity_types *= batch['entity_sample_masks'].long()

            rel_clf[rel_clf < rel_filter_threshold] = 0  # [batch_size, rel_size, rel_class]

            encoding = batch['encodings'][0]  # [sequence_length, ]
            entity_types = entity_types[0]  # [entity_size, ]
            entity_spans = batch['entity_spans'][0]  # [entity_size, 2]
            rel_clf = rel_clf[0]  # [rel_size, rel_class]
            rels = rels[0]  # [rel_size, 2]

            pred_entities = convert_pred_entities(entity_types, entity_spans, vocabulary, encoding, tokenizer)
            pred_relations = convert_pred_relations(entity_spans, rel_clf, rels, vocabulary, encoding, tokenizer)

            gold_entities = set(batch['entity_lists'][0])
            gold_relations = set(batch['rel_lists'][0])

            # 计算实体识别结果
            correct_entity_num += len(pred_entities & gold_entities)
            predict_entity_num += len(pred_entities)
            gold_entity_num += len(gold_entities)

            # 计算关系识别结果
            correct_relation_num += len(pred_relations & gold_relations)
            predict_relation_num += len(pred_relations)
            gold_relation_num += len(gold_relations)

    entity_precision = correct_entity_num / (predict_entity_num + 1e-10)
    entity_recall = correct_entity_num / (gold_entity_num + 1e-10)
    entity_f1_score = 2 * entity_precision * entity_recall / (entity_precision + entity_recall + 1e-10)

    relation_precision = correct_relation_num / (predict_relation_num + 1e-10)
    relation_recall = correct_relation_num / (gold_relation_num + 1e-10)
    relation_f1_score = 2 * relation_precision * relation_recall / (relation_precision + relation_recall + 1e-10)

    return (entity_precision, entity_recall, entity_f1_score), (relation_precision, relation_recall, relation_f1_score)


def convert_pred_entities(entity_types, entity_spans, vocabulary, encoding, tokenizer):
    valid_entity_indices = entity_types.nonzero().view(-1)  # 非0元素索引，即非None的实体类别对应的索引
    pred_entity_types = entity_types[valid_entity_indices]  # 根据索引拿到类别
    pred_entity_spans = entity_spans[valid_entity_indices]  # 根据索引拿到范围

    # convert to tuples (entity, entity type)
    converted_preds = set()
    for i in range(pred_entity_types.shape[0]):
        label_idx = pred_entity_types[i].item()
        entity_type = vocabulary.index_to_entity(label_idx)

        start, end = pred_entity_spans[i].tolist()
        entity = ''.join(tokenizer.decode(encoding[start: end]).split())

        converted_preds.add((entity, entity_type))

    return converted_preds


def convert_pred_relations(entity_spans, rel_clf, rels, vocabulary, encoding, tokenizer):
    rel_class_count = rel_clf.shape[1]
    rel_clf = rel_clf.view(-1)

    rel_nonzero = rel_clf.nonzero().view(-1)  # 非0元素索引

    pred_rel_types = (rel_nonzero % rel_class_count) + 1
    # 取余得到所有关系的预测类别，因为构建词典包括None关系，所以+1
    valid_rel_indices = rel_nonzero // rel_class_count
    # 如valid_rel_indices： tensor([0, 1, 1, 2, 3])表示第2，3个预测关系是由index为1的实体对预测出来

    valid_rels = rels[valid_rel_indices]  # 得到有效的关系对 shape: [valid_rel_nums, 2], 2表示头尾实体的index

    pred_rel_entity_spans = entity_spans[valid_rels].long()
    # 根据头尾实体的index，得到头尾实体的span范围，[valid_rel_nums, 2, 2], 第二维的2表示头尾实体，第三维表示对应的span

    # convert to tuples (subject, rel_type, object)
    converted_rels = set()

    for i in range(pred_rel_types.shape[0]):
        label_idx = pred_rel_types[i].item()
        pred_rel_type = vocabulary.index_to_relation(label_idx)

        spans = pred_rel_entity_spans[i]
        head_start, head_end = spans[0].tolist()
        tail_start, tail_end = spans[1].tolist()

        head_entity = ''.join(tokenizer.decode(encoding[head_start: head_end]).split())
        tail_entity = ''.join(tokenizer.decode(encoding[tail_start: tail_end]).split())

        converted_rels.add((head_entity, pred_rel_type, tail_entity))

    return converted_rels
