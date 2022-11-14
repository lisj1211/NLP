import time
import os
from tqdm import tqdm

import torch
import transformers
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from .utils import MyDataSet, move_dict_value_to_device, to_tuple, my_collate_fn


def train(config, model, loss_function, optimizer, rel_vocab):
    print("start training CasRel model")
    time_start = time.time()

    train_dataset = MyDataSet(config.train_path, config.bert_path, rel_vocab, config.max_len, is_test=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, collate_fn=my_collate_fn,
                              shuffle=True, drop_last=True)

    updates_total = len(train_dataset) // config.batch_size
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=config.lr_warmup * updates_total,
                                                             num_training_steps=updates_total)

    global_step = 0
    total_loss = 0.
    logg_loss = 0.
    train_loss = []
    best_f1_score = 0.

    for epoch in range(config.max_epoch):
        model.train()
        for batch_x, batch_y in tqdm(train_loader, desc=f"train epoch {epoch}"):
            move_dict_value_to_device(batch_x, batch_y, device=config.device)

            global_step += 1
            pred_dic = model(batch_x['token_ids'], batch_x['mask'], batch_x['sub_head'], batch_x['sub_tail'])

            loss = loss_function(pred_dic, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if global_step % config.period == 0:
                loss_scalar = (total_loss - logg_loss) / config.period
                logg_loss = total_loss
                print(f'epoch: {epoch}, iter: {global_step}, loss: {loss_scalar:.4f}')
                train_loss.append(loss_scalar)

        precision, recall, f1_score = evaluate(config, model, config.dev_path, rel_vocab)
        print(f'epoch {epoch}, evaluate result: f1: {f1_score:.2f}, precision: {precision:.2f}, recall: {recall:.2f}')

        if f1_score > best_f1_score:
            best_f1_score = f1_score
            print(f'Saving model, best f1: {f1_score:.2f}, precision: {precision:.2f}, recall: {recall:.2f}')
            if not os.path.exists(config.save_weights_dir):
                os.makedirs(config.save_weights_dir)
            path = os.path.join(config.save_weights_dir, config.weights_save_name)
            torch.save(model.state_dict(), path)

    time_end = time.time()
    print(f'training CasRel model takes total {int((time_end-time_start)/60)} m')

    return train_loss


def evaluate(config, model, eval_data_path, rel_vocab, h_bar=0.5, t_bar=0.5):
    eval_dataset = MyDataSet(eval_data_path, config.bert_path, rel_vocab, config.max_len, is_test=True)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=1, collate_fn=my_collate_fn, shuffle=False)

    correct_num, predict_num, gold_num = 0, 0, 0
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)

    model.eval()
    for batch_x, batch_y in tqdm(eval_loader, desc="evaluate"):
        with torch.no_grad():
            move_dict_value_to_device(batch_x, batch_y, device=config.device)
            token_ids = batch_x['token_ids']
            mask = batch_x['mask']

            encoded_text = model.get_encoded_text(token_ids, mask)
            pred_sub_heads, pred_sub_tails = model.get_subs(encoded_text)
            sub_heads = torch.where(pred_sub_heads[0] > h_bar)[0]
            sub_tails = torch.where(pred_sub_tails[0] > t_bar)[0]

            subjects = []
            for sub_head in sub_heads:  # 所有预测得到的实体
                sub_tail = sub_tails[sub_tails >= sub_head]
                if len(sub_tail) > 0:
                    sub_tail = sub_tail[0]
                    subject = ''.join(tokenizer.decode(token_ids[0][sub_head: sub_tail + 1]).split())
                    subjects.append((subject, sub_head, sub_tail))

            if subjects:  # 如果存在实体
                triple_list = []
                repeated_encoded_text = encoded_text.repeat(len(subjects), 1, 1)  # 对每一个实体预测其关系
                sub_head_mapping = torch.zeros((len(subjects), 1, encoded_text.size(1)), dtype=torch.float,
                                               device=config.device)
                sub_tail_mapping = torch.zeros((len(subjects), 1, encoded_text.size(1)), dtype=torch.float,
                                               device=config.device)
                for subject_idx, subject in enumerate(subjects):
                    sub_head_mapping[subject_idx][0][subject[1]] = 1
                    sub_tail_mapping[subject_idx][0][subject[2]] = 1
                pred_obj_heads, pred_obj_tails = model.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping,
                                                                                 repeated_encoded_text)
                for subject_idx, subject in enumerate(subjects):
                    sub = subject[0]
                    obj_heads = torch.where(pred_obj_heads[subject_idx] > h_bar)
                    obj_tails = torch.where(pred_obj_tails[subject_idx] > t_bar)
                    for obj_head, rel_head in zip(*obj_heads):  # 头坐标
                        for obj_tail, rel_tail in zip(*obj_tails):  # 尾坐标
                            if obj_head <= obj_tail and rel_head == rel_tail:
                                rel = rel_vocab.to_rel(int(rel_head))
                                obj = ''.join(tokenizer.decode(token_ids[0][obj_head: obj_tail + 1]).split())
                                triple_list.append((sub, rel, obj))
                                break

                triple_set = set()
                for s, r, o in triple_list:
                    triple_set.add((s, r, o))
                pred_list = list(triple_set)

            else:
                pred_list = []

            pred_triples = set(pred_list)
            gold_triples = set(to_tuple(batch_y['triples'][0]))
            correct_num += len(pred_triples & gold_triples)
            predict_num += len(pred_triples)
            gold_num += len(gold_triples)

    precision = correct_num / (predict_num + 1e-10)
    recall = correct_num / (gold_num + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)

    return precision, recall, f1_score
