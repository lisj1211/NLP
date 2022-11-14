import json
from collections import defaultdict
from random import choice

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
import matplotlib.pyplot as plt


class RelVocab:
    def __init__(self, rel_path: str):
        rel_list = json.load(open(rel_path, 'r', encoding='utf-8')).values()
        self.idx2rel = {idx: rel for idx, rel in enumerate(rel_list)}
        self.rel2idx = {rel: idx for idx, rel in enumerate(rel_list)}

    def to_index(self, rel: str):
        return self.rel2idx[rel]

    def to_rel(self, idx: int):
        return self.idx2rel[idx]

    def __len__(self):
        return len(self.idx2rel)


class MyDataSet(Dataset):
    def __init__(self, data_path: str, bert_path: str, rel_vocabulary: RelVocab, max_len: int, is_test: bool):
        self.max_len = max_len
        self.rel_vocab = rel_vocabulary
        self.is_test = is_test
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        with open(data_path, 'r', encoding='utf-8') as f:
            self.dataset = [json.loads(line) for line in f.readlines()]

    def __getitem__(self, idx):
        json_data = self.dataset[idx]
        text = json_data['text']
        tokenized = self.tokenizer(text, max_length=self.max_len, truncation=True)
        tokens = tokenized['input_ids']
        masks = tokenized['attention_mask']
        text_len = len(tokens)

        token_ids = torch.tensor(tokens, dtype=torch.long)
        masks = torch.tensor(masks, dtype=torch.bool)
        sub_heads, sub_tails = torch.zeros(text_len), torch.zeros(text_len)
        sub_head, sub_tail = torch.zeros(text_len), torch.zeros(text_len)
        obj_heads = torch.zeros((text_len, len(self.rel_vocab)))
        obj_tails = torch.zeros((text_len, len(self.rel_vocab)))

        if not self.is_test:
            s2ro_map = defaultdict(list)
            for spo in json_data['spo_list']:
                triple = (self.tokenizer(spo['subject'], add_special_tokens=False)['input_ids'],
                          self.rel_vocab.to_index(spo['predicate']),
                          self.tokenizer(spo['object'], add_special_tokens=False)['input_ids'])
                sub_head_idx = find_head_idx(tokens, triple[0])
                obj_head_idx = find_head_idx(tokens, triple[2])
                if sub_head_idx != -1 and obj_head_idx != -1:
                    sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                    s2ro_map[sub].append(
                        (obj_head_idx, obj_head_idx + len(triple[2]) - 1, triple[1]))

            if s2ro_map:
                for s in s2ro_map:
                    sub_heads[s[0]] = 1
                    sub_tails[s[1]] = 1
                sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))
                sub_head[sub_head_idx] = 1
                sub_tail[sub_tail_idx] = 1
                for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
                    obj_heads[ro[0]][ro[2]] = 1
                    obj_tails[ro[1]][ro[2]] = 1

        return token_ids, masks, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, json_data['spo_list']

    def __len__(self):
        return len(self.dataset)


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


def my_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    token_ids, masks, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, triples = zip(*batch)
    batch_token_ids = pad_sequence(token_ids, batch_first=True)
    batch_masks = pad_sequence(masks, batch_first=True)
    batch_sub_heads = pad_sequence(sub_heads, batch_first=True)
    batch_sub_tails = pad_sequence(sub_tails, batch_first=True)
    batch_sub_head = pad_sequence(sub_head, batch_first=True)
    batch_sub_tail = pad_sequence(sub_tail, batch_first=True)
    batch_obj_heads = pad_sequence(obj_heads, batch_first=True)
    batch_obj_tails = pad_sequence(obj_tails, batch_first=True)

    return {"token_ids": batch_token_ids,
            "mask": batch_masks,
            "sub_head": batch_sub_head,
            "sub_tail": batch_sub_tail,
            "sub_heads": batch_sub_heads,
            }, \
           {"mask": batch_masks,
            "sub_heads": batch_sub_heads,
            "sub_tails": batch_sub_tails,
            "obj_heads": batch_obj_heads,
            "obj_tails": batch_obj_tails,
            "triples": triples
            }


def move_dict_value_to_device(*args, device):
    for arg in args:
        for key, value in arg.items():
            if isinstance(value, torch.Tensor):
                arg[key] = value.to(device)


def to_tuple(triple_list):
    ret = []
    for triple in triple_list:
        ret.append((triple['subject'], triple['predicate'], triple['object']))
    return ret


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

