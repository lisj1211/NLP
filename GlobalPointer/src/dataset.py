import json

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


class MyDataset(Dataset):
    def __init__(self, data, pretrained_bert_path, ent2idx_path, max_sequence_length):
        self.tokenizer = BertTokenizerFast.from_pretrained(
            pretrained_bert_path, add_special_tokens=False, do_lower_case=True)
        self.samples = data
        self.ent2idx = json.load(open(ent2idx_path, 'r', encoding='utf-8'))
        self.ent_types = len(self.ent2idx)
        self.max_sequence_length = max_sequence_length

    def __getitem__(self, index):
        sample = self.samples[index]
        codes = self.tokenizer.encode_plus(
            sample['text'],
            return_offsets_mapping=True,
            add_special_tokens=False,
            max_length=self.max_sequence_length,
            padding='max_length')

        input_dic = {
            'input_ids': torch.tensor(codes['input_ids']),
            'attention_mask': torch.tensor(codes['attention_mask']),
            'h2t': self._get_head2tail_matrix(sample['entity_list']),
        }
        return input_dic

    def _get_head2tail_matrix(self, entity_list):
        head2tail = torch.zeros((self.ent_types, self.max_sequence_length, self.max_sequence_length), dtype=torch.long)
        for entity_dic in entity_list:
            ent_type = entity_dic['type']
            token_span_start, token_span_end = entity_dic['token_span']
            head2tail[self.ent2idx[ent_type]][token_span_start][token_span_end] = 1
        return head2tail

    def __len__(self):
        return len(self.samples)
