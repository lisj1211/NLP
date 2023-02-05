import json

import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from src.truncator import SampleTruncator
from src.tagging_scheme import HandshakingTaggingEncoder, TagMapping


class MyDataset(Dataset):
    def __init__(self, data_path, pretrained_bert_path, rel_path, max_sequence_length=100, window_size=50):
        self.tokenizer = BertTokenizerFast.from_pretrained(
            pretrained_bert_path, add_special_tokens=False, do_lower_case=True)
        self.truncator = SampleTruncator(max_sequence_length=max_sequence_length, window_size=window_size)
        self.samples = self._read_input_files(data_path)

        with open(rel_path, 'r', encoding='utf-8') as f:
            idx2rel = json.load(f)
        self.tag_mapping = TagMapping(idx2rel)
        self.encoder = HandshakingTaggingEncoder(self.tag_mapping)
        self.max_sequence_length = max_sequence_length

    def _read_input_files(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        all_samples = []
        for sample in tqdm(data, desc='Truncating data'):
            samples = self.truncator.truncate(sample)
            all_samples.extend(samples)
        return all_samples

    def __getitem__(self, index):
        sample = self.samples[index]
        codes = self.tokenizer.encode_plus(
            sample['text'],
            return_offsets_mapping=True,
            add_special_tokens=False,
            max_length=self.max_sequence_length,
            padding='max_length')

        h2t, h2h, t2t = self.encoder.encode(sample, max_sequence_length=self.max_sequence_length)

        input_dic = {
            'sample': json.dumps(sample, ensure_ascii=False),  # raw contents used to compute metrics
            'input_ids': torch.tensor(codes['input_ids']),
            'attention_mask': torch.tensor(codes['attention_mask']),
            'h2t': torch.tensor(h2t, dtype=torch.long),
            'h2h': torch.tensor(h2h, dtype=torch.long),
            't2t': torch.tensor(t2t, dtype=torch.long),
        }
        return input_dic
    
    def __len__(self):
        return len(self.samples)
