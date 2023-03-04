import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm

from src.logger import setup_logger

logger = setup_logger(__name__)


class MyDataset(Dataset):
    def __init__(self, data_path, vocab_path, max_seq_len):
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path)
        self.train_data = self.data_preprocess(data_path)

    def data_preprocess(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = f.read()
        if "\r\n" in raw_data:
            raw_data = raw_data.split("\r\n\r\n")
        else:
            raw_data = raw_data.split("\n\n")

        logger.info('=' * 70)
        logger.info(f'开始对{len(raw_data)}条对话进行预处理操作: 添加[CLS]和[SEP]字符')
        train_data = []
        for dialogue in tqdm(raw_data, desc='data_preprocess'):
            utterances = dialogue.split("\n")
            dialogue_ids = [self.tokenizer.cls_token_id]  # 每个dialogue以[CLS]开头
            for utterance in utterances:
                dialogue_ids.extend([self.tokenizer.convert_tokens_to_ids(word) for word in utterance])
                dialogue_ids.append(self.tokenizer.sep_token_id)  # 每个utterance之后添加[SEP]，表示utterance结束
            if len(dialogue_ids) > self.max_seq_len:  # 截断
                dialogue_ids = dialogue_ids[:self.max_seq_len]
            train_data.append(dialogue_ids)
        logger.info('=' * 70)
        return train_data

    def __getitem__(self, idx):
        sample = self.train_data[idx]
        return torch.tensor(sample)

    def __len__(self):
        return len(self.train_data)

    def collate_fn(self, batch, ignore_index=-100):
        """transformer的GPT model会内部shift target, 并且计算损失时会忽略ignore_index的值, 默认-100"""
        batch_size = len(batch)
        max_len = max(sample.shape[0] for sample in batch)

        batch_inputs = torch.zeros(batch_size, max_len, dtype=torch.long)
        batch_labels = torch.zeros(batch_size, max_len, dtype=torch.long)
        batch_inputs.fill_(self.tokenizer.pad_token_id)
        batch_labels.fill_(ignore_index)

        for i, sample in enumerate(batch):
            batch_inputs[i][:sample.shape[0]] = sample
            batch_labels[i][:sample.shape[0]] = sample

        return batch_inputs, batch_labels
