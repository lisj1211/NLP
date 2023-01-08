import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class LstmDataset(Dataset):
    def __init__(self, data, tags, word2idx, label_map):
        self.label2idx = label_map
        self.inputs, self.labels = self.trans_to_lstm_inputs(word2idx, data, tags)

    def trans_to_lstm_inputs(self, word2idx, data, tags):
        inputs, labels = [], []
        for word_list, tag_list in zip(data, tags):
            inputs.append([word2idx.get(word, 1) for word in word_list])  # 1表示 <UNK>
            labels.append([self.label2idx[tag] for tag in tag_list])

        return inputs, labels

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.labels[idx]), torch.tensor(len(self.inputs[idx]))

    def __len__(self):
        return len(self.labels)


class BertDataset(Dataset):
    def __init__(self, config, data, tags):
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)
        self.label2idx = config.label2idx
        self.inputs, self.masks, self.labels = self.trans_to_bert_inputs(data, tags)

    def trans_to_bert_inputs(self, data, tags):
        inputs, masks, labels = [], [], []
        for word_list, tag_list in zip(data, tags):
            word_list = ['[CLS]', *word_list, '[SEP]']  # add special token
            tag_list = ['O', *tag_list, 'O']  # set special token's tag to '0'

            idx = self.tokenizer.convert_tokens_to_ids(word_list)
            assert len(idx) == len(tag_list), f"For {word_list}, the length of inputs and tags must be same"

            inputs.append(idx)
            labels.append([self.label2idx[tag] for tag in tag_list])
            masks.append([1] * len(idx))

        return inputs, masks, labels

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.masks[idx], dtype=torch.bool), torch.tensor(self.labels[idx])

    def __len__(self):
        return len(self.labels)
