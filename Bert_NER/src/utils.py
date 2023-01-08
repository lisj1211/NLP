import pickle
import random
from collections import Counter
from itertools import chain

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


class Vocabulary:
    def __init__(self, config):
        self.config = config
        self.word2idx = {}
        self.idx2word = {}

    def build_vocab(self, data):
        count = Counter(chain(*data))
        valid_words = count.most_common(self.config.max_vocab - 2)
        valid_words = [word for word, freq in valid_words if freq >= self.config.min_freq]
        self.word2idx = {word: idx + 2 for idx, word in enumerate(valid_words)}  # 2个位置要留给 '<PAD>', '<UNK>'
        self.idx2word = {idx + 2: word for idx, word in enumerate(valid_words)}
        self.word2idx['<PAD>'] = 0
        self.word2idx['<UNK>'] = 1
        self.idx2word[0] = '<PAD>'
        self.idx2word[1] = '<UNK>'
        return self.word2idx, self.idx2word

    def save_vocab(self, vocab_path):
        save_pickle({'word2idx': self.word2idx, 'idx2word': self.idx2word}, vocab_path)

    def load_vocab(self, vocab_path):
        dic = load_pickle(vocab_path)
        self.word2idx = dic['word2idx']
        self.idx2word = dic['idx2word']
        return self.word2idx, self.idx2word
    
    
def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def rnn_collate_fn(batch):
    batch_inputs, batch_labels, batch_len = zip(*batch)
    batch_inputs = pad_sequence(batch_inputs, batch_first=True)
    batch_masks = create_mask(batch_len)
    batch_labels = pad_sequence(batch_labels, batch_first=True)
    batch_len = torch.tensor(batch_len)
    return batch_inputs, batch_masks, batch_len, batch_labels


def create_mask(length_list):
    max_len = max(length_list)
    masks = torch.zeros(len(length_list), max_len, dtype=torch.bool)
    for idx, length in enumerate(length_list):
        masks[idx][:length] = 1

    return masks


def bert_collate_fn(batch):
    batch_inputs, batch_masks, batch_labels = zip(*batch)
    batch_inputs = pad_sequence(batch_inputs, batch_first=True)
    batch_masks = pad_sequence(batch_masks, batch_first=True)
    batch_labels = pad_sequence(batch_labels, batch_first=True)
    return batch_inputs, batch_masks, batch_labels


def get_optimizer_params(config, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_params = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    return optimizer_params


def get_optimizer_params_for_bert_lstm_crf(config, model):
    bert_optimizer = list(model.bert.named_parameters())
    lstm_optimizer = list(model.lstm.named_parameters())
    classifier_optimizer = list(model.classifier.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_params = [
        {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
         'lr': config.lr * 5, 'weight_decay': config.weight_decay},
        {'params': [p for n, p in lstm_optimizer if any(nd in n for nd in no_decay)],
         'lr': config.lr * 5, 'weight_decay': 0.0},
        {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
         'lr': config.lr * 5, 'weight_decay': config.weight_decay},
        {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
         'lr': config.lr * 5, 'weight_decay': 0.0},
        {'params': model.crf.parameters(), 'lr': config.lr * 5}
    ]

    return optimizer_params


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
