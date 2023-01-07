import os
import re
import json
import sys
import time
import pickle
from collections import Counter

import jieba
import pkuseg
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm


class Vocabulary:
    def __init__(self, data_path, cut_method="jieba"):
        self.path = data_path
        self.cut_method = cut_method

        self.corpus = []
        self.word2idx = {}
        self.idx2word = {}
        self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        """load data"""
        with open(self.path, "r", encoding="utf-8") as f:
            raw_data = f.readlines()

        if self.cut_method == "pkuseg":
            seg = pkuseg.pkuseg()
            cut = seg.cut
        elif self.cut_method == "jieba":
            cut = jieba.lcut
        else:
            raise ValueError("cut_method must be pkuseg or jieba")

        for sentence in tqdm(raw_data, desc="cut data"):
            sentence = sentence.strip()
            pattern = re.compile("[^\u4e00-\u9fa5]")  # 去除英文和数字，只保留中文字符
            text = re.sub(pattern, '', sentence)
            text = cut(text)
            self.corpus.extend(text)

    def build_vocabulary(self, max_vocab_size):
        """build word to idx map, idx to word map"""
        count = Counter(self.corpus)
        word_size = sum(count.values())
        word2count = dict(count.most_common(max_vocab_size - 1))
        word2count["<unk>"] = word_size - sum(word2count.values())

        self.word2idx = {word: idx for idx, word in enumerate(word2count.keys())}
        self.idx2word = {idx: word for idx, word in enumerate(word2count.keys())}

        print(f'Total build {len(self.word2idx)} words')
        save_pickle({'word2idx': self.word2idx, 'idx2word': self.idx2word}, './data/vocab.pkl')

        return self.word2idx, self.idx2word

    def build_cooccurance_matrix(self, window_size):
        """build cooccurance matrix"""
        start = time.time()
        print('build cooccurance matrix ...')

        encoded_words = [self.word2idx.get(word, self.word2idx["<unk>"]) for word in self.corpus]
        word_size = len(self.word2idx)
        corpus_len = len(encoded_words)
        cooccur_matrix = torch.zeros(word_size, word_size)

        for idx, center_word_idx in enumerate(tqdm(encoded_words, desc='build cooccurance matrix')):
            context_span = list(range(idx - window_size, idx)) + list(range(idx + 1, idx + window_size + 1))
            context_span = [i % corpus_len for i in context_span]
            context_span_idx = [encoded_words[i] for i in context_span]
            for context_word_idx in context_span_idx:
                cooccur_matrix[center_word_idx][context_word_idx] += 1

        cost = time.time() - start
        save_pickle(cooccur_matrix, './data/cooccur_matrix.pkl')
        print(f'cooccurance matrix consumed {sys.getsizeof(cooccur_matrix) / 1024 / 1024:0.2f} MB')
        print(f'building cooccurance matrix took {cost / 60:0.2f} minutes')

        return cooccur_matrix


class MyDataset(Dataset):
    def __init__(self, cooccur_matrix, weight_matrix):
        super(MyDataset, self).__init__()
        self.cooccur_matrix = cooccur_matrix
        self.weight_matrix = weight_matrix
        if os.path.exists('./data/train_data.pkl'):
            self.train_data = load_pickle('./data/train_data.pkl')
        else:
            self.train_data = self.build_dataset()

    def build_dataset(self):
        start = time.time()
        print('build dataset ...')
        train_data = []
        for i in range(self.cooccur_matrix.shape[0]):
            for j in range(self.cooccur_matrix.shape[1]):
                if self.cooccur_matrix[i][j] != 0:
                    train_data.append((torch.tensor(i), torch.tensor(j)))
        cost = time.time() - start
        print(f'building dataset matrix took {cost / 60:0.2f} minutes')
        save_pickle(train_data, './data/train_data.pkl')
        return train_data

    def __getitem__(self, idx):
        i, j = self.train_data[idx]

        return i, j, self.cooccur_matrix[i][j], self.weight_matrix[i][j]

    def __len__(self):
        return len(self.train_data)


def build_weight_matrix(cooccur_matrix):
    """build weight matrix"""
    start = time.time()
    weight_matrix = torch.zeros_like(cooccur_matrix)

    print('build weight matrix ...')
    for i in range(weight_matrix.shape[0]):
        for j in range(weight_matrix.shape[1]):
            weight_matrix[i][j] = 1 if cooccur_matrix[i][j] >= 100 else np.power(cooccur_matrix[i][j] / 100, 3 / 4)

    cost = time.time() - start
    save_pickle(weight_matrix, './data/weight_matrix.pkl')
    print(f'building weight matrix took {cost / 60:0.2f} minutes')

    return weight_matrix


def get_similar_tokens(query, k, word_embedding, word2idx, idx2word):
    query_embedding = word_embedding[word2idx[query]]
    cos = np.dot(word_embedding, query_embedding) / \
          np.sqrt(np.sum(word_embedding * word_embedding, axis=1) * np.sum(query_embedding * query_embedding) + 1e-9)
    flat = cos.flatten()
    indices = np.argsort(-flat)[:k]
    for i in indices:
        print('for word %s, the similar word is %s' % (query, idx2word[i]))


def save_loss_curve(train_loss):
    plt.figure()
    ax = plt.axes()
    ax.spines['top'].set_visible(False)  # 去除上边框
    ax.spines['right'].set_visible(False)  # 去除右边框

    plt.xlabel('iters')
    plt.ylabel('loss')

    plt.plot(range(len(train_loss)), train_loss, linewidth=1, linestyle="solid", label="train loss")
    plt.legend()
    plt.title('Loss curve')
    plt.savefig('loss.png')


def save_dict(dic, name):
    with open(name, "w", encoding="utf-8") as f:
        json.dump(dic, f, ensure_ascii=False)


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
