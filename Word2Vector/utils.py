# -*- coding: utf-8 -*-
import re
import math
import random
from collections import Counter
import json

import jieba
import pkuseg
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt


class Vocabulary:
    def __init__(self, data_path, MAX_VOCAB_SIZE=10000, MIN_FREQUENCE=2, cut_method="jieba"):
        self.path = data_path
        self.MAX_VOCAB_SIZE = MAX_VOCAB_SIZE
        self.MIN_FREQUENCE = MIN_FREQUENCE
        self.cut_method = cut_method

        self.corpus = []
        self.word2idx = {}
        self.idx2word = {}
        self.load_and_preprocess_data()
        self.subsampling()
        self.build_vocabulary()

    def load_and_preprocess_data(self):
        with open(self.path, "r", encoding="utf-8") as f:
            raw_data = f.readlines()

        if self.cut_method == "pkuseg":
            seg = pkuseg.pkuseg()
            cut = seg.cut
        elif self.cut_method == "jieba":
            cut = jieba.lcut
        else:
            raise ValueError("cut_method must be pkuseg or jieba")

        words = []
        for sentence in tqdm(raw_data, desc="preprocess data"):
            sentence = sentence.strip()
            pattern = re.compile("[^\u4e00-\u9fa5]")  # 去除英文和数字，只保留中文字符
            text = re.sub(pattern, '', sentence)
            text = cut(text)
            words.extend(text)

        word_counts = Counter(words)
        self.corpus = [word for word in words if word_counts[word] > self.MIN_FREQUENCE]  # 去除低频词

    def subsampling(self):  # 下采样作用是去除高频词，作用类似于去除停用词
        count = dict(Counter(self.corpus))
        word_freq = {w: c / len(self.corpus) for w, c in count.items()}
        t = 1e-5
        prob_drop = {w: max(0, 1 - math.sqrt(t / c)) for w, c in word_freq.items()}  # 过采样计算公式
        self.corpus = [word for word in self.corpus if random.random() < (1 - prob_drop[word])]

    def build_vocabulary(self):
        count = Counter(self.corpus)
        word_size = sum(count.values())
        word2count = dict(count.most_common(self.MAX_VOCAB_SIZE - 1))
        word2count["<unk>"] = word_size - sum(word2count.values())
        self.word2idx = {word: idx for idx, word in enumerate(word2count.keys())}
        self.idx2word = {idx: word for idx, word in enumerate(word2count.keys())}
        word_count = np.array([count for count in word2count.values()])
        word_frequence = (word_count / np.sum(word_count)) ** (3. / 4.)
        self.word_frequence = word_frequence / np.sum(word_frequence)


class MyDataset(Dataset):
    def __init__(self, word_list, word2idx, word_frequence, k, WINDOW_SIZE):
        super(MyDataset, self).__init__()
        self.WINDOW_SIZE = WINDOW_SIZE
        self.k = k
        self.encoded_words = torch.tensor([word2idx.get(word, word2idx["<unk>"]) for word in word_list])
        self.word2idx = word2idx
        self.word_frequence = torch.tensor(word_frequence)

    def __getitem__(self, idx):
        encoded_center_word = torch.as_tensor(self.encoded_words[idx])
        pos_span = list(range(idx - self.WINDOW_SIZE, idx)) + list(range(idx + 1, idx + self.WINDOW_SIZE + 1))
        pos_span = [pos_idx % len(self.encoded_words) for pos_idx in pos_span]
        encoded_pos_words = self.encoded_words[pos_span]
        encoded_neg_words = self.negative_sampling(encoded_pos_words)

        encoded_pos_words = torch.as_tensor(encoded_pos_words)
        encoded_neg_words = torch.as_tensor(encoded_neg_words)

        return encoded_center_word, encoded_pos_words, encoded_neg_words

    def negative_sampling(self, pos_list):
        while True:
            encoded_neg_words = torch.multinomial(self.word_frequence, self.k * len(pos_list), False)
            # 如果负采样采样到正样本，则重新采样
            flag = True
            for w in encoded_neg_words:
                if w in pos_list:
                    flag = False
            if flag:
                break
        return encoded_neg_words

    def __len__(self):
        return len(self.encoded_words)


def get_similar_tokens(query, k, word_embedding_path, word2idx_path, idx2word_path):
    word_embedding = np.load(word_embedding_path)

    with open(word2idx_path, 'r', encoding='utf-8') as f:
        word2idx = json.load(f)

    with open(idx2word_path, 'r', encoding='utf-8') as f:
        idx2word = json.load(f)

    query_embedding = word_embedding[word2idx[query]]
    cos = np.dot(word_embedding, query_embedding) / \
          np.sqrt(np.sum(word_embedding * word_embedding, axis=1) * np.sum(query_embedding * query_embedding) + 1e-9)
    flat = cos.flatten()
    indices = np.argsort(-flat)[:k]
    for i in indices:
        print('for word %s, the similar word is %s' % (query, idx2word[str(i)]))


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


def save_dict(dic, name):
    with open(name, "w", encoding="utf-8") as f:
        json.dump(dic, f, ensure_ascii=False)
