from tqdm import tqdm
import re
import json
from collections import Counter
from itertools import chain

import pkuseg
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import BertTokenizer


class LoadData:
    def __init__(self):
        self.label_map = {}
        self.word2id = {}
        self.id2word = {}

    def load_corpus(self, file_path, stopword_path, label_map_path=None):
        """
        加载json格式文本数据，分词预处理
        """
        sentences = []
        labels = []

        stopwords = [line.strip() for line in open(stopword_path, 'r', encoding='utf-8').readlines()]
        seg = pkuseg.pkuseg()
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines(), desc='reading and preprocessing corpus'):
                tmp = json.loads(line)
                label, text = tmp["label"], tmp["text"]
                pattern = re.compile("[^\u4e00-\u9fa5]")  # 只保留中文
                text = re.sub(pattern, '', text)  # 把文本中匹配到的字符替换成空字符
                cut_words = seg.cut(text)
                sentences.append([word for word in cut_words if word not in stopwords])

                labels.append(int(label))

        if label_map_path is not None:
            self.label_map = json.load(open(label_map_path, 'r', encoding='utf-8'))

        return sentences, labels

    def build_vocabulary(self, sents, word_size=10000, min_freq=5):
        count = Counter(chain(*sents))
        valid_words = count.most_common(word_size - 2)
        valid_words = [word for word, freq in valid_words if freq >= min_freq]
        self.word2id = {word: idx+2 for idx, word in enumerate(valid_words)}  # 2个位置要留给 '<PAD>', '<UNK>'
        self.id2word = {idx+2: word for idx, word in enumerate(valid_words)}

        self.word2id['<PAD>'] = 0
        self.word2id['<UNK>'] = 1
        self.id2word[0] = '<PAD>'
        self.id2word[1] = '<UNK>'

        return self.word2id, self.id2word

    def save_vocab(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(dict(word2id=self.word2id, label_map=self.label_map), f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_vocab(file_path):
        with open(file_path, 'r') as f:
            entry = json.load(f)
        word2id = entry['word2id']
        label_map = entry['label_map']

        return word2id, label_map


class MyDataset(Dataset):
    def __init__(self, sents, labels, word2id):
        self.sents = sents
        self.word2id = word2id
        self.labels = labels
        self.words2indices()

    def words2indices(self):
        """
        将sents转为number index
        """
        if isinstance(self.sents[0], list):
            self.sents = [[self.word2id.get(word, 1) for word in sent] for sent in self.sents]
        else:
            self.sents = [[self.word2id.get(word, 1) for word in self.sents]]

    def __getitem__(self, idx):
        return self.sents[idx], self.labels[idx]

    def __len__(self):
        return len(self.sents)


def load_bert_data(data_path):
    sents = []
    labels = []
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = f.readlines()
    for line in raw_data:
        data = json.loads(line)
        sents.append(data["text"])
        labels.append(data["label"])

    return sents, labels


class BertDataset(Dataset):
    """
    因为该数据集文本长度均值在900左右，大于Bert模型的标准输入512（2个为特殊token），
    根据论文：https://arxiv.org/pdf/1905.05583.pdf  选择前128和后382个字
    """
    def __init__(self, data, label, cache_dir):
        self.data = data
        self.label = label
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir=cache_dir)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.label[idx]
        text = self.tokenizer.tokenize(text)
        if len(text) > 510:
            text = text[:128] + text[-382:]
        dic = self.tokenizer.encode_plus(text, max_length=512, padding='max_length')
        return dic['input_ids'], dic['attention_mask'], label

    def __len__(self):
        return len(self.label)


def my_collate_fn(batch_list):
    sents, labels = zip(*batch_list)
    sents = pad_sents(sents, 0)

    return sents, labels


def bert_collate_fn(batch_list):
    sents, masks, labels = zip(*batch_list)

    return sents, masks, labels


def pad_sents(sents, pad_token):
    """pad句子, word2id['<PAD>']=0"""
    sents_padded = []
    lengths = [len(s) for s in sents]
    max_len = max(lengths)
    for sent in sents:
        sent_padded = sent + [pad_token] * (max_len - len(sent))
        sents_padded.append(sent_padded)
    return sents_padded


def acc_and_f1(preds, labels):
    acc = accuracy_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')

    return acc, f1

