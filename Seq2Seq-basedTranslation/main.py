# -*- coding: utf-8 -*-
import pickle

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data_path, en_word_2_index, cn_word_2_index):
        all_data = pd.read_csv(data_path)
        self.en_data = list(all_data['english'])
        self.cn_data = list(all_data['chinese'])
        self.en_word_2_index = en_word_2_index
        self.cn_word_2_index = cn_word_2_index

    def __getitem__(self, index):
        en = self.en_data[index]
        ch = self.cn_data[index]

        en_inputs = [self.en_word_2_index[i] for i in en]
        cn_inputs = [self.cn_word_2_index[i] for i in ch]

        return en_inputs, cn_inputs

    def __len__(self):
        return len(self.cn_data)

    def collate_fn(self, batch_data):
        en_index, cn_index = [], []
        en_len, cn_len = [], []

        for en, cn in batch_data:
            en_index.append(en)
            cn_index.append(cn)
            en_len.append(len(en))
            cn_len.append(len(cn))

        max_en_len = max(en_len)
        max_cn_len = max(cn_len)

        en_index = [i + [en_word_2_index['<PAD>']] * (max_en_len - len(i)) for i in en_index]
        cn_index = [
            [self.cn_word_2_index['<BOS>']] + i + [self.cn_word_2_index['<EOS>']] + [self.cn_word_2_index['<PAD>']] * (
                        max_cn_len - len(i))
            for i in cn_index]

        en_index = torch.tensor(en_index)
        cn_index = torch.tensor(cn_index)

        return en_index, cn_index


class Encoder(nn.Module):
    def __init__(self, encoder_embedding_dim, encoder_hidden_dim, en_vocab_len):
        super().__init__()
        self.embedding = nn.Embedding(en_vocab_len, encoder_embedding_dim)
        self.lstm = nn.LSTM(encoder_embedding_dim, encoder_hidden_dim, batch_first=True)

    def forward(self, en_inputs):
        en_embedding = self.embedding(en_inputs)
        _, encoder_hidden = self.lstm(en_embedding)

        return encoder_hidden


class Decoder(nn.Module):
    def __init__(self, decoder_embedding_dim, decoder_hidden_dim, cn_vocab_len):
        super().__init__()
        self.embedding = nn.Embedding(cn_vocab_len, decoder_embedding_dim)
        self.lstm = nn.LSTM(decoder_embedding_dim, decoder_hidden_dim, batch_first=True)

    def forward(self, decoder_input, hidden):
        embedding = self.embedding(decoder_input)
        decoder_output, decoder_hidden = self.lstm(embedding, hidden)

        return decoder_output, decoder_hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder_embedding_dim, encoder_hidden_dim, en_vocab_len, decoder_embedding_dim,
                 decoder_hidden_dim, ch_vocab_len):
        super().__init__()
        self.encoder = Encoder(encoder_embedding_dim, encoder_hidden_dim, en_vocab_len)
        self.decoder = Decoder(decoder_embedding_dim, decoder_hidden_dim, ch_vocab_len)
        self.classifier = nn.Linear(decoder_hidden_dim, ch_vocab_len)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, en_inputs, cn_inputs):
        decoder_input = cn_inputs[:, :-1]
        label = cn_inputs[:, 1:]

        encoder_hidden = self.encoder(en_inputs)
        decoder_output, _ = self.decoder(decoder_input, encoder_hidden)

        pre = self.classifier(decoder_output)
        loss = self.loss_fn(pre.reshape(-1, pre.shape[-1]), label.reshape(-1))

        return loss


def translate(model, sentence, en_word_2_index, ch_word_2_index, ch_index_2_word, device):
    en_index = torch.tensor([[en_word_2_index[i] for i in sentence]], device=device)

    result = []
    encoder_hidden = model.encoder(en_index)
    decoder_input = torch.tensor([[ch_word_2_index["<BOS>"]]], device=device)

    decoder_hidden = encoder_hidden
    while True:
        decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
        logit = model.classifier(decoder_output)

        w_index = int(torch.argmax(logit, dim=-1))
        word = ch_index_2_word[w_index]

        if word == "<EOS>" or len(result) > 50:
            break

        result.append(word)
        decoder_input = torch.tensor([[w_index]], device=device)

    print("译文: ", "".join(result))
    
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('data/ch.vec', 'rb') as f1:
        _, cn_word_2_index, cn_index_2_word = pickle.load(f1)

    with open('data/en.vec', 'rb') as f2:
        _, en_word_2_index, en_inputs_2_word = pickle.load(f2)

    cn_vocab_len = len(cn_word_2_index)
    en_vocab_len = len(en_word_2_index)

    cn_word_2_index.update({'<PAD>': cn_vocab_len, '<BOS>': cn_vocab_len + 1, '<EOS>': cn_vocab_len + 2})
    en_word_2_index.update({'<PAD>': en_vocab_len})

    cn_index_2_word += ['<PAD>', '<BOS>', '<EOS>']
    en_inputs_2_word += ['<PAD>']

    cn_vocab_len += 3
    en_vocab_len += 1

    encoder_embedding_dim = 128
    encoder_hidden_dim = 128
    decoder_embedding_dim = 128
    decoder_hidden_dim = 128

    batch_size = 8
    epochs = 40
    lr = 0.001

    dataset = MyDataset('data/translate.csv', en_word_2_index, cn_word_2_index)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    model = Seq2Seq(encoder_embedding_dim, encoder_hidden_dim, en_vocab_len, decoder_embedding_dim, decoder_hidden_dim,
                    cn_vocab_len)
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for en_inputs, cn_inputs in dataloader:
            loss = model(en_inputs, cn_inputs)
            loss.backward()
            opt.step()
            opt.zero_grad()

        print(f'loss:{loss:.3f}')

    while True:
        s = input('请输入英文: ')
        translate(model, s, en_word_2_index, cn_word_2_index, cn_index_2_word, device)
