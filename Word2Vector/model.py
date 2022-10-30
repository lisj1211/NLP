# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class SkipGram(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.in_embedding = nn.Embedding(vocab_size, d_model)
        torch.nn.init.xavier_uniform(self.in_embedding.weight.data)
        self.out_embedding = nn.Embedding(vocab_size, d_model)
        torch.nn.init.xavier_uniform(self.out_embedding.weight.data)

    def forward(self, center_word, pos_word, neg_word):
        """
        :param center_word:  [batch_size]
        :param pos_word:  [batch_size, WINDOW_SIZE * 2]
        :param neg_word:  [batch_size, WINDOW_SIZE * 2 * K]
        :return:
        """
        # center_word_embedding: [batch_size, embedding_dim]
        center_word_embedding = self.in_embedding(center_word)

        # pos_word_embedding: [batch_size, WINDOW_SIZE * 2, embedding_dim]
        pos_word_embedding = self.out_embedding(pos_word)

        # neg_word_embedding: [batch_size, WINDOW_SIZE * 2 * K, embedding_dim]
        neg_word_embedding = self.out_embedding(neg_word)

        # log_pos: [batch_size, WINDOW_SIZE * 2]
        log_pos = torch.bmm(pos_word_embedding, center_word_embedding.unsqueeze(2)).squeeze()

        # log_neg: [batch_size, WINDOW_SIZE * 2 * K]
        log_neg = torch.bmm(neg_word_embedding, -center_word_embedding.unsqueeze(2)).squeeze()

        loss = nn.functional.logsigmoid(log_pos).sum(1) + nn.functional.logsigmoid(log_neg).sum(1)
        return -loss.mean()

    def get_word_embedding(self):
        return self.in_embedding.weight.data.cpu().numpy()
