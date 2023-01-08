import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torchcrf import CRF
from transformers import BertPreTrainedModel, BertModel


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_class, drop_out):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=2,
            dropout=drop_out,
            bidirectional=True,
        )
        self.classifier = nn.Linear(2 * hidden_size, num_class)
        self.crf = CRF(num_class, batch_first=True)
        self._init_crf_weight()

    def forward(self, input_idx, length_list):
        embedding = self.embedding(input_idx)
        packed = pack_padded_sequence(embedding, length_list, batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        logits = self.classifier(output)
        return logits

    def forward_with_crf(self, input_idx, input_mask, length_list, input_tag):
        logits = self.forward(input_idx, length_list)
        loss = -1 * self.crf(logits, input_tag, input_mask)
        return logits, loss

    def _init_crf_weight(self):
        for p in self.crf.parameters():
            torch.nn.init.uniform_(p, -1, 1)


class Bert(BertPreTrainedModel):
    def __init__(self, config, num_class, dropout):
        super(Bert, self).__init__(config)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(768, num_class)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input_ids, masks):
        encoded_text = self.bert(input_ids, attention_mask=masks)[0]
        encoded_text = self.dropout(encoded_text)
        logits = self.classifier(encoded_text)  # B, L, num_class

        return logits


class Bert_CRF(BertPreTrainedModel):
    def __init__(self, config, num_class, dropout):
        super(Bert_CRF, self).__init__(config)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(768, num_class)
        self.dropout = nn.Dropout(dropout)
        self.crf = CRF(num_class, batch_first=True)
        self.init_weights()
        self._init_crf_weight()

    def forward(self, input_ids, masks, input_labels=None):
        encoded_text = self.bert(input_ids, attention_mask=masks)[0]
        encoded_text = self.dropout(encoded_text)
        logits = self.classifier(encoded_text)  # B, L, num_class

        if input_labels is None:
            return logits
        else:
            loss = self.crf(logits, input_labels, masks) * (-1)
            return loss

    def _init_crf_weight(self):
        for p in self.crf.parameters():
            torch.nn.init.uniform_(p, -1, 1)


class Bert_LSTM_CRF(BertPreTrainedModel):
    def __init__(self, config, num_class, dropout):
        super(Bert_LSTM_CRF, self).__init__(config)
        self.bert = BertModel(config)
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=768,
            batch_first=True,
            num_layers=2,
            dropout=dropout,
            bidirectional=True,
        )
        self.classifier = nn.Linear(768 * 2, num_class)
        self.crf = CRF(num_class, batch_first=True)
        self.init_weights()
        self._init_crf_weight()

    def forward(self, input_ids, masks, input_labels=None):
        encoded_text = self.bert(input_ids, attention_mask=masks)[0]
        length_list = torch.tensor([torch.nonzero(mask).shape[0] for mask in masks])
        packed = pack_padded_sequence(encoded_text, length_list, batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        logits = self.classifier(output)

        if input_labels is None:
            return logits
        else:
            loss = self.crf(logits, input_labels, masks) * (-1)
            return loss

    def _init_crf_weight(self):
        for p in self.crf.parameters():
            torch.nn.init.uniform_(p, -1, 1)
