import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertModel


class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_label, dropout=0.2, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embedding_dim, num_label)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text.shape: [batch_size, seq_len]
        embedded = self.embedding(text)
        # embedded.shape: [batch_size, seq_len, embed_size]
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        pooled = self.dropout(pooled)

        return self.fc(pooled)


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=2, bidirectional=True, dropout=0.2, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text.shape: [batch_size, seq_len]
        embedded = self.embedding(text)
        # embedded.shape: [batch_size, seq_len, embed_size]
        output, _ = self.lstm(embedded)
        # output: [batch_size, seq_len, 2*hidden_dim ]
        hidden = self.dropout(output[:, -1, :])
        # 取最后一个时刻的输出

        return self.fc(hidden)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filter,
                 filter_sizes, num_class, dropout=0.2, pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.conv1d = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim, out_channels=num_filter, kernel_size=kernel)
                                     for kernel in filter_sizes])

        self.fc = nn.Linear(len(filter_sizes) * num_filter, num_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.permute(0, 2, 1)
        conv_out = [F.relu(conv(embedded)) for conv in self.conv1d]
        pooled = [F.max_pool1d(conv, conv.shape[-1]).squeeze() for conv in conv_out]
        cat = self.dropout(torch.cat(pooled, dim=1))

        return self.fc(cat)


class BertClassify(nn.Module):
    def __init__(self, num_classes, cache_dir="", freeze=False):
        super(BertClassify, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese', cache_dir=cache_dir)
        if freeze:  # 冻结bert参数
            for p in self.bert.parameters():
                p.requires_grad = False
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_idx, att_mask):
        pooled = self.bert(input_idx, attention_mask=att_mask)[1]
        out = self.fc(pooled)
        return out
