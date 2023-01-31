# -*- coding: utf-8 -*-
"""
https://arxiv.org/pdf/1508.04025)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(Seq2SeqEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)

    def forward(self, input_ids, seq_lens):
        embeddings = self.embedding(input_ids)
        embeddings = pack_padded_sequence(embeddings, seq_lens, batch_first=True, enforce_sorted=False)
        output_states, (h_t, h_c) = self.lstm(embeddings)
        output_states, _ = pad_packed_sequence(output_states, batch_first=True)
        return output_states, h_t


class Seq2SeqAttention(nn.Module):
    """dot product Attention Mechanism"""

    def __init__(self):
        super(Seq2SeqAttention, self).__init__()

    def forward(self, decoder_state_t, encoder_states):
        batch_size, seq_lens, hidden_size = encoder_states.shape
        decoder_state_t = decoder_state_t.unsqueeze(1).repeat(1, seq_lens, 1)
        score = torch.sum(decoder_state_t * encoder_states, dim=-1)
        att_prob = F.softmax(score, dim=-1)
        context = torch.sum(att_prob.unsqueeze(-1) * encoder_states, dim=1)
        return att_prob, context


class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, start_id, end_id):
        super(Seq2SeqDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_size)
        self.proj_layer = nn.Linear(hidden_size * 2, num_classes)
        self.att_mechanism = Seq2SeqAttention()
        self.num_classes = num_classes
        self.start_id = start_id
        self.end_id = end_id

    def forward(self, shifted_target_ids, encoder_states):
        shifted_target = self.embedding(shifted_target_ids)
        batch_size, target_lens, embedding_dim = shifted_target.shape
        batch_size, seq_lens, hidden_size = encoder_states.shape

        logits = torch.zeros(batch_size, target_lens, self.num_classes)
        att_probs = torch.zeros(batch_size, target_lens, seq_lens)

        for t in range(target_lens):
            decoder_input_t = shifted_target[:, t, :]
            if t == 0:
                h_t, c_t = self.lstm_cell(decoder_input_t)
            else:
                h_t, c_t = self.lstm_cell(decoder_input_t, (h_t, c_t))

            att_prob, context = self.att_mechanism(h_t, encoder_states)
            logits[:, t, :] = self.proj_layer(torch.cat([h_t, context], dim=-1))
            att_probs[:, t, :] = att_prob

        return att_probs, logits

    def inference(self, encoder_states):
        target_id = self.start_id
        h_t, c_t = None, None
        result = []

        while True:
            decoder_input_t = self.embedding(target_id)
            if h_t is None:
                h_t, c_t = self.lstm_cell(decoder_input_t)
            else:
                h_t, c_t = self.lstm_cell(decoder_input_t, (h_t, c_t))

            att_prob, context = self.att_mechanism(h_t, encoder_states)
            logits = self.proj_layer(torch.cat([h_t, context], dim=-1))
            target_id = torch.argmax(logits, dim=-1)
            result.append(target_id.item())

            if target_id == self.end_id:
                break

        return result


class Seq2SeqModel(nn.Module):
    def __init__(self, source_vocab_size, embedding_dim, hidden_size, target_vocab_size, num_classes, start_id, end_id):
        super(Seq2SeqModel, self).__init__()
        self.encoder = Seq2SeqEncoder(source_vocab_size, embedding_dim, hidden_size)
        self.decoder = Seq2SeqDecoder(target_vocab_size, embedding_dim, hidden_size, num_classes, start_id, end_id)

    def forward(self, input_ids, seq_lens, shifted_target_idx):
        encoder_states, final_h = self.encoder(input_ids, seq_lens)
        att_probs, logits = self.decoder(shifted_target_idx, encoder_states)
        return att_probs, logits


if __name__ == '__main__':
    source_length = 3
    target_length = 4
    embedding_dim = 8
    hidden_size = 16
    num_classes = 10
    batch_size = 2
    start_id = end_id = 0
    source_vocab_size = target_vocab_size = 100

    input_seq_ids = torch.randint(source_vocab_size, size=(batch_size, source_length)).to(torch.long)
    seq_lens = torch.tensor([source_length] * batch_size, dtype=torch.long)

    target_ids = torch.randint(target_vocab_size, size=(batch_size, target_length)).to(torch.long)
    target_ids = torch.cat([target_ids, end_id * torch.ones(batch_size, 1)], dim=-1)

    shifted_target_ids = torch.cat([end_id * torch.ones(batch_size, 1), target_ids[:, 1:]], dim=-1).to(torch.long)

    model = Seq2SeqModel(source_vocab_size, embedding_dim, hidden_size, target_vocab_size,
                         num_classes, start_id, end_id)
    probs, logits = model(input_seq_ids, seq_lens, shifted_target_ids)
    print(probs)
    print(logits)
