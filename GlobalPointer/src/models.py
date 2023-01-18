import torch
import torch.nn as nn

from transformers import BertModel, BertPreTrainedModel


class GlobalPointer(BertPreTrainedModel):
    def __init__(self, config, ent_type_size, inner_dim, RoPE=True):
        super().__init__(config)
        self.bert = BertModel(config)
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = self.bert.config.hidden_size
        self.RoPE = RoPE

        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        return embeddings

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.bert(input_ids, attention_mask)[0]
        batch_size, seq_len = last_hidden_state.size()[:2]

        outputs = self.dense(last_hidden_state)
        # outputs: (batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        # outputs: (batch_size, seq_len, ent_type_size, inner_dim*2)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]
        # qw, kw: (batch_size, seq_len, ent_type_size, inner_dim)

        if self.RoPE:
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            pos_emb = pos_emb.to(input_ids.device)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5
