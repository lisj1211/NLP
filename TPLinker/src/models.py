import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel


class TaggingProjector(nn.Module):
    """relation classifier"""
    def __init__(self, hidden_size, num_relations, name='rel_proj'):
        super().__init__()
        self.name = name
        self.rel_classifier = [nn.Linear(hidden_size, 3) for _ in range(num_relations)]
        for index, fc in enumerate(self.rel_classifier):
            self.register_parameter(f'{self.name}_weights_{index}', fc.weight)
            self.register_parameter(f'{self.name}_bias_{index}', fc.bias)

    def forward(self, hidden_state: torch.Tensor):
        """
        Args:
            hidden_state: Tensor, shape (batch_size, 1+2+...+seq_len, hidden_size)
        Returns:
            outputs: Tensor, shape (batch_size, num_relations, 1+2+...+seq_len, num_tags=3)
        """
        outputs = []
        for fc in self.rel_classifier:
            outputs.append(fc(hidden_state))
        outputs = torch.stack(outputs, dim=1)
        outputs = torch.softmax(outputs, dim=-1)
        return outputs


class ConcatHandshaking(nn.Module):
    """Handshaking"""
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden_state: torch.Tensor):
        """
        Args:
            hidden_state: Tensor, shape (batch_size, seq_len, hidden_size)
        Returns:
            handshaking_hiddens: Tensor, shape (batch_size, 1+2+...+seq_len, hidden_size)
        """
        seq_len = hidden_state.size()[1]
        handshaking_hiddens = []
        for i in range(seq_len):
            _h = hidden_state[:, i, :]
            repeat_hidden = _h[:, None, :].repeat(1, seq_len - i, 1)
            visible_hidden = hidden_state[:, i:, :]
            shaking_hidden = torch.cat([repeat_hidden, visible_hidden], dim=-1)
            shaking_hidden = self.fc(shaking_hidden)
            shaking_hidden = torch.tanh(shaking_hidden)
            handshaking_hiddens.append(shaking_hidden)
        handshaking_hiddens = torch.cat(handshaking_hiddens, dim=1)
        return handshaking_hiddens


class TPLinker(nn.Module):
    """TPLinker model"""
    def __init__(self, hidden_size, num_relations):
        super().__init__()
        self.handshaking = ConcatHandshaking(hidden_size)
        self.h2t_proj = nn.Linear(hidden_size, 2)
        self.h2h_proj = TaggingProjector(hidden_size, num_relations, name='h2hproj')
        self.t2t_proj = TaggingProjector(hidden_size, num_relations, name='t2tproj')

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden: Tensor, output of BERT or BiLSTM, shape (batch_size, seq_len, hidden_size)
        Returns:
            h2t_hidden: Tensor, shape (batch_size, 1+2+...+seq_len, 2),
                logits for entity recognition
            h2h_hidden: Tensor, shape (batch_size, num_relations, 1+2+...+seq_len, 3),
                logits for relation recognition
            t2t_hidden: Tensor, shape (batch_size, num_relations, 1+2+...+seq_len, 3),
                logits for relation recognition
        """
        handshaking_hidden = self.handshaking(hidden)
        h2t_hidden, rel_hidden = handshaking_hidden, handshaking_hidden
        h2t_hidden = self.h2t_proj(h2t_hidden)
        h2h_hidden = self.h2h_proj(rel_hidden)
        t2t_hidden = self.t2t_proj(rel_hidden)
        return h2t_hidden, h2h_hidden, t2t_hidden


class TPLinkerBert(BertPreTrainedModel):
    def __init__(self, config, num_relations):
        super().__init__(config)
        self.bert = BertModel(config)
        self.tplinker = TPLinker(hidden_size=config.hidden_size, num_relations=num_relations)

    def forward(self, input_ids, attn_mask):
        sequence_output = self.bert(input_ids, attn_mask)[0]
        h2t_outputs, h2h_outputs, t2t_outputs = self.tplinker(sequence_output)
        return h2t_outputs, h2h_outputs, t2t_outputs
