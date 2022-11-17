import torch.nn as nn
import torch
from transformers import BertModel


class CasRel(nn.Module):
    def __init__(self, config):
        super(CasRel, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)

        self.sub_heads_linear = nn.Linear(config.bert_dim, 1)
        torch.nn.init.xavier_uniform_(self.sub_heads_linear.weight)

        self.sub_tails_linear = nn.Linear(config.bert_dim, 1)
        torch.nn.init.xavier_uniform_(self.sub_tails_linear.weight)

        self.obj_heads_linear = nn.Linear(config.bert_dim, config.num_relations)
        torch.nn.init.xavier_uniform_(self.obj_heads_linear.weight)

        self.obj_tails_linear = nn.Linear(config.bert_dim, config.num_relations)
        torch.nn.init.xavier_uniform_(self.obj_tails_linear.weight)

        self.layer_norm = nn.LayerNorm(config.bert_dim)

    def get_encoded_text(self, token_ids, mask):
        encoded_text = self.bert(token_ids, attention_mask=mask)[0]

        return encoded_text

    def get_subs(self, encoded_text):
        encoded_text = self.layer_norm(encoded_text)
        pred_sub_heads = torch.sigmoid(self.sub_heads_linear(encoded_text))
        pred_sub_tails = torch.sigmoid(self.sub_tails_linear(encoded_text))

        return pred_sub_heads, pred_sub_tails

    def get_objs_for_specific_sub(self, sub_head_mapping, sub_tail_mapping, encoded_text):
        # sub_head_mapping [batch, 1, seq] * encoded_text [batch, seq, dim]
        sub_head = torch.matmul(sub_head_mapping, encoded_text)
        sub_tail = torch.matmul(sub_tail_mapping, encoded_text)

        sub = (sub_head + sub_tail) / 2
        encoded_text = encoded_text + sub

        pred_obj_heads = torch.sigmoid(self.obj_heads_linear(encoded_text))
        pred_obj_tails = torch.sigmoid(self.obj_tails_linear(encoded_text))

        return pred_obj_heads, pred_obj_tails

    def forward(self, token_ids, mask, sub_head, sub_tail):
        encoded_text = self.get_encoded_text(token_ids, mask)
        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_text)

        sub_head_mapping = sub_head.unsqueeze(1)
        sub_tail_mapping = sub_tail.unsqueeze(1)

        pred_obj_heads, pre_obj_tails = self.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping, encoded_text)

        return {
            "sub_heads": pred_sub_heads,
            "sub_tails": pred_sub_tails,
            "obj_heads": pred_obj_heads,
            "obj_tails": pre_obj_tails,
        }


class FGM:
    """
    Fast Gradient Method
    """
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
