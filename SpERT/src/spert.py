# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertModel

from src.sampling import create_rel_mask
from src.utils import batch_index, padded_stack


def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """
    获取bert输出特定idx的embedding (比如 [CLS])
    :param h: [batch_size, seq_len, bert_dim]
    :param x: [batch_size, seq_len]
    :param token:
    :return:
    """
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    token_h = token_h[flat == token, :]

    return token_h


class SpERT(nn.Module):
    """
    Span-based Joint Entity and Relation Extraction with Transformer Pre-training
    https://arxiv.org/abs/1909.07755
    """
    def __init__(self, config):
        super(SpERT, self).__init__()

        self.bert = BertModel.from_pretrained(config.bert_path)

        self.rel_classifier = nn.Linear(config.bert_dim * 3 + config.size_embedding * 2, config.relation_types - 1)
        self.entity_classifier = nn.Linear(config.bert_dim * 2 + config.size_embedding, config.entity_types)
        self.size_embeddings = nn.Embedding(100, config.size_embedding)
        self.dropout = nn.Dropout(config.prop_drop)

        self.cls_token = 101  # tokenizer.convert_tokens_to_ids('[CLS]')

        if config.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                       entity_sizes: torch.tensor, relations: torch.tensor, rel_masks: torch.tensor):
        """
        :param encodings: [batch_size, sequence_len]
        :param context_masks: [batch_size, sequence_len]
        :param entity_masks: [batch_size, entity_size, sequence_len]
        :param entity_sizes: [batch_size, entity_size]
        :param relations: [batch_size, rel_size, 2]
        :param rel_masks: [batch_size, rel_size, sequence_len]
        :return:
        """
        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']
        # h.shape: [batch_size, sequence_len, bert_dim]

        # ==========classify entities==========
        size_embeddings = self.size_embeddings(entity_sizes)
        # size_embeddings.shape: [batch_size, entity_size, size_embedding]
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings)
        # entity_clf.shape: [batch_size, entity_size, entity_classes]
        # entity_spans_pool.shape: [batch_size, entity_size, bert_dim]

        # ==========classify relations==========
        h_large = h.unsqueeze(1).repeat(1, relations.shape[1], 1, 1)
        # h_large.shape: [batch_size, rel_size, sequence_len, bert_dim]

        rel_clf = self._classify_relations(entity_spans_pool, size_embeddings, relations, rel_masks, h_large)

        return entity_clf, rel_clf

    def _forward_inference(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                           entity_sizes: torch.tensor, entity_spans: torch.tensor, entity_sample_masks: torch.tensor):
        """

        :param encodings: [batch_size, sequence_len]
        :param context_masks: [batch_size, sequence_len]
        :param entity_masks: [batch_size, entity_size, sequence_len]
        :param entity_sizes: [batch_size, entity_size]
        :param entity_spans: [batch_size, entity_size, 2]
        :param entity_sample_masks: [batch_size, entity_size]
        :return:
        """
        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']
        # h.shape: [batch_size, sequence_len, bert_dim]

        ctx_size = context_masks.shape[-1]

        size_embeddings = self.size_embeddings(entity_sizes)
        # size_embeddings.shape: [batch_size, sequence_len, size_embedding]
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings)
        # entity_clf.shape.shape: [batch_size, entity_size, entity_classes]
        # entity_spans_pool.shape: [batch_size, entity_size, bert_dim]

        # ignore entity candidates that do not constitute an actual entity for relations (based on classifier)
        relations, rel_masks, rel_sample_masks = self._filter_spans(entity_clf, entity_spans,
                                                                    entity_sample_masks, ctx_size)
        # relations: [batch_size, rel_size, 2]
        # rel_masks: [batch_size, rel_size, sequence_len]
        # rel_sample_masks: [batch_size, rel_size]

        rel_sample_masks = rel_sample_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(1, relations.shape[1], 1, 1)

        rel_logits = self._classify_relations(entity_spans_pool, size_embeddings, relations, rel_masks, h_large)
        rel_clf = torch.sigmoid(rel_logits)
        rel_clf = rel_clf * rel_sample_masks  # mask

        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)

        return entity_clf, rel_clf, relations

    def _classify_entities(self, encodings, h, entity_masks, size_embeddings):
        """
        :param encodings: [batch_size, sequence_len]
        :param h: [batch_size, sequence_len, bert_dim]
        :param entity_masks: [batch_size, entity_size, sequence_len]
        :param size_embeddings: [batch_size, entity_size, size_embedding]
        :return: 
        """
        # 现在mask为true的地方是实体的位置，mask==0之后其他位置为true，实体位置为false，乘以非常小的数之后，相当于取出了实体
        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)
        # m.shape: [batch_size, entity_size, sequence_len, 1]

        entity_spans_pool = m + h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        # entity_spans_pool.shape: [batch_size, entity_size, sequence_len, bert_dim]

        entity_spans_pool = entity_spans_pool.max(dim=2)[0]
        # entity_spans_pool.shape: [batch_size, entity_size, bert_dim]

        cls_token = get_token(h, encodings, self.cls_token)
        # cls_token.shape: [batch_size, bert_dim]

        # ==========concatenate==========
        entity_repr = torch.cat([cls_token.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
                                 entity_spans_pool, size_embeddings], dim=2)
        # entity_repr.shape: [batch_size, entity_size, 2*bert_dim+size_embedding]

        entity_repr = self.dropout(entity_repr)

        # ==========classify entity candidates==========
        entity_clf = self.entity_classifier(entity_repr)
        # entity_clf: [batch_size, entity_size, entity_classes]

        return entity_clf, entity_spans_pool

    def _classify_relations(self, entity_spans, size_embeddings, relations, rel_masks, h):
        """
        :param entity_spans: [batch_size, entity_size, bert_dim]
        :param size_embeddings: [batch_size, entity_size, size_embedding]
        :param relations: [batch_size, rel_size, 2]
        :param rel_masks: [batch_size, rel_size, sequence_len]
        :param h: [batch_size, rel_size, sequence_len, bert_dim]
        :return:
        """
        batch_size = relations.shape[0]

        # ==========get pairs of entity candidate representations==========
        entity_pairs = batch_index(entity_spans, relations)
        # entity_pairs: [batch_size, rel_size, 2, bert_dim]

        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)
        # entity_pairs: [batch_size, rel_size, 2*bert_dim]

        # ==========get corresponding size embeddings==========
        size_pair_embeddings = batch_index(size_embeddings, relations)
        # size_pair_embeddings: [batch_size, rel_size, 2, embedding_dim]

        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)
        # size_pair_embeddings: [batch_size, rel_size, 2*embedding_dim]

        # ==========relation context (context between entity candidate pair)==========
        m = ((rel_masks == 0).float() * (-1e30)).unsqueeze(-1)
        # m.shape: [batch_size, rel_size, sequence_len, 1]

        rel_ctx = m + h
        # rel_ctx.shape: [batch_size, rel_size, sequence_len, bert_dim]

        rel_ctx = rel_ctx.max(dim=2)[0]
        # rel_ctx.shape: [batch_size, rel_size, bert_dim]

        # 源代码这一步操作是为了将相邻实体的关系向量设置为0，但我测试了一下这个操作对rel_ctx没什么影响，
        # 相邻等状况在得到rel_mask时都已考虑到，目前还是先留着，之后在研究研究
        rel_ctx[rel_masks.to(torch.uint8).any(-1) == 0] = 0
        # rel_ctx.shape: [batch_size, rel_size, bert_dim]

        # ==========concatenate==========
        rel_repr = torch.cat([rel_ctx, entity_pairs, size_pair_embeddings], dim=2)
        rel_repr = self.dropout(rel_repr)

        # ==========classify relation candidates==========
        rel_logits = self.rel_classifier(rel_repr)

        return rel_logits

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_masks, ctx_size):
        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_masks.long()
        # entity_logits_max.shape: [batch_size, entity_size]

        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []

            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)  # 非0元素索引
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()  # 根据索引取出相应实体的头和尾index ([[1,3],```])
            non_zero_indices = non_zero_indices.tolist()

            # 根据目前有可能是实体的所有备选实体构建关系
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        rel_masks.append(create_rel_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # stack
        device = self.rel_classifier.weight.device
        batch_relations = padded_stack(batch_relations).to(device)
        batch_rel_masks = padded_stack(batch_rel_masks).to(device)
        batch_rel_sample_masks = padded_stack(batch_rel_sample_masks).to(device)

        return batch_relations, batch_rel_masks, batch_rel_sample_masks

    def forward(self, *args, inference=False, **kwargs):
        if not inference:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_inference(*args, **kwargs)
