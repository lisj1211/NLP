from collections import namedtuple

import numpy as np
import torch


class TagMapping:
    """Convert raw label to index"""
    def __init__(self, index2relation):
        # relation mapping
        self.id2relation = {int(idx): rel for idx, rel in index2relation.items()}
        self.relation2id = {rel: idx for idx, rel in self.id2relation.items()}

        # tag2id mapping: entity head to entity tail
        self.head2tail_tag2id = {
            "O": 0,
            "ENT-H2T": 1,  # entity head to entity tail
        }
        self.head2tail_id2tag = {v: k for k, v in self.head2tail_tag2id.items()}

        # tag2id mapping: entity1 head to entity2 head
        self.head2head_tag2id = {
            "O": 0,
            "REL-SH2OH": 1,  # subject head to object head
            "REL-OH2SH": 2,  # object head to subject head
        }
        self.head2head_id2tag = {v: k for k, v in self.head2head_tag2id.items()}

        # tag2id mapping: entity1 tail to entity2 tail
        self.tail2tail_tag2id = {
            "O": 0,
            "REL-ST2OT": 1,  # subject tail to object tail
            "REL-OT2ST": 2,  # object tail to subject tail
        }
        self.tail2tail_id2tag = {v: k for k, v in self.tail2tail_tag2id.items()}

    def relation_id(self, key: str):
        return self.relation2id.get(key, None)

    def relation_tag(self, index: int):
        return self.id2relation.get(index, None)

    def h2t_id(self, key: str):
        return self.head2tail_tag2id.get(key, None)

    def h2t_tag(self, index: int):
        return self.head2tail_id2tag.get(index, 'O')

    def h2h_id(self, key: str):
        return self.head2head_tag2id.get(key, None)

    def h2h_tag(self, index: int):
        return self.head2head_id2tag.get(index, 'O')

    def t2t_id(self, key: str):
        return self.tail2tail_tag2id.get(key, None)

    def t2t_tag(self, index: int):
        return self.tail2tail_id2tag.get(index, 'O')


# p -> point to head of entity, q -> point to tail of entity, tagid -> id of h2t tag
Head2TailItem = namedtuple('Head2TailItem', ['p', 'q', 'tagid'])
# p -> point to head of entity1, q -> point to head of entity2
# relid -> relation_id between entity1 and entity2, tagid -> id of h2h tag
Head2HeadItem = namedtuple('Head2HeadItem', ['relid', 'p', 'q', 'tagid'])
# p -> point to tail of entity1, q -> point to tail of entity2
# relid -> relation_id between entity1 and entity2, tagid -> id of h2h tag
Tail2TailItem = namedtuple('Tail2TailItem', ['relid', 'p', 'q', 'tagid'])


class HandshakingTaggingEncoder:
    """Generate head2tail, head2head, tail2tail metrix"""
    def __init__(self, tag_mapping: TagMapping):
        self.tag_mapping = tag_mapping

    def encode(self, sample, max_sequence_length):
        index_matrix = self._build_index_matrix(max_sequence_length=max_sequence_length)
        flatten_length = max_sequence_length * (max_sequence_length + 1) // 2

        h2t_spots, h2h_spots, t2t_spots = self._collect_spots(sample)
        
        h2t_tagging = self._encode_head2tail(h2t_spots, index_matrix=index_matrix, sequence_length=flatten_length)
        h2h_tagging = self._encode_head2head(h2h_spots, index_matrix=index_matrix, sequence_length=flatten_length)
        t2t_tagging = self._encode_tail2tail(t2t_spots, index_matrix=index_matrix, sequence_length=flatten_length)
        return h2t_tagging, h2h_tagging, t2t_tagging

    def _collect_spots(self, sample):
        h2t_spots, h2h_spots, t2t_spots = [], [], []
        for relation in sample['relation_list']:
            subject_span = relation['subj_tok_span']
            object_span = relation['obj_tok_span']

            # add head-to-tail spot
            h2t_i0 = Head2TailItem(p=subject_span[0], q=subject_span[1] - 1, tagid=self.tag_mapping.h2t_id('ENT-H2T'))
            h2t_spots.append(h2t_i0)
            h2t_i1 = Head2TailItem(p=object_span[0], q=object_span[1] - 1, tagid=self.tag_mapping.h2t_id('ENT-H2T'))
            h2t_spots.append(h2t_i1)

            # convert relation to id
            rel_id = self.tag_mapping.relation_id(relation['predicate'])
            # add head-to-head spot
            p = subject_span[0] if subject_span[0] <= object_span[0] else object_span[0]
            q = object_span[0] if subject_span[0] <= object_span[0] else subject_span[0]
            k = 'REL-SH2OH' if subject_span[0] <= object_span[0] else 'REL-OH2SH'
            h2h_item = Head2HeadItem(relid=rel_id, p=p, q=q, tagid=self.tag_mapping.h2h_id(k))
            h2h_spots.append(h2h_item)

            # add tail-to-tail spot
            p = subject_span[1] - 1 if subject_span[1] <= object_span[1] else object_span[1] - 1
            q = object_span[1] - 1 if subject_span[1] <= object_span[1] else subject_span[1] - 1
            k = 'REL-ST2OT' if subject_span[1] <= object_span[1] else 'REL-OT2ST'
            t2t_item = Tail2TailItem(relid=rel_id, p=p, q=q, tagid=self.tag_mapping.t2t_id(k))
            t2t_spots.append(t2t_item)

        return h2t_spots, h2h_spots, t2t_spots

    @staticmethod
    def _encode_head2tail(h2t_spots, index_matrix, sequence_length):
        # shape (sequence_length,)
        tagging_sequence = np.zeros(sequence_length, dtype=np.int32)
        for item in h2t_spots:
            index = index_matrix[item.p][item.q]
            tagging_sequence[index] = item.tagid
        return tagging_sequence

    def _encode_head2head(self, h2h_spots, index_matrix, sequence_length):
        num_relations = len(self.tag_mapping.relation2id)
        # shape (num_relations, sequence_length)
        tagging_sequence = np.zeros([num_relations, sequence_length], dtype=np.int32)
        for item in h2h_spots:
            index = index_matrix[item.p][item.q]
            tagging_sequence[item.relid][index] = item.tagid
        return tagging_sequence

    def _encode_tail2tail(self, t2t_spots, index_matrix, sequence_length):
        num_relations = len(self.tag_mapping.relation2id)
        # shape (batch_size, num_relations, sequence_length)
        tagging_sequence = np.zeros([num_relations, sequence_length], dtype=np.int32)
        for item in t2t_spots:
            index = index_matrix[item.p][item.q]
            tagging_sequence[item.relid][index] = item.tagid
        return tagging_sequence

    @staticmethod
    def _build_index_matrix(max_sequence_length):
        # e.g [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        pairs = [(i, j) for i in range(max_sequence_length) for j in list(range(max_sequence_length))[i:]]
        # shape: (max_sequence_length, max_sequence_length)
        matrix = [[0 for _ in range(max_sequence_length)] for _ in range(max_sequence_length)]
        for index, values in enumerate(pairs):
            matrix[values[0]][values[1]] = index
        return matrix


class HandshakingTaggingDecoder:

    def __init__(self, tag_mapping: TagMapping):
        super().__init__()
        self.tag_mapping = tag_mapping

    def decode(self, samples, batch_h2t_pred, batch_h2h_pred, batch_t2t_pred, max_sequence_length, do_split=False):
        batch_relations = []
        for sample, h2t_pred, h2h_pred, t2t_pred in zip(samples, batch_h2t_pred, batch_h2h_pred, batch_t2t_pred):
            token_offset = sample['token_offset'] if do_split else 0
            char_offset = sample['char_offset'] if do_split else 0

            # e.g [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
            index_matrix = [(i, j) for i in range(max_sequence_length) for j in list(range(max_sequence_length))[i:]]
            # decode predictions
            h2t_spots = self._decode_head2tail(h2t_pred, index_matrix)
            h2h_spots = self._decode_head2head(h2h_pred, index_matrix)
            t2t_spots = self._decode_tail2tail(t2t_pred, index_matrix)

            entities_head_map = self._parse_entities(h2t_spots, sample)
            relation_tails = self._parse_tails(t2t_spots)
            relations = self._parse_relations(h2h_spots, entities_head_map, relation_tails, token_offset, char_offset)
            batch_relations.append(relations)
        return batch_relations

    @staticmethod
    def _decode_head2tail(h2t_pred, index_matrix):
        """Decode head2tail tagging.

        Args:
            h2t_pred: Tensor, shape (1+2+...+seq_len, 2)
            index_matrix: List of indexes

        Returns:
            items: List of Head2TailItem
        """
        items = []
        # shape: (1+2+...+seq_len)
        h2t_pred = torch.argmax(h2t_pred, dim=-1)
        for index in torch.nonzero(h2t_pred):
            flat_index = index[0].item()
            matrix_ind = index_matrix[flat_index]
            item = Head2TailItem(p=matrix_ind[0], q=matrix_ind[1], tagid=h2t_pred[flat_index].item())
            items.append(item)
        return items

    def _parse_entities(self, h2t_items, sample):
        entities_head_map = {}
        for item in h2t_items:
            if item.tagid != self.tag_mapping.h2t_id('ENT-H2T'):
                continue
            char_offset_list = sample['offset_mapping'][item.p: item.q + 1]
            if not char_offset_list:  # item.p, item.q 为padding部分则忽略
                continue
            char_span = [char_offset_list[0][0], char_offset_list[-1][1]]
            entity_txt = sample['text'][char_span[0]: char_span[1]]
            head = item.p
            if head not in entities_head_map:
                entities_head_map[head] = []
            entities_head_map[head].append({
                'text': entity_txt,
                'tok_span': [item.p, item.q],
                'char_span': char_span,
            })
        return entities_head_map

    @staticmethod
    def _decode_head2head(h2h_pred, index_matrix):
        """Decode head2head predictions.

        Args:
            h2h_pred: Tensor, shape (num_relations, 1+2+...+seq_len, 3)
            index_matrix: List of indexes

        Returns:
            items: List of Head2HeadItem
        """
        items = []
        # shape: (num_relations, 1+2+...+seq_len)
        h2h_pred = torch.argmax(h2h_pred, dim=-1)
        for index in torch.nonzero(h2h_pred):
            relation_id, flat_index = index[0].item(), index[1].item()
            matrix_index = index_matrix[flat_index]
            item = Head2HeadItem(
                relid=relation_id, p=matrix_index[0], q=matrix_index[1],
                tagid=h2h_pred[relation_id][flat_index].item())
            items.append(item)
        return items

    def _parse_relations(self, h2h_items, entities_head_map, relation_tails, token_offset, char_offset):
        relations = []
        for item in h2h_items:
            subj_head, obj_head = None, None
            if item.tagid == self.tag_mapping.h2h_id('REL-SH2OH'):
                subj_head, obj_head = item.p, item.q
            elif item.tagid == self.tag_mapping.h2h_id('REL-OH2SH'):
                subj_head, obj_head = item.q, item.p
            if subj_head is None or obj_head is None:
                continue
            if subj_head not in entities_head_map or obj_head not in entities_head_map:
                continue

            subj_list = entities_head_map[subj_head]
            obj_list = entities_head_map[obj_head]
            for subj in subj_list:
                for obj in obj_list:
                    tail = f"{item.relid}-{subj['tok_span'][1]}-{obj['tok_span'][1]}"
                    if tail not in relation_tails:
                        continue
                    relations.append({
                        'subject': subj['text'],
                        'object': obj['text'],
                        'subj_tok_span': [subj['tok_span'][0] + token_offset, subj['tok_span'][1] + token_offset + 1],
                        'subj_char_span': [subj['char_span'][0] + char_offset, subj['char_span'][1] + char_offset + 1],
                        'obj_tok_span': [obj['tok_span'][0] + token_offset, obj['tok_span'][1] + token_offset + 1],
                        'obj_char_span': [obj['char_span'][0] + char_offset, obj['char_span'][1] + char_offset + 1],
                        'predicate': self.tag_mapping.relation_tag(item.relid),
                    })
        return relations

    @staticmethod
    def _decode_tail2tail(t2t_pred, index_matrix):
        """Decode tail2tail predictions.

        Args:
            t2t_pred: Tensor, shape (num_relations, 1+2+...+seq_len, 3)
            index_matrix: List of indexes

        Returns:
            items: List of Tail2TailItem
        """
        items = []
        t2t_pred = torch.argmax(t2t_pred, dim=-1)
        for index in torch.nonzero(t2t_pred):
            relation_id, flat_index = index[0].item(), index[1].item()
            matrix_index = index_matrix[flat_index]
            item = Tail2TailItem(
                relid=relation_id, p=matrix_index[0], q=matrix_index[1],
                tagid=t2t_pred[relation_id][flat_index].item())
            items.append(item)
        return items

    def _parse_tails(self, t2t_items):
        tails = set()
        for item in t2t_items:
            if item.tagid == self.tag_mapping.t2t_id('REL-ST2OT'):
                tails.add(f'{item.relid}-{item.p}-{item.q}')
            elif item.tagid == self.tag_mapping.t2t_id('REL-OT2ST'):
                tails.add(f'{item.relid}-{item.q}-{item.p}')
        return tails
