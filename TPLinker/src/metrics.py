import json
from itertools import chain

import torch

from src.tagging_scheme import TagMapping, HandshakingTaggingDecoder


class MetricsCalculator:
    def __init__(self, rel_path, max_sequence_length):
        with open(rel_path, 'r', encoding='utf-8') as f:
            idx2rel = json.load(f)
        self.tag_mapping = TagMapping(idx2rel)
        self.decoder = HandshakingTaggingDecoder(self.tag_mapping)
        self.max_sequence_length = max_sequence_length

        self.h2t_acc = 0
        self.h2h_acc = 0
        self.t2t_acc = 0

        self.correct_num = 0
        self.predict_num = 0
        self.gold_num = 0

    @staticmethod
    def get_sample_accuracy(predict, truth):
        """
        计算所有抽取字段都正确的样本比例,即该batch的输出与truth全等的样本比例
        :param predict: (batch_size, rel_nums or None, sequence_len, tag_nums)
        :param truth: (batch_size, rel_nums or None, sequence_len)
        :return:
        """
        predict_idx = torch.argmax(predict, dim=-1)
        predict_idx = predict_idx.view(predict_idx.size()[0], -1)
        truth = truth.view(truth.size()[0], -1)

        correct_tag_num = torch.sum(torch.eq(truth, predict_idx).float(), dim=1)

        # seq维上所有tag必须正确，所以correct_tag_num必须等于seq的长度才算一个correct的sample
        sample_acc_ = torch.eq(correct_tag_num, torch.ones_like(correct_tag_num) * truth.size()[-1]).float()
        sample_acc = torch.mean(sample_acc_)
        return sample_acc

    def get_rel_count(self, batch_data, batch_h2t_pred, batch_h2h_pred, batch_t2t_pred, pattern='whole_text'):
        """获得当前batch中关系三元组的混淆矩阵指标"""
        pred_relations = self.decoder.decode(batch_data, batch_h2t_pred, batch_h2h_pred, batch_t2t_pred,
                                             max_sequence_length=self.max_sequence_length)
        gold_relations = [s['relation_list'] for s in batch_data]

        if pattern == 'whole_span':
            gold_set = set([
                f"{rel['subj_tok_span'][0]}-{rel['subj_tok_span'][1]}-{rel['predicate']}-{rel['obj_tok_span'][0]}-{rel['obj_tok_span'][1]}"
                for rel in chain(*gold_relations)])
            pred_set = set([
                f"{rel['subj_tok_span'][0]}-{rel['subj_tok_span'][1]}-{rel['predicate']}-{rel['obj_tok_span'][0]}-{rel['obj_tok_span'][1]}"
                for rel in chain(*pred_relations)])
        elif pattern == 'whole_text':
            gold_set = set([
                f"{rel['subject']}-{rel['predicate']}-{rel['object']}" for rel in chain(*gold_relations)
            ])
            pred_set = set([
                f"{rel['subject']}-{rel['predicate']}-{rel['object']}" for rel in chain(*pred_relations)
            ])
        elif pattern == 'only_head_index':
            gold_set = set([
                f"{rel['subj_tok_span'][0]}-{rel['predicate']}-{rel['obj_tok_span'][0]}" for rel in chain(*gold_relations)
            ])
            pred_set = set([
                f"{rel['subj_tok_span'][0]}-{rel['predicate']}-{rel['obj_tok_span'][0]}" for rel in chain(*pred_relations)
            ])
        else:
            raise ValueError('Invalid Argument')

        correct_num = len(pred_set & gold_set)
        predict_num = len(pred_set)
        gold_num = len(gold_set)

        return correct_num, predict_num, gold_num

    def update(self, head2tail_acc, head2head_acc, tail2tail_acc, correct_rel_num, predict_rel_num, gold_rel_num):
        """更新所有指标"""
        self.h2t_acc += head2tail_acc
        self.h2h_acc += head2head_acc
        self.t2t_acc += tail2tail_acc

        self.correct_num += correct_rel_num
        self.predict_num += predict_rel_num
        self.gold_num += gold_rel_num

    def get_entity_acc(self, data_loader_length):
        """获得head2tail, head2head, tail2tail的准确率"""
        return self.h2t_acc / data_loader_length, self.h2h_acc / data_loader_length, self.t2t_acc / data_loader_length

    def get_relation_prf_scores(self):
        """获得关系三元组的p, r, f"""
        epsilon = 1e-10
        precision = self.correct_num / (self.predict_num + epsilon)
        recall = self.correct_num / (self.gold_num + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        return precision, recall, f1

    def reset(self):
        """每轮epoch重置"""
        self.h2t_acc = 0
        self.h2h_acc = 0
        self.t2t_acc = 0

        self.correct_num = 0
        self.predict_num = 0
        self.gold_num = 0
