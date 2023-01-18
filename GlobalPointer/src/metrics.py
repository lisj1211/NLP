import numpy as np


class MetricsCalculator:
    def __init__(self):
        self.correct_num = 0
        self.predict_num = 0
        self.gold_num = 0

    @staticmethod
    def get_pred_count(y_pred, y_true):
        """获得当前batch中关系三元组的混淆矩阵指标"""
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred_set = set()
        gold_set = set()
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred_set.add((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            gold_set.add((b, l, start, end))

        correct_num = len(pred_set & gold_set)
        predict_num = len(pred_set)
        gold_num = len(gold_set)

        return correct_num, predict_num, gold_num

    def update(self, correct_ent_num, predict_ent_num, gold_ent_num):
        """更新所有指标"""
        self.correct_num += correct_ent_num
        self.predict_num += predict_ent_num
        self.gold_num += gold_ent_num

    def get_entity_prf_scores(self):
        """获得实体的p, r, f"""
        epsilon = 1e-10
        precision = self.correct_num / (self.predict_num + epsilon)
        recall = self.correct_num / (self.gold_num + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        return precision, recall, f1

    def reset(self):
        """每轮epoch重置"""
        self.correct_num = 0
        self.predict_num = 0
        self.gold_num = 0
