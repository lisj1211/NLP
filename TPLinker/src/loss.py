import torch
from torch import nn


class MyLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, num_classes=3, reduction='mean'):
        """
        :param alpha: 阿尔法α,类别权重.当α是列表时,为各类别权重
        :param gamma: 伽马γ,难易样本调节参数
        :param num_classes: 类别数量
        :param reduction: 损失计算方式,默认取均值
        """
        super().__init__()
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            self.alpha = torch.Tensor([alpha] * num_classes)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, h2t_pred, h2h_pred, t2t_pred, h2t_truth, h2h_truth, t2t_truth):
        entity_loss = self._multilabel_categorical_crossentropy(h2t_pred, h2t_truth)
        relation_loss = self.focal_loss(h2h_pred, h2h_truth) + self.focal_loss(t2t_pred, t2t_truth)
        return entity_loss + relation_loss

    @staticmethod
    def _multilabel_categorical_crossentropy(y_pred, y_true):
        """
        https://kexue.fm/archives/7359
        """
        batch_size, ent_type_size = y_pred.shape[:2]
        y_true = y_true.reshape(batch_size * ent_type_size, -1)
        y_pred = y_pred.reshape(batch_size * ent_type_size, -1)

        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
        y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        return (neg_loss + pos_loss).mean()

    def focal_loss(self, pred, target):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        """
        pred = pred.view(-1, pred.size()[-1])
        target = target.view(-1)
        # 为当前batch内的样本，逐个分配类别权重，shape=(batch_size,), 一维向量
        alpha = self.alpha[target].to(pred.device)
        # 对模型裸输出做softmax再取log, shape=(batch_size, num_class)
        log_softmax = torch.log_softmax(pred, dim=1)
        # 取出每个样本在类别标签位置的log_softmax值, shape=(batch_size, 1)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))
        logpt = logpt.view(-1)
        # 对log_softmax再取负，就是交叉熵了
        ce_loss = -logpt
        # 对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(batch_size)
        pt = torch.exp(logpt)
        floss = alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return floss.mean()
        if self.reduction == "sum":
            return floss.sum()
        return floss
