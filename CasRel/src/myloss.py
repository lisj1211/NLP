import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLoss(nn.Module):
    """
    CasRel model loss function: subject loss + object and relation loss
    """
    def __init__(self):
        super(MyLoss, self).__init__()

    def _loss_fn(self, pred, gold, mask):
        pred = pred.squeeze(-1)
        loss = F.binary_cross_entropy(pred, gold, reduction='none')
        if loss.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        loss = torch.sum(loss * mask) / torch.sum(mask)

        return loss

    def forward(self, predict, target):
        mask = target['mask']

        return self._loss_fn(predict['sub_heads'], target['sub_heads'], mask) + \
               self._loss_fn(predict['sub_tails'], target['sub_tails'], mask) + \
               self._loss_fn(predict['obj_heads'], target['obj_heads'], mask) + \
               self._loss_fn(predict['obj_tails'], target['obj_tails'], mask)
