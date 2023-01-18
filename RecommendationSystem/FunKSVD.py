# -*- coding: utf-8 -*-
"""
基于FunkSVD矩阵分解的LFM(Latent Factor Model)
"""

import pandas as pd
import numpy as np


class LFM:
    """FunkSVD based"""
    def __init__(self, lr, lambda_p, lambda_q, num_factor=10, num_epochs=10, columns=None):
        self.lr = lr  # 学习率
        self.lambda_p = lambda_p  # P矩阵正则系数
        self.lambda_q = lambda_q  # Q矩阵正则系数
        self.num_factor = num_factor  # Latent Factor个数
        self.num_epochs = num_epochs  # 迭代次数
        self.columns = ['userId', 'movieId', 'rating'] if not columns else columns