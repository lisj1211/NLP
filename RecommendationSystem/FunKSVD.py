# -*- coding: utf-8 -*-
"""
基于FunkSVD矩阵分解的LFM(Latent Factor Model)
rating_matrix = P * Q
shape: (num_user, num_item) = (num_user, latent_dim) * (latent_dim * num_item)
最小化 sum(r_ui - q_i * p_u) ** 2 + lambda(|q_i| ** 2 + |p_u| ** 2)
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
        self.columns = ['userId', 'movieId', 'rating'] if columns is None else columns

    def fit(self, dataset):
        self.dataset = pd.DataFrame(dataset)
        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]

        self.global_mean = self.dataset[self.columns[2]].mean()
        self.P, self.Q = self.sgd()

    def _init_matrix(self):
        """init P and Q matrix"""
        # user-LF
        P = dict(zip(self.users_ratings.index, np.random.rand(len(self.users_ratings), self.num_factor).astype(np.float)))
        # item-LF
        Q = dict(zip(self.items_ratings.index, np.random.rand(len(self.items_ratings), self.num_factor).astype(np.float)))

        return P, Q

    def sgd(self):
        P, Q = self._init_matrix()
        for i in range(self.num_epochs):
            print(f'iter: {i}')
            error_list = []
            for uid, iid, r_ui in self.dataset.itertuples(index=False):
                p_u = P[uid]
                q_i = Q[uid]
                err = np.float(r_ui - np.dot(p_u, q_i))
                p_u += self.lr * (err * q_i - self.lambda_p * p_u)
                q_i += self.lr * (err * p_u - self.lambda_q * q_i)

                P[uid] = p_u
                Q[iid] = q_i

                error_list.append(err ** 2)
            print(np.sqrt(np.mean(error_list)))
        return P, Q

    def predict(self, uid, iid):
        if uid not in self.users_ratings or iid not in self.items_ratings:
            return self.global_mean

        p_u = self.P[uid]
        q_i = self.Q[iid]
        return np.dot(p_u, q_i)
