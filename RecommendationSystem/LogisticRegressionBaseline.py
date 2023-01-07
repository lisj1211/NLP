# -*- coding: utf-8 -*-
"""
基于逻辑回归梯度下降的协同过滤baseline模型
b_u = b_u - lr * (-(r_ui - (avg + b_u + b_i)) + l_2 * b_u)
b_i = b_i - lr * (-(r_ui - (avg + b_u + b_i)) + l_2 * b_i)

error = r_ui - (avg + b_u + b_i)
b_u = b_u + lr * (error - l_2 * b_u)
b_i = b_i + lr * (error - l_2 * b_i)
"""

import pandas as pd
import numpy as np


class BaselineCFbySGD:
    def __init__(self, num_epochs, learning_rate, cof, columns=None):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.cof = cof
        self.columns = ['userId', 'movieId', 'rating'] if not columns else columns

    def fit(self, dataset):
        self.dataset = dataset
        # 用户评分数据
        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]
        # 计算全局平均分
        self.global_mean = self.dataset[self.columns[2]].mean()
        self.bu, self.bi = self.sgd()

    def sgd(self):
        # 初始化bu,bi的值为0
        bu = dict(zip(self.users_ratings.index, np.zeros(len(self.users_ratings))))
        bi = dict(zip(self.items_ratings.index, np.zeros(len(self.items_ratings))))

        for i in range(self.num_epochs):
            print(f'iter {i}:')
            for uid, iid, real_rating in self.dataset.itertuples(index=False):
                error = real_rating - (self.global_mean + bu[uid] + bi[iid])
                bu[uid] += self.learning_rate * (error - self.cof * bu[uid])
                bi[iid] += self.learning_rate * (error - self.cof * bi[iid])

        return bu, bi

    def predict(self, user_id, item_id):
        return self.global_mean + self.bu[user_id] + self.bi[item_id]


def data_split(data_path, train_size=0.8):
    

if __name__ == '__main__':
    dtype = [('userId', np.int32), ('movieId', np.int32), ('rating', np.float32)]
    dataset = pd.read_csv('./data/ml-latest-small/ratings.csv', usecols=range(3), dtype=dict(dtype))
    model = BaselineCFbySGD(20, 0.1, 0.1, columns=['userId', 'movieId', 'rating'])
    model.fit(dataset)

    while True:
        uid = int(input('uid:'))
        iid = int(input('iid:'))
        print(model.predict(uid, iid))
