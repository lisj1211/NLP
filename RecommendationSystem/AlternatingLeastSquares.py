# -*- coding: utf-8 -*-
"""
基于交替最小二乘法的协同过滤baseline模型
J = sum((r_ui - avg - b_u - b_i) ** 2) + lambda * (sum(b_i ** 2) + sum(b_u ** 2))
J 对 b_u 的偏导：-2 * sum(r_ui - avg - b_u - b_i) + 2 * lambda * b_u
J 对 b_i 的偏导：-2 * sum(r_ui - avg - b_u - b_i) + 2 * lambda * b_i

令偏导为0
sum(r_ui - avg - b_u - b_i) = lambda * b_u
sum(r_ui - avg - b_i) = sum(b_u) + lambda * b_u
设 sum(b_u) = |R(u)| * b_u ， |R(u)表示用户u的有过评分数量
b_u = sum(r_ui - avg - b_i) / (lambda + R(u))
同理：
b_i = sum(r_ui - avg - b_u) / (lambda + R(i))
"""

import pandas as pd
import numpy as np


class BaselineCFbyALS:
    def __init__(self, num_epochs, lambda_bu, lambda_bi, columns=None):
        self.num_epochs = num_epochs  # 迭代次数
        self.lambda_bu = lambda_bu    # bu的正则参数
        self.lambda_bi = lambda_bi  # bi的正则参数
        self.columns = ['userId', 'movieId', 'rating'] if not columns else columns

    def fit(self, dataset):
        self.dataset = dataset
        # 用户评分数据
        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        # 物品评分数据
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]
        # 计算全局平均分
        self.global_mean = self.dataset[self.columns[2]].mean()
        self.bu, self.bi = self.als()

    def als(self):
        # 初始化bu,bi的值为0
        bu = dict(zip(self.users_ratings.index, np.zeros(len(self.users_ratings))))
        bi = dict(zip(self.items_ratings.index, np.zeros(len(self.items_ratings))))

        for i in range(self.num_epochs):
            print(f'iter {i}:')
            for iid, uids, ratings in self.items_ratings.itertuples(index=True):
                sum_ = 0
                for uid, rating in zip(uids, ratings):
                    sum_ += rating - (self.global_mean + bu[uid])
                bi[iid] = sum_ / (self.lambda_bi + len(uids))

            for uid, iids, ratings in self.users_ratings.itertuples(index=True):
                sum_ = 0
                for iid, rating in zip(iids, ratings):
                    sum_ += rating - (self.global_mean + bi[iid])
                bu[uid] = sum_ / (self.lambda_bu + len(iids))

        return bu, bi

    def predict(self, user_id, item_id):
        if item_id not in self.items_ratings.index:
            raise ValueError('当期itemId不存在于训练数据中，无法进行预测')
        return self.global_mean + self.bu[user_id] + self.bi[item_id]

    def test(self, test_data):
        results = []
        for uid, iid, real_rating in test_data.itertuples(index=False):
            try:
                predict_result = self.predict(uid, iid)
            except ValueError as e:
                print(e)
            else:
                results.append([uid, iid, real_rating, predict_result])
        return results


def data_split(data_path, train_size=0.8):
    print('开始划分数据集')
    dtype = {'userId': np.int32, 'movieId': np.int32, 'ratings': np.float32}
    ratings = pd.read_csv(data_path, dtype=dtype, usecols=range(3))
    test_data_idx = []
    for uid in ratings.groupby('userId').any().index:
        user_rating_data = ratings.where(ratings['userId'] == uid).dropna()
        index = list(user_rating_data.index)
        np.random.shuffle(index)
        _index = round(len(user_rating_data) * train_size)
        test_data_idx.extend(list(user_rating_data.index)[_index:])

    test_data = ratings.loc[test_data_idx]
    train_data = ratings.drop(test_data_idx)
    print('数据集划分完成')
    return train_data, test_data


def accuracy(predict_result):
    """
    :param predict_result: (uid, iid, real_rating, predict_rating)
    :return:
    """
    sum_ = 0
    for uid, iid, real_rating, predict_rating in predict_result:
        sum_ += abs(predict_rating - real_rating)
    return round(sum_ / len(predict_result))


if __name__ == '__main__':
    dtype = {'userId': np.int32, 'movieId': np.int32, 'rating': np.float32}
    dataset = pd.read_csv('./data/ml-latest-small/ratings.csv')
    model = BaselineCFbyALS(20, 0.1, 0.1, columns=['userId', 'movieId', 'rating'])
    model.fit(dataset)
    print(model.predict(1, 1))
