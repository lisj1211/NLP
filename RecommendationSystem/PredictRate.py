# -*- coding: utf-8 -*-
import os
from pprint import pprint

import pandas as pd
import numpy as np

DATA_PATH = './data/ml-latest-small/ratings.csv'
CACHE_DIR = './data/cache'


def load_data(data_path):
    """
    加载数据
    :param data_path: 数据集路径
    :return: 用户物品评分矩阵
    """
    cache_dir = os.path.join(CACHE_DIR, 'ratings_matrix.cache')
    os.makedirs(CACHE_DIR, exist_ok=True)

    print('开始加载数据集...')
    if os.path.exists(cache_dir):
        print('加载缓存中...')
        ratings_matrix = pd.read_pickle(cache_dir)
        print('加载缓存数据完毕')
    else:
        print('加载新数据集中...')
        dtype = {'userId': np.int32, 'movieId': np.int32, 'rating': np.float32}
        ratings = pd.read_csv(data_path, dtype=dtype, usecols=range(3))
        ratings_matrix = ratings.pivot_table(index=['userId'], columns=['movieId'], values='rating')
        ratings_matrix.to_pickle(cache_dir)
        print('数据集加载完毕')

    return ratings_matrix


def compute_pearson_similarity(ratings_matrix, based='user'):
    """
    计算皮尔逊相关系数
    :param ratings_matrix:
    :param based:
    :return: 相似度矩阵
    """
    user_similarity_cache_path = os.path.join(CACHE_DIR, 'user_similarity.cache')
    item_similarity_cache_path = os.path.join(CACHE_DIR, 'item_similarity.cache')
    if based == 'user':
        if os.path.exists(user_similarity_cache_path):
            print('正在从缓存中加载用户相似度矩阵')
            similarity_matrix = pd.read_pickle(user_similarity_cache_path)
        else:
            print('开始计算用户相似度矩阵')
            similarity_matrix = ratings_matrix.T.corr()
            similarity_matrix.to_pickle(user_similarity_cache_path)
    elif based == 'item':
        if os.path.exists(item_similarity_cache_path):
            print('正在从缓存中加载物品相似度矩阵')
            similarity_matrix = pd.read_pickle(item_similarity_cache_path)
        else:
            print('开始计算物品相似度矩阵')
            similarity_matrix = ratings_matrix.corr()
            similarity_matrix.to_pickle(item_similarity_cache_path)
    else:
        raise ValueError(f"Unhandled 'based' Value: f{based}")
    print('相似度矩阵计算/加载完毕')

    return similarity_matrix


def predict(uid, iid, ratings_matrix, user_similar):
    """
    预测给定用户对给定物品的评分值
    :param uid:
    :param iid:
    :param ratings_matrix:
    :param user_similar:
    :return:
    """
    print(f'开始预测用户{uid}对电影{iid}的评分...')
    # 找出与uid相似的用户
    similar_users = user_similar[uid].drop([uid]).dropna()
    similar_users = similar_users.where(similar_users > 0).dropna()
    if similar_users.empty:
        raise ValueError(f'用户{uid}没有相似的用户')

    # 在与uid相似的用户中筛选出对iid物品有评分的用户
    ids = set(ratings_matrix[iid].dropna().index) & set(similar_users.index)
    final_similar_users = similar_users[list(ids)]

    # 利用uid的相似用户预测uid对iid物品的评分
    sum_up = 0
    sum_down = 0
    for sim_uid, similarity in final_similar_users.iteritems():
        sim_user_rated_movies = ratings_matrix.loc[sim_uid].dropna()
        sim_user_rating_for_item = sim_user_rated_movies[iid]
        sum_up += similarity * sim_user_rating_for_item
        sum_down += similarity

    predict_rating = sum_up / sum_down
    print(f'用户{uid}对物品{iid}的预测评分为: {predict_rating: .2f}')
    return round(predict_rating, 2)


def _predict_all(uid, item_ids, ratings_matrix, user_similar):
    """
    预测全部评分
    :param uid:
    :param ratings_matrix:
    :param user_similar:
    :return:
    """
    for iid in item_ids:
        try:
            rating = predict(uid, iid, ratings_matrix, user_similar)
        except ValueError as e:
            print(e)
        else:
            yield uid, iid, rating


def predict_all(uid, ratings_matrix, user_similar, filter_rule=None):
    """
    预测全部评分，并可根据条件进行前置过滤
    :param uid:
    :param ratings_matrix:
    :param user_similar:
    :param filter_rule: 过滤规则，四选一：'unhot', 'rated', ['unhot', 'rated'], None
    :return:
    """
    if not filter_rule:
        item_ids = ratings_matrix.columns
    elif isinstance(filter_rule, str) and filter_rule == 'unhot':
        '''过滤非热门电影'''
        count = ratings_matrix.count()
        item_ids = count.where(count > 10).dropna().index
    elif isinstance(filter_rule, str) and filter_rule == 'rated':
        '''过滤用户评分过的电影'''
        user_ratings = ratings_matrix.loc[uid]
        # 评分范围是1-5，所以用6过滤
        _ = user_ratings < 6
        item_ids = _.where(_ == False).dropna().index
    elif isinstance(filter_rule, list) and set(filter_rule) == {'unhot', 'rated'}:
        count = ratings_matrix.count()
        ids1 = count.where(count > 10).dropna().index

        user_ratings = ratings_matrix.loc[uid]
        _ = user_ratings < 6
        ids2 = _.where(_==False).dropna().index
        item_ids = set(ids1) & set(ids2)
    else:
        raise ValueError('Invalid filter_rule')

    yield from _predict_all(uid, item_ids, ratings_matrix, user_similar)


def topk_rs_result(k):
    """
    topK 推荐
    :param k:
    :return:
    """
    ratings_matrix = load_data(DATA_PATH)
    user_similar = compute_pearson_similarity(ratings_matrix, based='user')
    results = predict_all(1, ratings_matrix, user_similar, filter_rule=['unhot', 'rated'])

    return sorted(results, key=lambda x: x[2], reverse=True)[:k]


if __name__ == '__main__':
    ratings_matrix = load_data(DATA_PATH)
    user_similar = compute_pearson_similarity(ratings_matrix, based='user')
    for result in predict_all(1, ratings_matrix, user_similar, filter_rule=['unhot', 'rated']):
        print(result)

    result = topk_rs_result(20)
    pprint(result)
