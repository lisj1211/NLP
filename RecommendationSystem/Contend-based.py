# -*- coding: utf-8 -*-
from pprint import pprint
from functools import reduce
import collections

import pandas as pd
import numpy as np
from gensim.models import TfidfModel
from gensim.corpora import Dictionary


# 建立物品画像
# 利用tags.csv中每部电影的标签作为电影的候选关键词
# 利用TF-IDF计算每部电影的标签TF-IDF值，选取TOP-N个关键词作为电影画像标签
# 并将电影的分类词直接作为每部电影的画像标签
def get_movie_dataset():
    _tags = pd.read_csv('data/ml-latest-small/tags.csv', usecols=range(1, 3)).dropna()
    tags = _tags.groupby('movieId').agg(list)

    movies = pd.read_csv('data/ml-latest-small/movies.csv', index_col='movieId')
    movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))

    movie_index = set(movies.index) & set(tags.index)
    new_tags = tags.loc[list(movie_index)]
    ret = movies.join(new_tags)

    # 构建电影数据集，包含ID、名称、类别、标签四个字段
    movie_dataset = pd.DataFrame(
        map(lambda x: (x[0], x[1], x[2], x[2] + x[3]) if x[3] is not np.nan else (x[0], x[1], x[2], []),
            ret.itertuples()),
        columns=['movieId', 'title', 'genres', 'tags']
    )
    movie_dataset.set_index('movieId', inplace=True)
    return movie_dataset


def create_movie_profile(movie_dataset):
    """
    使用tf-idf提取TOP-N关键词
    :param movie_dataset:
    :return:
    """
    dataset = movie_dataset['tags'].values
    dic = Dictionary(dataset)
    corpus = [dic.doc2bow(line) for line in dataset]
    model = TfidfModel(corpus)

    movie_profile = []
    for i, data in enumerate(movie_dataset.itertuples()):
        mid = data[0]
        title = data[1]
        genres = data[2]
        vector = model[corpus[i]]  # 获取TFIDF向量
        movie_tags = sorted(vector, key=lambda x: x[1], reverse=True)[:30]  # 获得前n个关键词
        topN_tags_weights = {dic[word_idx]: tfidf_value for word_idx, tfidf_value in movie_tags}

        # 将类别词添加进去，并设置权重为1.0
        for g in genres:
            topN_tags_weights[g] = 1.0
        topN_tags = [i[0] for i in topN_tags_weights.items()]
        movie_profile.append((mid, title, topN_tags, topN_tags_weights))

    movie_profile = pd.DataFrame(movie_profile, columns=['movieId', 'title', 'profile', 'weights'])
    movie_profile.set_index('movieId', inplace=True)
    return movie_profile


def create_inverse_table(movie_profile):
    """通过标签找对应的电影"""
    inverse_table = collections.defaultdict(list)  # key: 电影标签， value: [(movieId, weight)]
    for mid, weights in movie_profile['weights'].iteritems():
        for tag, weight in weights.items():
            inverse_table[tag].append((mid, weight))
    return inverse_table


# 用户画像的建立
# 根据用户的评分历史，结合物品画像，将有观影记录的电影的画像标签作为初始标签反打到用户身上
# 通过对用户观影标签的此时进行统计，计算用户的每个初始标签的权重值，排序后选取TOP-N作为用户最终的画像标签
def create_user_profile():
    watch_record = pd.read_csv('data/ml-latest-small/ratings.csv', usecols=range(2),
                               dtype={'userId': np.int, 'movieId': np.int})
    watch_record = watch_record.groupby('userId').agg(list)

    movie_dataset = get_movie_dataset()
    movie_profile = create_movie_profile(movie_dataset)

    user_profile = {}
    for uid, mids in watch_record.itertuples():
        record_movie_profile = movie_profile.loc[list(mids)]  # 用户所有评分电影的关键词
        counter = collections.Counter(reduce(lambda x, y: list(x) + list(y), record_movie_profile['profile'].values))
        interest_words = counter.most_common(50)  # 用户的兴趣词
        max_count = interest_words[0][1]
        interest_words = [(w, round(c / max_count, 4)) for w, c in interest_words]  # 归一化获得比例
        user_profile[uid] = interest_words

    return user_profile


if __name__ == '__main__':
    dataset = get_movie_dataset()
    movie_profile = create_movie_profile(dataset)
    inverse_table = create_inverse_table(movie_profile)
    user_profile = create_user_profile()
    pprint(movie_profile)
    # 产生推荐结果
    for uid, interest_words in user_profile.items():  # 遍历每一个用户
        result_table = collections.defaultdict(list)  # 针对该用户的推荐结果
        for interest_word, interest_weight in interest_words:  # 遍历该用户的兴趣词
            related_movies = inverse_table[interest_word]  # 获得与当前兴趣词相关的所有电影
            for mid, related_weight in related_movies:  # 遍历所有相关电影
                result_table[mid].append(interest_weight)  # 只考虑用户的兴趣程度
                # result_table[mid].append(related_weight)  # 只考虑兴趣词与电影的兴趣程度
                # result_table[mid].append(related_weight * interest_weight)  # 二者都考虑

        rs_result = map(lambda x: (x[0], sum(x[1])), result_table.items())
        rs_result = sorted(rs_result, key=lambda x: x[1], reverse=True)[:10]
        print(uid)
        print(rs_result)
