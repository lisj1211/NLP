import warnings
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

warnings.filterwarnings('ignore')


def prepare_data():
    users = ['User1', 'User2', 'User3', 'User4', 'User5']
    items = ['ItemA', 'ItemB', 'ItemC', 'ItemD', 'ItemE']

    # 用户购买记录，1表示购买，0表示未购买
    datasets = [[1, 0, 1, 1, 0],
                [1, 0, 0, 1, 1],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 1],
                [1, 1, 1, 0, 1]]
    df = pd.DataFrame(datasets, columns=items, index=users)
    return users, items, df


def user_based(users, df):
    """
    User-Based CF
    :param users:
    :param df:
    :return:
    """
    # 计算用户相似度
    user_similar = 1 - pairwise_distances(df.values, metric='jaccard')
    user_similar = pd.DataFrame(user_similar, columns=users, index=users)

    # 计算top2相似用户
    topN_users = {}
    for i in user_similar.index:
        _df = user_similar.loc[i].drop([i])
        _df_sorted = _df.sort_values(ascending=False)

        top2 = list(_df_sorted.index[:2])
        topN_users[i] = top2

    # 构建推荐结果
    rs_results = {}
    for user, sim_users in topN_users.items():
        rs_result = set()  # 存储推荐结果
        for sim_user in sim_users:
            rs_result = rs_result.union(set(df.loc[sim_user].replace(0, np.nan).dropna().index))
        rs_result -= set(df.loc[user].replace(0, np.nan).dropna().index)  # 过滤已买的商品
        rs_results[user] = rs_result

    print("基于用户的推荐结果:")
    pprint(rs_results)


def item_based(items, df):
    """
    Item-Based CF
    :param items:
    :param df:
    :return:
    """
    # 计算物品相似度
    item_similar = 1 - pairwise_distances(df.values.T, metric='jaccard')
    item_similar = pd.DataFrame(item_similar, columns=items, index=items)

    # 计算top2相似物品
    topN_items = {}
    for i in item_similar.index:
        _df = item_similar.loc[i].drop([i])
        _df_sorted = _df.sort_values(ascending=False)

        top2 = list(_df_sorted.index[:2])
        topN_items[i] = top2

    # 构建推荐结果
    rs_results = {}
    for user in df.index:
        rs_result = set()
        for item in df.loc[user].replace(0, np.nan).dropna().index:  # 遍历每个人已购买的商品列表
            rs_result = rs_result.union(topN_items[item])
        rs_result -= set(df.loc[user].replace(0, np.nan).dropna().index)  # 过滤已买商品
        rs_results[user] = rs_result

    print("基于商品的推荐结果:")
    pprint(rs_results)


def main():
    users, items, df = prepare_data()
    user_based(users, df)
    item_based(items, df)


if __name__ == '__main__':
    main()
    