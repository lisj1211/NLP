# -*- coding: utf-8 -*-
"""
Swing属于召回算法的一种
itemCF：如果同时喜欢两个物品的用户越多，那么这两个物品的相似度越高。
Swing：如果同时喜欢两个物品的用户越多，且这些用户的重合度越低，那么这两个物品的相似度越高。
Swing计算相似度公式： sim(i1, i2) = sum[1 / (alpha + overlap(u1, u2))], overlap(u1, u2)表示同时喜欢物品i1, i2的两个用户各自
喜欢物品的交集overlap(u1,u2)=∣I_u1 ∩ I_u2∣, I_u1表示用户u1喜欢的物品集合
"""
from itertools import combinations
from collections import defaultdict
from pprint import pprint

import pandas as pd

alpha = 1


def get_user_item_map(df):
    user2item = df.groupby('userId')['itemId'].agg(set)
    item2user = df.groupby('itemId')['userId'].agg(set)
    return dict(user2item), dict(item2user)


def swing(user2item, item2user):
    item_pairs = combinations(item2user.keys(), 2)
    item_sim_dict = defaultdict(dict)
    for i, j in item_pairs:
        # 喜欢item_i和item_j的user取交集后组合得到user对
        user_pairs = combinations(item2user[i] & item2user[j], 2)
        result = 0
        for u, v in user_pairs:
            result += 1 / (alpha + len(user2item[u] & user2item[v]))
        if result != 0:
            item_sim_dict[i][j] = round(result, 3)
            item_sim_dict[j][i] = round(result, 3)
    return dict(item_sim_dict)


if __name__ == '__main__':
    data = [[1, 1, 3],
            [1, 2, 0],
            [1, 3, 2],
            [1, 4, 1],
            [1, 5, 0],
            [2, 2, 2],
            [2, 4, 4],
            [2, 7, 2],
            [2, 6, 0],
            [2, 1, 3],
            [3, 6, 1],
            [3, 4, 2],
            [3, 7, 0],
            [3, 2, 0],
            [3, 8, 4]]
    data = pd.DataFrame(data, columns=['userId', 'itemId', 'rate'])
    user2item, item2user = get_user_item_map(data)
    item_sim_dict = swing(user2item, item2user)
    pprint(item_sim_dict)
