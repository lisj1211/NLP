"""
Apriori频繁项集
"""
from collections import defaultdict
from pprint import pprint


def generate_freq_supports(data_set, item_set, min_support):
    """获得当前候选集item_set的频繁项集和支持度"""
    freq_set = set()  # 保存频繁项集元素
    item_count = defaultdict(int)  # 保存元素频次，用于计算支持度
    supports = {}  # 保存支持度

    # 如果项集中元素在数据集中则计数
    for record in data_set:
        for item in item_set:
            if item.issubset(record):
                item_count[item] += 1

    data_len = float(len(data_set))

    # 计算项集支持度
    for item, count in item_count.items():
        if count / data_len >= min_support:
            freq_set.add(item)
            supports[item] = count / data_len

    return freq_set, supports


def generate_new_combinations(freq_set, k):
    """根据频繁k-1项集，获得候选k项集"""
    new_combinations = set()  # 保存新组合
    sets_len = len(freq_set)  # 集合含有元素个数，用于遍历求得组合
    freq_set_list = list(freq_set)  # 集合转为列表用于索引

    for i in range(sets_len):
        for j in range(i + 1, sets_len):
            l1 = list(freq_set_list[i])
            l2 = list(freq_set_list[j])
            l1.sort()
            l2.sort()

            # 任一频繁项的所有非空子集也必须是频繁的，即所有频繁k-1项集必是频繁k项集的子集
            # 因此频繁k项集的任意一项肯定由两个只有1个不同元素的频繁k-1项集构成，所以先进行排序之后比较[:k-2]个元素
            # 比如对于频繁2项集: {[1,3],[2,3],[2,5],[3,5]}, [2,3]和[2,5]可以构成频繁3项集，
            # [1,3]和[2,3]不满足的原因为[1,2,3]的子集[1,2]不是频繁2项集
            # 因此排序操作是必要的
            if l1[0:k - 2] == l2[0:k - 2]:
                freq_item = freq_set_list[i] | freq_set_list[j]
                new_combinations.add(freq_item)

    return new_combinations


def apriori(data_set, min_support, max_len=None):
    max_items = 2  # 初始项集元素个数
    freq_sets = []  # 保存所有频繁项集
    supports = {}  # 保存所有支持度

    # 候选项1项集
    c1 = set()
    for items in data_set:
        for item in items:
            item_set = frozenset([item])
            c1.add(item_set)

    # 频繁项1项集及其支持度
    l1, support1 = generate_freq_supports(data_set, c1, min_support)

    freq_sets.append(l1)
    supports.update(support1)

    if max_len is None:
        max_len = float('inf')

    while max_items <= max_len:
        ci = generate_new_combinations(freq_sets[-1], max_items)  # 生成候选集
        li, support = generate_freq_supports(data_set, ci, min_support)  # 生成频繁项集和支持度

        # 如果有频繁项集则进入下个循环
        if li:
            freq_sets.append(li)
            supports.update(support)
            max_items += 1
        else:
            break

    return freq_sets, supports


def association_rules(freq_sets, supports, min_conf):
    rules = []
    max_len = len(freq_sets)

    # 生成关联规则，筛选符合规则的频繁集计算置信度，满足最小置信度的关联规则添加到列表
    for k in range(max_len - 1):
        for freq_set in freq_sets[k]:
            for sub_set in freq_sets[k + 1]:
                if freq_set.issubset(sub_set):
                    conf = supports[sub_set] / supports[freq_set]
                    rule = (freq_set, sub_set - freq_set, conf)
                    if conf >= min_conf:
                        rules.append(rule)
    return rules


if __name__ == '__main__':
    data = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

    L, support_data = apriori(data, min_support=0.5)
    print('support_data:')
    pprint(support_data)
    print('='*50)
    association_rules = association_rules(L, support_data, min_conf=0.7)
    print('association_rules:\n', association_rules)
