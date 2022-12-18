import os
from tqdm import tqdm


def make_label(data_path: str, state_path: str) -> None:
    """
    将训练数据转换为"BMES"标注文件
    :param data_path: 训练数据路径
    :param state_path: 转换后的输出路径
    :return:
    """
    if os.path.exists(state_path):
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    total = len(data) - 1
    with open(state_path, 'w', encoding='utf-8') as f:
        for idx, sent in enumerate(tqdm(data, desc='label trans')):
            state = ' '.join(_make_label_for_word(word) for word in sent.split())
            if idx != total:
                state += '\n'
            f.write(state)


def _make_label_for_word(word: str) -> str:
    """
    从单词到label的转换, 如: 今天 ----> BE  麻辣肥牛: ---> BMME  的 ---> S
    :param word:
    :return:
    """
    if len(word) == 1:
        return 'S'

    return 'B' + 'M' * (len(word) - 2) + 'E'


