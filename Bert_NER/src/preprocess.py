import os
import json

import src.utils as utils


class DataProcess:
    """
    read and preprocess data
    """
    def __init__(self, data_path, processed_data_path):
        self.data_path = data_path
        self.processed_data_path = processed_data_path

    def preprocess(self, dtype):
        """
        将一段文本，如"我下一个玩的游戏是艾尔登法环"，转换为BIOES或BIO格式，seqeval库支持两种格式的指标评估：
        data：['我', '下', '一', '个', '玩', '的', '游', '戏', '是', '艾', '尔', '登', '法', '环']
        tag：['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-game', 'I-game', 'I-game', 'I-game', 'E-game']
        :return:
        """
        if os.path.exists(self.processed_data_path):
            dic = utils.load_pickle(self.processed_data_path)
            return dic['data'], dic['tags']

        data, tags = [], []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            raw_data = f.readlines()

        for line in raw_data:
            line = json.loads(line)
            text = line.get('text')
            label = line.get('label')

            if not text or not label:
                continue

            text = text.strip().lower()  # 中文Bert词典只有小写英文，保证正常训练，全部调整为小写
            data.append(list(text))
            if dtype == 'BIOES':
                tags.append(_bioes(text, label))
            elif dtype == 'BIO':
                tags.append(_bio(text, label))
            else:
                raise ValueError('Argument Error')

        utils.save_pickle({'data': data, 'tags': tags}, self.processed_data_path)

        return data, tags


def _bioes(text, label):
    tmp_tag = ['O'] * len(text)
    for entity_label, entity_dic in label.items():
        for name, index in entity_dic.items():
            name = name.lower()  # 同时调整标签中的实体为小写
            for start, end in index:
                assert text[start:end + 1] == name, f"label is {name}, got {text[start:end + 1]}"
                if start == end:
                    tmp_tag[start] = 'S-' + entity_label
                else:
                    tmp_tag[start] = 'B-' + entity_label
                    tmp_tag[start + 1:end] = ['I-' + entity_label] * (len(name) - 2)
                    tmp_tag[end] = 'E-' + entity_label

    return tmp_tag


def _bio(text, label):
    tmp_tag = ['O'] * len(text)
    for entity_label, entity_dic in label.items():
        for name, index in entity_dic.items():
            name = name.lower()  # 同时调整标签中的实体为小写
            for start, end in index:
                assert text[start:end + 1] == name, f"label is {name}, got {text[start:end + 1]}"

                tmp_tag[start] = 'B-' + entity_label
                if start != end:
                    tmp_tag[start + 1:end + 1] = ['I-' + entity_label] * (len(name) - 1)

    return tmp_tag
