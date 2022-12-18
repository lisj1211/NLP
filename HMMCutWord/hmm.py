import os
import pickle
from typing import List
from tqdm.contrib import tzip

import numpy as np


class HMM:
    """
    隐马尔可夫模型
    """

    def __init__(self, data_path: str, state_path: str):
        self.data_path = data_path
        self.state_path = state_path
        self.state2index = {"B": 0, "M": 1, "S": 2, "E": 3}
        self.index2state = ["B", "M", "S", "E"]
        self.len_states = len(self.state2index)

        self.init_matrix = np.zeros(self.len_states)
        self.transfer_matrix = np.zeros((self.len_states, self.len_states))
        self.emit_matrix = {"B": {"total": 0}, "M": {"total": 0}, "S": {"total": 0}, "E": {"total": 0}}
        # 这里初始化了一个  total 键 , 存储当前状态出现的总次数, 为了后面的归一化使用

    def cal_init_matrix(self, state: str):
        """
        计算初始状态矩阵
        :param state: 
        :return: 
        """
        self.init_matrix[self.state2index[state[0]]] += 1

    def cal_transfer_matrix(self, states: List[str]):
        """
        计算转移状态矩阵
        :param states: 
        :return: 
        """
        sta_join = "".join(states)  # 状态转移,从当前状态转移到后一状态, 即从 sta1 每一元素转移到 sta2 中
        sta1 = sta_join[:-1]  # trick
        sta2 = sta_join[1:]
        for s1, s2 in zip(sta1, sta2):  # 同时遍历 s1 , s2
            self.transfer_matrix[self.state2index[s1], self.state2index[s2]] += 1

    def cal_emit_matrix(self, words: List[str], states: List[str]):
        """
        计算发射矩阵
        :param words: 
        :param states: 
        :return: 
        """
        for word, state in zip("".join(words), "".join(states)):
            self.emit_matrix[state][word] = self.emit_matrix[state].get(word, 0) + 1
            self.emit_matrix[state]["total"] += 1

    def normalize(self):
        """
        归一化
        :return: 
        """
        self.init_matrix = self.init_matrix / np.sum(self.init_matrix)
        self.transfer_matrix = self.transfer_matrix / np.sum(self.transfer_matrix, axis=1, keepdims=True)
        self.emit_matrix = {
            state: {word: t / word_times["total"] * 1000 for word, t in word_times.items() if word != "total"}
            for state, word_times in self.emit_matrix.items()}  # * 1000为了防止小数连乘后近似为0

    def train(self):
        if os.path.exists("./data/three_matrix.pkl"):
            self.init_matrix, self.transfer_matrix, self.emit_matrix = pickle.load(open("./data/three_matrix.pkl", "rb"))
            return

        states_list = open(self.state_path, "r", encoding="utf-8").readlines()
        data_list = open(self.data_path, "r", encoding="utf-8").readlines()
        for words, states in tzip(data_list, states_list, desc="training"):
            words = words.split()
            states = states.split()
            self.cal_init_matrix(states[0])
            self.cal_transfer_matrix(states)
            self.cal_emit_matrix(words, states)
        self.normalize()
        pickle.dump([self.init_matrix, self.transfer_matrix, self.emit_matrix], open("./data/three_matrix.pkl", "wb"))


def viterbi(text: str, hmm: HMM) -> List[str]:
    idx2state = hmm.index2state
    state2idx = hmm.state2index
    emit_mat = hmm.emit_matrix
    trans_mat = hmm.transfer_matrix
    init_mat = hmm.init_matrix
    V = [{}]  # 每个step时，每个状态的概率
    path = {}  # 来到当前step之前经过的路径
    for state in idx2state:  # 计算初始概率
        V[0][state] = init_mat[state2idx[state]] * emit_mat[state].get(text[0], 0)
        path[state] = [state]
    for t in range(1, len(text)):
        V.append({})
        new_path = {}

        # 检验训练的发射概率矩阵中是否有该字
        has_word = text[t] in emit_mat['S'].keys() or \
                   text[t] in emit_mat['M'].keys() or \
                   text[t] in emit_mat['E'].keys() or \
                   text[t] in emit_mat['B'].keys()
        for cur_state in idx2state:
            emit_prob = emit_mat[cur_state].get(text[t], 0) if has_word else 1.0  # 设置未知字单独成词
            temp = []
            for pre_state in idx2state:
                if V[t - 1][pre_state] > 0:  # 只计算非0值
                    # 当前状态prob = 前一状态prob * 转移prob * 发射prob
                    cur_state_prob = V[t - 1][pre_state] * \
                                     trans_mat[state2idx[pre_state], state2idx[cur_state]] * \
                                     emit_prob
                    temp.append((cur_state_prob, pre_state))
            (prob, pre_state) = max(temp)  # 求最大的prob
            V[t][cur_state] = prob  # 更新来到当前step，不同状态的概率
            new_path[cur_state] = path[pre_state] + [cur_state]
        path = new_path  # 更新路径信息

    (last_prob, last_state) = max([(V[len(text) - 1][state], state) for state in idx2state])  # 求最后一个step的max_prob

    result = []
    for t, s in zip(text, path[last_state]):
        result.append(t)
        if s == "S" or s == "E":  # 如果是 S 或者 E 就在后面添加空格
            result.append(" ")
    return ''.join(result).split()
