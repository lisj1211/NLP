import jieba
import pkuseg

import utils
from hmm import HMM, viterbi


def main():
    data_path = './data/train.txt'
    state_path = './data/state.txt'

    utils.make_label(data_path, state_path)
    hmm = HMM(data_path, state_path)
    hmm.train()

    texts = ["虽然一路上队伍里肃静无声",
             "刘晓庆1970年毕业于四川音乐学院附中，1975年走上银幕",
             "《西游记》的总导演杨洁由于心脏病去世"]
    seg = pkuseg.pkuseg()
    for text in texts:
        pk_cut = seg.cut(text)
        jieba_cut = jieba.lcut(text)
        hmm_cut = viterbi(text, hmm)
        print(f'raw_text: {text}\n'
              f'pk_cut: {pk_cut}\n'
              f'jieba_cut: {jieba_cut}\n'
              f'hmm_cut: {hmm_cut}\n')


if __name__ == '__main__':
    main()
