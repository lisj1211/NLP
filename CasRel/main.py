import os
import argparse
import random

import numpy as np
import torch
from torch.optim import AdamW

from src.casRel import CasRel
from src.config import Config
from src.trainer import train, evaluate
from src.utils import RelVocab, save_loss_curve
from src.myloss import MyLoss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description='Model Controller')

    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
    parser.add_argument('--max_epoch', default=10, type=int, help='max_epoch')
    parser.add_argument('--max_len', default=300, type=int, help='input max length')
    parser.add_argument('--max_norm', default=1, type=int, help='clip_grad_norm')
    parser.add_argument('--weight_decay', default=0.01, type=int, help='weight_decay')
    parser.add_argument('--lr_warmup', default=0.1, type=int, help='warmup step')
    parser.add_argument("--bert_path", default='./src/bert-base-chinese', type=str, help='pretrained bert dir')
    parser.add_argument('--bert_dim', default=768, type=int, help='dimension of bert model')
    parser.add_argument('--train_path', default='data/baidu/train.json', type=str)
    parser.add_argument('--dev_path', default='data/baidu/dev.json', type=str)
    parser.add_argument('--test_path', default='data/baidu/test.json', type=str)
    parser.add_argument('--rel_path', default='data/baidu/rel.json', type=str)
    parser.add_argument('--do_test', default=True, type=bool)

    args = parser.parse_args()

    config = Config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    set_seed(1211)

    model = CasRel(config).to(device)
    rel_vocab = RelVocab(config.rel_path)

    loss_func = MyLoss()
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    train_loss = train(config, model, loss_func, optimizer, rel_vocab)
    np.save('loss.npy', np.array(train_loss))
    save_loss_curve(train_loss)

    if args.do_test:
        model_path = os.path.join(config.save_weights_dir, config.weights_save_name)
        model.load_state_dict(torch.load(model_path))
        precision, recall, f1_score = evaluate(config, model, config.test_path, rel_vocab)
        print(f'test result: f1: {f1_score:.2f}, precision: {precision:.2f}, recall: {recall:.2f}')


if __name__ == '__main__':
    main()
