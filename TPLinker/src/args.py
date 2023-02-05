import argparse
import json
import os

import torch


def arg_parser():
    parser = argparse.ArgumentParser(description='Model Controller')
    # path
    parser.add_argument('--train_path', default='./data/baidu/preprocessed/train.json', type=str)
    parser.add_argument('--dev_path', default='./data/baidu/preprocessed/dev.json', type=str)
    parser.add_argument('--test_path', default='./data/baidu/preprocessed/test.json', type=str)
    parser.add_argument('--rel_path', default='./data/baidu/preprocessed/rel.json', type=str)
    parser.add_argument('--model_save_path', default='./weight/model.pt', type=str)
    parser.add_argument("--pretrained_bert_path", default='./src/bert-base-chinese', type=str)
    # train
    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--train_batch_size', default=6, type=int)
    parser.add_argument('--eval_batch_size', default=32, type=int)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--max_grad_norm', default=5, type=float)
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='warmup step')
    parser.add_argument('--loss_weight_recover_steps', default=200000, type=int, help='entity and relation loss weight')
    parser.add_argument('--train_max_length', default=100, type=int, help='train max sequence length')
    parser.add_argument('--train_window_size', default=50, type=int, help='train truncate window length')
    parser.add_argument('--period', default=400, type=int, help='logging period')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    with open(args.rel_path, 'r', encoding='utf-8') as f:
        args.num_rels = len(json.load(f))

    return args
