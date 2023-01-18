import argparse
import json
import os

import torch


def arg_parser():
    parser = argparse.ArgumentParser(description='Model Controller')
    # path
    parser.add_argument('--train_path', default='./data/preprocessed/train.json', type=str)
    parser.add_argument('--test_path', default='./data/preprocessed/dev.json', type=str)
    parser.add_argument('--ent2idx_path', default='./data/preprocessed/ent2idx.json', type=str)
    parser.add_argument('--model_save_path', default='./weight/model_r.pt', type=str)
    parser.add_argument("--pretrained_bert_path", default='./src/bert-base-chinese', type=str)
    # train
    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--eval_batch_size', default=32, type=int)
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--max_grad_norm', default=5, type=float)
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='warmup step')
    parser.add_argument('--max_sequence_length', default=128, type=int)
    parser.add_argument('--inner_dim', default=64, type=int)
    parser.add_argument('--period', default=200, type=int, help='logging period')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    with open(args.ent2idx_path, 'r', encoding='utf-8') as f:
        args.ent_type_size = len(json.load(f))

    return args
