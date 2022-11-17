import os
import random
import argparse

import numpy as np
import torch
from transformers import AdamW

from src.spert import SpERT
from src.config import Config
from src.trainer import train, evaluate
from src.utils import Vocabulary, get_optimizer_params, save_loss_curve
from src.loss import MyLoss


def set_seed(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(num)


def main():
    parser = argparse.ArgumentParser(description='Model Controller')

    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
    parser.add_argument('--max_epoch', default=10, type=int, help='max_epoch')
    parser.add_argument('--max_norm', default=1.0, type=float, help='clip_grad_norm')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight_decay')
    parser.add_argument('--lr_warmup', default=0.1, type=float, help='lr_warmup')
    parser.add_argument("--bert_path", default='./src/bert-base-chinese', type=str, help='pretrained bert dir')
    parser.add_argument('--bert_dim', default=768, type=int, help='dimension of bert model')
    parser.add_argument('--size_embedding', default=25, type=int, help='span size embedding')
    parser.add_argument('--prop_drop', default=0.1, type=float, help='drop out rate')
    parser.add_argument('--freeze_bert', default=False, type=bool, help='freeze_bert')
    parser.add_argument('--neg_entity_count', default=100, type=int, help='negative entity counts')
    parser.add_argument('--neg_rel_count', default=100, type=int, help='negative relation counts')
    parser.add_argument('--max_span_size', default=10, type=int, help='max_span_size')
    parser.add_argument('--train_path', default='data/baidu/train.json', type=str)
    parser.add_argument('--dev_path', default='data/baidu/dev.json', type=str)
    parser.add_argument('--test_path', default='data/baidu/test.json', type=str)
    parser.add_argument('--type_path', default='data/baidu/types.json', type=str)
    parser.add_argument('--do_test', default=True, type=bool)

    args = parser.parse_args()

    config = Config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    set_seed(1211)

    type_vocab = Vocabulary(config.type_path)
    config.relation_types = len(type_vocab.rel2idx)
    config.entity_types = len(type_vocab.entity2idx)
    model = SpERT(config).to(device)

    rel_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    loss_func = MyLoss(rel_criterion, entity_criterion)

    optimizer_params = get_optimizer_params(model, config.weight_decay)
    optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)

    train_loss = train(config, model, optimizer, loss_func, type_vocab)
    save_loss_curve(train_loss)

    if args.do_test:
        model_path = os.path.join(config.save_weights_dir, config.weights_save_name)
        model.load_state_dict(torch.load(model_path, map_location=device))
        entity_result, relation_result = evaluate(config, model, config.test_path, type_vocab)
        print(f'test entity result: precision: {entity_result[0]:.4f}, recall: {entity_result[1]:.4f}, '
              f'f1: {entity_result[2]:.4f}\n '
              f'test relation result: precision: {relation_result[0]:.4f}, recall: {relation_result[1]:.4f}, '
              f'f1: {relation_result[2]:.4f}')


if __name__ == '__main__':
    main()
