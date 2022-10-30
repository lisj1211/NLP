import random
import time
import argparse
from tqdm import tqdm
import os
import json

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from utils import load_bert_data, BertDataset, bert_collate_fn, acc_and_f1
from models import BertClassify


def set_seed():
    random.seed(1211)
    np.random.seed(1211)
    torch.manual_seed(1211)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1211)


def train(args, model, train_data, train_label, val_data, val_label, dtype='Bert'):
    print("start training model {} ".format(dtype).center(50, "="))
    time_start = time.time()

    train_dataset = BertDataset(train_data, train_label, args.cache_dir)
    train_loader = DataLoader(train_dataset, args.batch_size, collate_fn=bert_collate_fn, shuffle=True)
    val_dataset = BertDataset(val_data, val_label, args.cache_dir)
    val_loader = DataLoader(val_dataset, args.batch_size, collate_fn=bert_collate_fn, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    total_loss = 0.
    logg_loss = 0.
    train_loss = []
    val_acc = []

    for epoch in range(args.num_epoch):
        model.train()
        preds = None
        y_true = None

        for inputs, masks, labels in tqdm(train_loader, desc="training epoch {}".format(epoch)):
            inputs = torch.tensor(inputs, device=args.device)
            masks = torch.tensor(masks, device=args.device)
            y_labels = torch.tensor(labels, device=args.device)

            global_step += 1
            logits = model(inputs, masks)
            loss = criterion(logits, y_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if preds is None:
                preds = logits.detach().cpu().numpy()
                y_true = np.array(labels)
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                y_true = np.append(y_true, np.array(labels), axis=0)

            if global_step % 100 == 0:
                loss_scalar = (total_loss - logg_loss) / 100
                logg_loss = total_loss
                print("epoch: {}, iter: {}, loss: {:.4f}".format(epoch, global_step, loss_scalar))
                train_loss.append(loss_scalar)

        preds = np.argmax(preds, axis=1)
        acc, f1 = acc_and_f1(preds, y_true)
        print("Epoch: {}, Training loss: {}, Acc: {}, F1: {}".format(epoch, total_loss / global_step, acc, f1))
        eval_loss, eval_acc, eval_f1 = evaluate(args, criterion, model, val_loader)
        print(("Evaluate: loss: {}, eval_acc: {}, eval_f1: {}".format(eval_loss, eval_acc, eval_f1)))

        if len(val_acc) == 0 or eval_acc > max(val_acc):
            print("best model on epoch: {}, eval_acc: {}".format(epoch, eval_acc))
            torch.save(model.state_dict(), os.path.join(args.model_save_path, "model_{}.th".format(dtype)))
            val_acc.append(eval_acc)

    time_end = time.time()
    np.save(os.path.join(args.model_save_path, "{}_loss.npy".format(dtype)), np.array(train_loss))
    print("run model of {} taking total {} m".format(dtype, (time_end-time_start)/60))


def evaluate(args, criterion, model, val_data_loader):
    model.eval()
    total_loss = 0.
    total_step = 0.
    preds = None
    y_true = None
    with torch.no_grad():
        for inputs, masks, labels in tqdm(val_data_loader, desc="evaluate"):
            inputs = torch.tensor(inputs, device=args.device)
            masks = torch.tensor(masks, device=args.device)
            y_labels = torch.tensor(labels, device=args.device)

            logits = model(inputs, masks)
            loss = criterion(logits, y_labels)

            total_loss += loss.item()
            total_step += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                y_true = np.array(labels)
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                y_true = np.append(y_true, np.array(labels), axis=0)

    preds = np.argmax(preds, axis=1)
    acc, f1 = acc_and_f1(preds, y_true)
    model.train()
    return total_loss/total_step, acc, f1


def main():
    parse = argparse.ArgumentParser()

    parse.add_argument("--train_data_path", default='./data/cnews/train.json', type=str, required=False)
    parse.add_argument("--val_data_path", default='./data/cnews/val.json', type=str, required=False)
    parse.add_argument("--label_map_path", default='./data/cnews/label.json', type=str, required=False)
    parse.add_argument("--cache_dir", default='./bert_pretrain', type=str, required=False)
    parse.add_argument("--batch_size", default=8, type=int)
    parse.add_argument("--do_train", default=False, action="store_true", help="Whether to run training.")
    parse.add_argument("--do_test", default=True, action="store_true", help="Whether to run training.")
    parse.add_argument("--learning_rate", default=1e-4, type=float)
    parse.add_argument("--num_epoch", default=10, type=int)
    parse.add_argument("--model_save_path", default='./model_weights', type=str)

    args = parse.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    set_seed()

    if args.do_train:
        # ========== load data ==========
        train_sents, train_labels = load_bert_data(args.train_data_path)
        val_sents, val_labels = load_bert_data(args.val_data_path)
        label_map = json.load(open(args.label_map_path, 'r', encoding='utf-8'))

        bert = BertClassify(len(label_map), cache_dir=args.cache_dir, freeze=True)
        bert.to(device)
        train(args, bert, train_sents, train_labels, val_sents, val_labels)

    if args.do_test:
        test_sents, test_labels = load_bert_data('./data/cnews/test.json')
        label_map = json.load(open(args.label_map_path, 'r', encoding='utf-8'))

        test_dataset = BertDataset(test_sents, test_labels, args.cache_dir)
        test_loader = DataLoader(test_dataset, args.batch_size, collate_fn=bert_collate_fn, shuffle=False)

        bert = BertClassify(len(label_map), cache_dir=args.cache_dir)
        bert.load_state_dict(torch.load(os.path.join(args.model_save_path, "model_Bert.th")))
        bert.to(device)
        criterion = nn.CrossEntropyLoss()
        bert_loss, bert_acc, bert_f1 = evaluate(args, criterion, bert, test_loader)

        print('test result:')
        print("test model of {}, loss: {}, acc: {}, f1: {}".format('Bert', bert_loss, bert_acc, bert_f1))


if __name__ == '__main__':
    main()
