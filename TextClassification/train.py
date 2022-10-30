import time
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from utils import MyDataset, my_collate_fn, acc_and_f1


def train_model(args, model, train_data, train_label, val_data, val_label, word2idx, dtype='TextCNN'):
    print("start training model {} ".format(dtype).center(50, "="))
    time_start = time.time()

    train_dataset = MyDataset(train_data, train_label, word2idx)
    train_loader = DataLoader(train_dataset, args.batch_size, collate_fn=my_collate_fn, shuffle=True)
    val_dataset = MyDataset(val_data, val_label, word2idx)
    val_loader = DataLoader(val_dataset, args.batch_size, collate_fn=my_collate_fn, shuffle=False)

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

        for sents, labels in tqdm(train_loader, desc="training epoch {}".format(epoch)):
            sents = torch.tensor(sents, device=args.device)

            global_step += 1
            logits = model(sents)
            y_labels = torch.tensor(labels, device=args.device)
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
        eval_loss, eval_acc, eval_f1 = evaluate_model(args, criterion, model, val_loader)
        print(("Evaluate: loss: {}, eval_acc: {}, eval_f1: {}".format(eval_loss, eval_acc, eval_f1)))

        if len(val_acc) == 0 or eval_acc > max(val_acc):
            print("best model on epoch: {}, eval_acc: {}".format(epoch, eval_acc))
            torch.save(model.state_dict(), os.path.join(args.model_save_path, "model_{}.th".format(dtype)))
            val_acc.append(eval_acc)

    time_end = time.time()
    np.save(os.path.join(args.model_save_path, "{}_loss.npy".format(dtype)), np.array(train_loss))
    print("run model of {} taking total {} m".format(dtype, (time_end-time_start)/60))


def evaluate_model(args, criterion, model, val_data_loader):
    model.eval()
    total_loss = 0.
    total_step = 0.
    preds = None
    y_true = None
    with torch.no_grad():
        for val_sents, val_labels in tqdm(val_data_loader, desc="evaluate"):
            val_sents = torch.tensor(val_sents, device=args.device)
            logits = model(val_sents)
            labels = torch.tensor(val_labels, device=args.device)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            total_step += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                y_true = np.array(val_labels)
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                y_true = np.append(y_true, np.array(val_labels), axis=0)

    preds = np.argmax(preds, axis=1)
    acc, f1 = acc_and_f1(preds, y_true)
    model.train()
    return total_loss/total_step, acc, f1
