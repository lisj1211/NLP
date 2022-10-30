import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import LoadData, MyDataset, my_collate_fn
from models import FastText, LSTM, TextCNN
from train import train_model, evaluate_model


def set_seed():
    random.seed(1211)
    np.random.seed(1211)
    torch.manual_seed(1211)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1211)


def main():
    parse = argparse.ArgumentParser()

    parse.add_argument("--train_data_path", default='./data/cnews/train.json', type=str, required=False)
    parse.add_argument("--val_data_path", default='./data/cnews/val.json', type=str, required=False)
    parse.add_argument("--label_map_path", default='./data/cnews/label.json', type=str, required=False)
    parse.add_argument("--stop_word_path", default='./data/stopwords.txt', type=str, required=False)
    parse.add_argument("--batch_size", default=8, type=int)
    parse.add_argument("--do_train", default=False, help="Whether to run training.")
    parse.add_argument("--do_test", default=True, help="Whether to run training.")
    parse.add_argument("--learning_rate", default=1e-4, type=float)
    parse.add_argument("--num_epoch", default=10, type=int)
    parse.add_argument("--max_vocab_size", default=10000, type=int)
    parse.add_argument("--min_freq", default=3, type=int)
    parse.add_argument("--embed_size", default=300, type=int)
    parse.add_argument("--hidden_size", default=256, type=int)
    parse.add_argument("--dropout_rate", default=0.2, type=float)
    parse.add_argument("--model_save_path", default='./model_weights', type=str)
    parse.add_argument("--do_textcnn", default=True, help="Whether to run training.")
    parse.add_argument("--do_lstm", default=True, help="Whether to run training.")
    parse.add_argument("--do_fasttext", default=True, help="Whether to run training.")
    parse.add_argument("--num_filter", default=100, type=int, help="TextCNN模型的卷积核数量")

    args = parse.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    set_seed()

    if args.do_train:
    # ========== load train data ==========
        train_data_reader = LoadData()
        train_sents, train_labels = train_data_reader.load_corpus(args.train_data_path, args.stop_word_path, args.label_map_path)
        word2id, id2word = train_data_reader.build_vocabulary(train_sents, word_size=args.max_vocab_size, min_freq=args.min_freq)
        label_map = train_data_reader.label_map

    # ========== load val data ==========
        val_data_reader = LoadData()
        val_sents, val_labels = val_data_reader.load_corpus(args.val_data_path, args.stop_word_path)

    # ========== save vocabulary ==========
        if not os.path.exists(args.model_save_path):
            os.makedirs(args.model_save_path)
        train_data_reader.save_vocab(os.path.join(args.model_save_path, "vocab.json"))

        if args.do_textcnn:
            textcnn = TextCNN(len(word2id), args.embed_size, args.num_filter, [2, 3, 4],
                              len(label_map), dropout=args.dropout_rate)
            textcnn.to(device)
            train_model(args, textcnn, train_sents, train_labels, val_sents, val_labels, word2id, dtype='TextCNN')

        if args.do_fasttext:
            fasttext = FastText(len(word2id), args.embed_size, len(label_map), dropout=args.dropout_rate)
            fasttext.to(device)
            train_model(args, fasttext, train_sents, train_labels, val_sents, val_labels, word2id, dtype='FastText')

        if args.do_lstm:
            lstm = LSTM(len(word2id), args.embed_size, args.hidden_size, len(label_map),
                        n_layers=2, bidirectional=True, dropout=args.dropout_rate)
            lstm.to(device)
            train_model(args, lstm, train_sents, train_labels, val_sents, val_labels, word2id, dtype='LSTM')

    if args.do_test:
        test_data_path = './data/cnews/test.json'
        test_data_reader = LoadData()
        test_sents, test_labels = test_data_reader.load_corpus(test_data_path, args.stop_word_path)
        word2id, label_map = LoadData.load_vocab(os.path.join(args.model_save_path, "vocab.json"))

        test_dataset = MyDataset(test_sents, test_labels, word2id)
        test_loader = DataLoader(test_dataset, args.batch_size, collate_fn=my_collate_fn, shuffle=False)

        cirtion = nn.CrossEntropyLoss()

        textcnn_model = TextCNN(len(word2id), args.embed_size, args.num_filter, [2, 3, 4],
                                len(label_map), dropout=args.dropout_rate)
        textcnn_model.load_state_dict(torch.load("./model_weights/model_TextCNN.th"))
        textcnn_model.to(device)
        cnn_loss, cnn_acc, cnn_f1 = evaluate_model(args, cirtion, textcnn_model, test_loader)

        fasttext_model = FastText(len(word2id), args.embed_size, len(label_map), dropout=args.dropout_rate)
        fasttext_model.load_state_dict(torch.load("./model_weights/model_FastText.th"))
        fasttext_model.to(device)
        fasttext_loss, fasttext_acc, fasttext_f1 = evaluate_model(args, cirtion, fasttext_model, test_loader)

        lstm_model = LSTM(len(word2id), args.embed_size, args.hidden_size,
                          len(label_map), n_layers=2, bidirectional=True, dropout=args.dropout_rate)
        lstm_model.load_state_dict(torch.load("./model_weights/model_LSTM.th"))
        lstm_model.to(device)
        lstm_loss, lstm_acc, lstm_f1 = evaluate_model(args, cirtion, lstm_model, test_loader)

        print("test result:")
        print("test model of {}, loss: {}, acc: {}, f1: {}".format('TextCNN', cnn_loss, cnn_acc, cnn_f1))
        print("test model of {}, loss: {}, acc: {}, f1: {}".format('FastText', fasttext_loss, fasttext_acc, fasttext_f1))
        print("test model of {}, loss: {}, acc: {}, f1: {}".format('LSTM', lstm_loss, lstm_acc, lstm_f1))


if __name__ == '__main__':
    main()
