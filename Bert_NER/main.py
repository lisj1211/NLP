import os
import argparse
import warnings

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import logging, BertConfig, get_linear_schedule_with_warmup, AdamW

from src.args import bilstm_crf_argparser, bert_argparser
from src.log import NERLoging
from src.trainer import train_lstm_with_crf, train_bert, train_bert_with_crf
from src.preprocess import DataProcess
from src.dataset import LstmDataset, BertDataset
from src.utils import (
    set_seed,
    Vocabulary,
    rnn_collate_fn,
    bert_collate_fn,
    get_optimizer_params,
    get_optimizer_params_for_bert_lstm_crf)
from src.models import BiLSTM_CRF, Bert, Bert_CRF, Bert_LSTM_CRF
from src.evaluate import evaluate_lstm_crf, evaluate_bert, evaluate_bert_with_crf

logging.set_verbosity_error()
warnings.filterwarnings('ignore')


def lstm_with_crf_model():
    config = bilstm_crf_argparser()
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = NERLoging('BiLSTM-CRF').logger
    logger.info("start training BiLSTM-CRF model")

    logger.info("loading train data...")
    train_processor = DataProcess(config.train_path, config.processed_train_data_path)
    train_data, train_tags = train_processor.preprocess(dtype=config.annotation_dtype)

    logger.info("building vocabulary...")
    vocab = Vocabulary(config)
    if os.path.exists(config.vocab_path):
        word2idx, idx2word = vocab.load_vocab(config.vocab_path)
    else:
        word2idx, idx2word = vocab.build_vocab(train_data)
        vocab.save_vocab(config.vocab_path)

    logger.info("preparing train and dev data...")
    x_train, x_dev, y_train, y_dev = train_test_split(train_data, train_tags, test_size=0.2, random_state=1211)
    train_dataset = LstmDataset(x_train, y_train, word2idx, config.label2idx)
    train_loader = DataLoader(train_dataset, config.batch_size, collate_fn=rnn_collate_fn, shuffle=True, drop_last=True)
    dev_dataset = LstmDataset(x_dev, y_dev, word2idx, config.label2idx)
    dev_loader = DataLoader(dev_dataset, config.batch_size, collate_fn=rnn_collate_fn, shuffle=False)

    logger.info("initializing model and optimizer...")
    model = BiLSTM_CRF(vocab_size=len(word2idx),
                       embedding_size=config.embedding_size,
                       hidden_size=config.hidden_size,
                       num_class=config.num_class,
                       drop_out=config.drop_out)

    updates_total = len(train_dataset) // config.batch_size
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=config.lr_warmup * updates_total,
                                                num_training_steps=updates_total)

    logger.info("training process:")
    train_lstm_with_crf(config, model, train_loader, dev_loader, optimizer, scheduler, logger)

    logger.info("testing process:")
    test_processor = DataProcess(config.test_path, config.processed_test_data_path)
    test_data, test_tags = test_processor.preprocess(dtype=config.annotation_dtype)

    test_dataset = LstmDataset(test_data, test_tags, word2idx, config.label2idx)
    test_loader = DataLoader(test_dataset, config.batch_size, collate_fn=rnn_collate_fn, shuffle=False)

    model.load_state_dict(torch.load(os.path.join(config.weight_save_dir, 'BiLSTM_CRF.pt'), map_location=config.device))
    precision, recall, f1 = evaluate_lstm_crf(config, model, test_loader)
    logger.info(f"test result:\nprecision: {precision:.2f}, recall {recall:.2f}, f1 {f1:.2f}")


def bert_model():
    config = bert_argparser()
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = NERLoging('Bert').logger
    logger.info("start training Bert model")

    logger.info("loading train data...")
    train_processor = DataProcess(config.train_path, config.processed_train_data_path)
    train_data, train_tags = train_processor.preprocess(dtype=config.annotation_dtype)

    logger.info("preparing train and dev data...")
    x_train, x_dev, y_train, y_dev = train_test_split(train_data, train_tags, test_size=0.2, random_state=1211)
    train_dataset = BertDataset(config, x_train, y_train)
    train_loader = DataLoader(train_dataset, config.batch_size, collate_fn=bert_collate_fn, shuffle=True, drop_last=True)
    dev_dataset = BertDataset(config, x_dev, y_dev)
    dev_loader = DataLoader(dev_dataset, config.batch_size, collate_fn=bert_collate_fn, shuffle=False)

    logger.info("initializing model and optimizer...")
    bertconfig = BertConfig.from_pretrained(config.bert_path)
    model = Bert.from_pretrained(config.bert_path,
                                 config=bertconfig,
                                 num_class=config.num_class,
                                 dropout=config.drop_out)

    updates_total = len(train_dataset) // config.batch_size
    optimizer_params = get_optimizer_params(config, model)
    optimizer = AdamW(optimizer_params, lr=config.lr, weight_decay=config.weight_decay, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=config.lr_warmup * updates_total,
                                                num_training_steps=updates_total)

    logger.info("training process...")
    train_bert(config, model, train_loader, dev_loader, optimizer, scheduler, logger)

    logger.info("testing process:")
    test_processor = DataProcess(config.test_path, config.processed_test_data_path)
    test_data, test_tags = test_processor.preprocess(dtype=config.annotation_dtype)

    test_dataset = BertDataset(config, test_data, test_tags)
    test_loader = DataLoader(test_dataset, config.batch_size, collate_fn=bert_collate_fn, shuffle=False)

    model.load_state_dict(torch.load(os.path.join(config.weight_save_dir, 'BERT.pt'), map_location=config.device))
    precision, recall, f1 = evaluate_bert(config, model, test_loader)
    logger.info(f"test result:\nprecision: {precision:.2f}, recall {recall:.2f}, f1 {f1:.2f}")


def bert_crf_model():
    config = bert_argparser()
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = NERLoging('Bert').logger
    logger.info("start training Bert_CRF model")

    logger.info("loading train data...")
    train_processor = DataProcess(config.train_path, config.processed_train_data_path)
    train_data, train_tags = train_processor.preprocess(dtype=config.annotation_dtype)

    logger.info("preparing train and dev data...")
    x_train, x_dev, y_train, y_dev = train_test_split(train_data, train_tags, test_size=0.2, random_state=1211)
    train_dataset = BertDataset(config, x_train, y_train)
    train_loader = DataLoader(train_dataset, config.batch_size, collate_fn=bert_collate_fn, shuffle=True, drop_last=True)
    dev_dataset = BertDataset(config, x_dev, y_dev)
    dev_loader = DataLoader(dev_dataset, config.batch_size, collate_fn=bert_collate_fn, shuffle=False)

    logger.info("initializing model and optimizer...")
    bertconfig = BertConfig.from_pretrained(config.bert_path)
    model = Bert_CRF.from_pretrained(config.bert_path,
                                     config=bertconfig,
                                     num_class=config.num_class,
                                     dropout=config.drop_out)

    updates_total = len(train_dataset) // config.batch_size
    optimizer_params = get_optimizer_params(config, model)
    optimizer = AdamW(optimizer_params, lr=config.lr, weight_decay=config.weight_decay, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=config.lr_warmup * updates_total,
                                                num_training_steps=updates_total)

    logger.info("training process...")
    train_bert_with_crf(config, model, train_loader, dev_loader, optimizer, scheduler, logger, model_name='Bert_CRF')

    logger.info("testing process:")
    test_processor = DataProcess(config.test_path, config.processed_test_data_path)
    test_data, test_tags = test_processor.preprocess(dtype=config.annotation_dtype)

    test_dataset = BertDataset(config, test_data, test_tags)
    test_loader = DataLoader(test_dataset, config.batch_size, collate_fn=bert_collate_fn, shuffle=False)

    model.load_state_dict(torch.load(os.path.join(config.weight_save_dir, 'Bert_CRF.pt'), map_location=config.device))
    precision, recall, f1 = evaluate_bert_with_crf(config, model, test_loader)
    logger.info(f"test result:\nprecision: {precision:.2f}, recall {recall:.2f}, f1 {f1:.2f}")


def bert_lstm_crf_model():
    config = bert_argparser()
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = NERLoging('Bert').logger
    logger.info("start training Bert_LSTM_CRF model")

    logger.info("loading train data...")
    train_processor = DataProcess(config.train_path, config.processed_train_data_path)
    train_data, train_tags = train_processor.preprocess(dtype=config.annotation_dtype)

    logger.info("preparing train and dev data...")
    x_train, x_dev, y_train, y_dev = train_test_split(train_data, train_tags, test_size=0.2, random_state=1211)
    train_dataset = BertDataset(config, x_train, y_train)
    train_loader = DataLoader(train_dataset, config.batch_size, collate_fn=bert_collate_fn, shuffle=True, drop_last=True)
    dev_dataset = BertDataset(config, x_dev, y_dev)
    dev_loader = DataLoader(dev_dataset, config.batch_size, collate_fn=bert_collate_fn, shuffle=False)

    logger.info("initializing model and optimizer...")
    bertconfig = BertConfig.from_pretrained(config.bert_path)
    model = Bert_LSTM_CRF.from_pretrained(config.bert_path,
                                          config=bertconfig,
                                          num_class=config.num_class,
                                          dropout=config.drop_out)

    updates_total = len(train_dataset) // config.batch_size
    optimizer_params = get_optimizer_params_for_bert_lstm_crf(config, model)
    optimizer = AdamW(optimizer_params, lr=config.lr, weight_decay=config.weight_decay, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=config.lr_warmup * updates_total,
                                                num_training_steps=updates_total)

    logger.info("training process...")
    train_bert_with_crf(config, model, train_loader, dev_loader, optimizer, scheduler, logger,
                        model_name='BERT_LSTM_CRF')

    logger.info("testing process:")
    test_processor = DataProcess(config.test_path, config.processed_test_data_path)
    test_data, test_tags = test_processor.preprocess(dtype=config.annotation_dtype)

    test_dataset = BertDataset(config, test_data, test_tags)
    test_loader = DataLoader(test_dataset, config.batch_size, collate_fn=bert_collate_fn, shuffle=False)

    model.load_state_dict(torch.load(os.path.join(config.weight_save_dir, 'BERT_LSTM_CRF.pt'), map_location=config.device))
    precision, recall, f1 = evaluate_bert_with_crf(config, model, test_loader)
    logger.info(f"test result:\nprecision: {precision:.2f}, recall {recall:.2f}, f1 {f1:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--model_name', default='BERT', type=str, help='select one model from model lists',
                        choices=['BiLSTM_CRF', 'BERT', 'BERT_CRF', 'BERT_LSTM_CRF'])
    args = parser.parse_args()

    set_seed(1211)

    if args.model_name == 'BiLSTM_CRF':
        lstm_with_crf_model()
    elif args.model_name == 'BERT':
        bert_model()
    elif args.model_name == 'BERT_CRF':
        bert_crf_model()
    elif args.model_name == 'BERT_LSTM_CRF':
        bert_lstm_crf_model()
    else:
        raise ValueError("Invalid argument 'model_name', 'model_name' must in "
                         "['BiLSTM_CRF', 'BERT', 'BERT_CRF', 'BERT_LSTM_CRF']")


if __name__ == '__main__':
    main()
