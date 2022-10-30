# -*- coding: utf-8 -*-
from tqdm import tqdm
import argparse

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils import Vocabulary, MyDataset, get_similar_tokens, save_loss_curve, save_dict
from model import SkipGram


def train(model, epochs, lr, train_dataloader, device):
    print('training on:', device)
    model.to(device)
    optimizer = Adam((param for param in model.parameters() if param.requires_grad), lr=lr)

    global_step = 0
    total_loss = 0.
    logg_loss = 0.
    train_loss = []

    for epoch in range(epochs):
        print(f"—————— training epoch {epoch} ——————")

        model.train()
        for encoded_center_word, encoded_pos_words, encoded_neg_words in tqdm(train_dataloader, desc='train'):

            encoded_center_word = encoded_center_word.to(device)
            encoded_pos_words = encoded_pos_words.to(device)
            encoded_neg_words = encoded_neg_words.to(device)

            loss = model(encoded_center_word, encoded_pos_words, encoded_neg_words)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            total_loss += loss

            if global_step % 100 == 0:
                loss_scalar = (total_loss - logg_loss) / 100
                logg_loss = total_loss
                print(f"epoch: {epoch}, iter: {global_step}, loss: {loss_scalar}")
                train_loss.append(loss_scalar)

    return train_loss


def main():
    parse = argparse.ArgumentParser()

    parse.add_argument("--corpus_path", type=str, help="your train corpus path.")
    parse.add_argument("--k", default=10, type=int, help="negative samples.")
    parse.add_argument("--window_size", default=2, type=int)
    parse.add_argument("--batch_size", default=4, type=int)
    parse.add_argument("--embed_size", default=200, type=int)
    parse.add_argument("--learning_rate", default=1e-4, type=float)
    parse.add_argument("--num_epoch", default=20, type=int)

    args = parse.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== load train data ==========
    voc = Vocabulary(args.corpus_path)
    dataset = MyDataset(voc.corpus, voc.word2idx, voc.word_frequence, k=args.k, WINDOW_SIZE=args.window_size)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    # ========== train ==========
    model = SkipGram(vocab_size=len(voc.word2idx), d_model=args.embed_size)
    train_loss = train(model, args.num_epoch, args.learning_rate, data_loader, device)

    # ========== save result ==========
    np.save("word_embedding.npy", model.get_word_embedding())
    save_dict(voc.word2idx, "word2idx.json")
    save_dict(voc.idx2word, "idx2word.json")
    save_loss_curve(train_loss)

    # ========== test ==========
    get_similar_tokens("篮球", 10, word_embedding_path="word_embedding.npy", word2idx_path="word2idx.json",
                       idx2word_path="idx2word.json")


if __name__ == "__main__":
    main()
