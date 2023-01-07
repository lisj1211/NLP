import argparse

import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adagrad
from torch.utils.data import DataLoader

from utils import Vocabulary, MyDataset, get_similar_tokens, save_dict, build_weight_matrix, load_pickle
from models import Glove


def train(model, epochs, lr, train_dataloader, device):
    print('training on:', device)
    model.to(device)
    optimizer = Adagrad(model.parameters(), lr=lr)

    global_step = 0
    total_loss = 0.
    log_loss = 0.

    for epoch in range(epochs):

        model.train()
        for batch in tqdm(train_dataloader, desc=f'train epoch {epoch}'):
            center_words, context_words, co_mat_vals, weight_mat_vals = batch

            center_words = center_words.to(device)
            context_words = context_words.to(device)
            co_mat_vals = co_mat_vals.to(device)
            weight_mat_vals = weight_mat_vals.to(device)

            loss = model(center_words, context_words, co_mat_vals, weight_mat_vals)
            total_loss += loss.item()

            global_step += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % 1000 == 0:
                loss_scaler = (total_loss - log_loss) / 1000
                log_loss = total_loss
                print(f"epoch:{epoch}, iter:{global_step}, loss:{loss_scaler}")


def main():
    parser = argparse.ArgumentParser(description='Model Controller')

    parser.add_argument("--corpus_path", default='./data/corpus.txt', type=str)
    parser.add_argument("--is_processed", default=False, type=bool, help='already has preprocessed data')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--embedding_dim', type=int, default=200)
    parser.add_argument('--max_vocab_size', type=int, default=10000)
    parser.add_argument('--window_size', type=int, default=5)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.is_processed:
        vocab = load_pickle('./data/vocab.pkl')
        word2idx, idx2word = vocab['word2idx'], vocab['idx2word']
        cooccur_matrix = load_pickle('./data/cooccur_matrix.pkl')
        weight_matrix = load_pickle('./data/weight_matrix.pkl')
    else:
        vocab = Vocabulary(args.corpus_path)
        word2idx, idx2word = vocab.build_vocabulary(args.max_vocab_size)
        cooccur_matrix = vocab.build_cooccurance_matrix(args.window_size)
        weight_matrix = build_weight_matrix(cooccur_matrix)

    dataset = MyDataset(cooccur_matrix, weight_matrix)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    model = Glove(vocab_size=len(word2idx), d_model=args.embedding_dim)

    train(model=model, epochs=args.max_epoch, lr=args.lr, train_dataloader=dataloader, device=device)

    np.save("word_embedding.npy", model.get_word_embedding())
    get_similar_tokens("篮球", 10, model.get_word_embedding(), word2idx, idx2word)


if __name__ == "__main__":
    main()
