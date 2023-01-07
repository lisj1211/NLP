import torch
import torch.nn as nn


class Glove(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Glove, self).__init__()
        self.center_embedding = nn.Embedding(vocab_size, d_model)
        torch.nn.init.xavier_uniform_(self.center_embedding.weight)
        self.context_embedding = nn.Embedding(vocab_size, d_model)
        torch.nn.init.xavier_uniform_(self.context_embedding.weight)

        self.center_bias = nn.Embedding(vocab_size, 1)
        torch.nn.init.constant_(self.center_bias.weight, 0.0)
        self.context_bias = nn.Embedding(vocab_size, 1)
        torch.nn.init.constant_(self.context_bias.weight, 0.0)

    def forward(self, center_word, context_word, co_mat_val, weight_mat_val):
        center_word_embedding = self.center_embedding(center_word)
        context_word_embedding = self.context_embedding(context_word)
        center_word_bias = self.center_bias(center_word)
        context_word_bias = self.context_bias(context_word)

        # loss = f(X_ij) * (i_T * j + b_i + b_j - log(X_ij)) * (i_T * j + b_i + b_j - log(X_ij))
        similarity = torch.mul(center_word_embedding, context_word_embedding).sum(1)
        loss = similarity + center_word_bias + context_word_bias - torch.log(co_mat_val)
        loss = 0.5 * weight_mat_val * loss * loss

        return loss.sum().mean()

    def get_word_embedding(self):
        return self.center_embedding.weight.data.cpu().numpy() + self.context_embedding.weight.data.cpu().numpy()


