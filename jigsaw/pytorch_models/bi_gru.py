from collections import OrderedDict
import torch
from torch import nn


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class BiGru(nn.Module):
    def __init__(self, embedding_matrix, features_size, max_features, num_aux_targets, lstm_units, hidden_units,
                 emb_dropout_rate, gru_dropout_rate, dense_dropout_rate):
        super(BiGru, self).__init__()
        embed_size = embedding_matrix.shape[1]
        self.lstm_units = lstm_units
        self.hidden_units = hidden_units

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(emb_dropout_rate)

        self.gru1 = nn.GRU(embed_size, lstm_units, bidirectional=True, batch_first=True, dropout=gru_dropout_rate)
        self.gru2 = nn.GRU(lstm_units * 2, lstm_units, bidirectional=True, batch_first=True, dropout=gru_dropout_rate)

        self.classifier = nn.Sequential(
            OrderedDict([
                ('gru_dropout', nn.Dropout(dense_dropout_rate)),
                ('h1', nn.Linear(lstm_units * 6, 72)),
                ('relu1', nn.ReLU()),
                ('out', nn.Linear(72, 1)),
            ]))
        self.classifier_aux = nn.Sequential(
            OrderedDict([
                ('gru_dropout_aux', nn.Dropout(dense_dropout_rate)),
                ('h1_aux', nn.Linear(lstm_units * 6, 72)),
                ('relu1_aux', nn.ReLU()),
                ('out_aux', nn.Linear(72, num_aux_targets)),
            ]))

    def forward(self, x, features):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        h_gru1, _ = self.gru1(h_embedding)
        h_gru2, hn_gru2 = self.gru2(h_gru1)

        # global average pooling
        avg_pool = torch.mean(h_gru2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_gru2, 1)
        # last hidden state
        hn_gru2 = hn_gru2.view(-1, self.lstm_units * 2)
        #  print(avg_pool.size(), '==', max_pool.size(), '==', hn_gru2.size())

        h_conc = torch.cat((hn_gru2, max_pool, avg_pool), 1)

        result = self.classifier(h_conc)
        aux_result = self.classifier_aux(h_conc)
        out = torch.cat([result, aux_result], 1)

        return out
