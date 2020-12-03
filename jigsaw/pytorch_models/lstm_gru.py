import torch
from torch import nn
from torch.nn import functional as F
from pytorch_models.dropout import SequentialDropout


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class LstmGru(nn.Module):
    def __init__(self, embedding_matrix, features_size, max_features, num_aux_targets, lstm_units, dense_hidden_units,
                 dropout_rate):
        super(LstmGru, self).__init__()
        embed_size = embedding_matrix.shape[1]
        self.lstm_units = lstm_units

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(dropout_rate)
        self.test = SequentialDropout()

        self.lstm1 = nn.LSTM(embed_size, lstm_units, bidirectional=True, batch_first=True)
        self.gru1 = nn.GRU(lstm_units * 2, lstm_units, bidirectional=True, batch_first=True)

        self.linear_out = nn.Linear(dense_hidden_units + features_size, 1)
        self.linear_aux_out = nn.Linear(dense_hidden_units + features_size, num_aux_targets)

    def forward(self, x, features):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_gru1, hn_gru1 = self.gru1(h_lstm1)

        avg_pool = torch.mean(h_gru1, 1)
        max_pool, _ = torch.max(h_gru1, 1)
        hn_gru1 = hn_gru1.view(-1, self.lstm_units * 2)

        h_conc = torch.cat((avg_pool, hn_gru1, max_pool), 1)

        result = self.linear_out(h_conc)
        aux_result = self.linear_aux_out(h_conc)
        out = torch.cat([result, aux_result], 1)

        return out


class LstmGru_v2(nn.Module):
    def __init__(self, embedding_matrix, features_size, max_features, num_aux_targets, lstm_units, gru_units,
                 hidden_units, dropout_rate):
        super(LstmGru_v2, self).__init__()
        embed_size = embedding_matrix.shape[1]
        self.lstm_units = lstm_units
        self.gru_units = gru_units

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(dropout_rate)
        self.test = SequentialDropout()

        self.lstm1 = nn.LSTM(embed_size, lstm_units, bidirectional=True, batch_first=True)
        self.gru1 = nn.GRU(lstm_units * 2, gru_units, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(gru_units * 6, hidden_units)

        self.linear_out = nn.Linear(hidden_units + features_size, 1)
        self.linear_aux_out = nn.Linear(hidden_units + features_size, num_aux_targets)

    def forward(self, x, features):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_gru1, hn_gru1 = self.gru1(h_lstm1)

        avg_pool = torch.mean(h_gru1, 1)
        max_pool, _ = torch.max(h_gru1, 1)
        hn_gru1 = hn_gru1.view(-1, self.gru_units * 2)

        h_conc = torch.cat((hn_gru1, avg_pool, max_pool), 1)

        h_conc_linear1 = F.relu(self.linear1(h_conc))

        result = self.linear_out(h_conc_linear1)
        aux_result = self.linear_aux_out(h_conc_linear1)
        out = torch.cat([result, aux_result], 1)

        return out
