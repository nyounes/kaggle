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


class BiLSTM(nn.Module):
    def __init__(self, embedding_matrix, features_size, max_features, num_aux_targets, lstm_units,
                 dense_units1, dense_units2, dense_units3, dropout_rate):
        super(BiLSTM, self).__init__()
        embed_size = embedding_matrix.shape[1]
        self.lstm_units = lstm_units

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(dropout_rate)

        self.conv1 = nn.Conv1d(

        self.lstm1 = nn.LSTM(embed_size, lstm_units, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_units * 2, lstm_units, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(dense_units1, dense_units2)
        self.linear2 = nn.Linear(dense_units2, dense_units3)

        self.linear_out = nn.Linear(dense_units3 + features_size, 1)
        self.linear_aux_out = nn.Linear(dense_units3 + features_size, num_aux_targets)

    def forward(self, x, features):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, (hn_lstm2, cn_lstm2) = self.lstm2(h_lstm1)
        # print(h_lstm2[:, 0, :][:2, 128:130], '====', hn_lstm2[1][:2, :2])

        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)
        # last hidden state
        hn_lstm2 = hn_lstm2.view(-1, self.lstm_units * 2)

        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))
        # h_conc_linear2 = F.relu(self.linear2(h_conc))
        hidden = F.relu(self.linear2(h_conc_linear1))
        # hidden = torch.cat((hidden, features), 1)
        # hidden = h_conc + h_conc_linear1 + h_conc_linear2

        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)

        return out
