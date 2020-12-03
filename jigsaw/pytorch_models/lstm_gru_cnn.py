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


class LstmGruCnn(nn.Module):

    def __init__(self, embedding_matrix, max_features, num_aux_targets, lstm_units, dense_units1, dense_units2,
                 dense_units3, dropout_rate):
        super(LstmGruCnn, self).__init__()
        embed_size = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(dropout_rate)

        self.max_pool = nn.MaxPool1d(2)
        self.avg_pool = nn.AvgPool1d(2)

        self.lstm1 = nn.GRU(embed_size, lstm_units, bidirectional=True, batch_first=True)
        self.conv1 = nn.Conv1d(lstm_units * 2 , 128, kernel_size=2)
        self.lstm2 = nn.LSTM(embed_size, lstm_units, bidirectional=True, batch_first=True)
        self.conv2 = nn.Conv1d(lstm_units * 2 , 128, kernel_size=2)

        self.linear1 = nn.Linear(dense_units1, dense_units2)
        self.linear2 = nn.Linear(dense_units2, dense_units3)

        self.linear_out = nn.Linear(dense_units3, 1)
        self.linear_aux_out = nn.Linear(dense_units3, num_aux_targets)

    def forward(self, x, features):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm1 = h_lstm1.transpose(1, 2)
        cnn1 = self.conv1(h_lstm1)
        h_lstm2, _ = self.lstm2(h_embedding)
        h_lstm2 = h_lstm2.transpose(1, 2)
        cnn2 = self.conv2(h_lstm2)

        avg_pool1 = nn.AvgPool1d(cnn1.size()[-1])(cnn1)
        max_pool1 = nn.MaxPool1d(cnn1.size()[-1])(cnn1)
        avg_pool2 = nn.AvgPool1d(cnn2.size()[-1])(cnn2)
        max_pool2 = nn.MaxPool1d(cnn2.size()[-1])(cnn2)

        x = torch.cat((max_pool1, avg_pool1, max_pool2, avg_pool2), 1)
        x = x.view(-1, 512)
        linear1 = F.relu(self.linear1(x))
        linear2 = F.relu(self.linear2(linear1))
        result = self.linear_out(linear2)
        aux_result = self.linear_aux_out(linear2)
        out = torch.cat([result, aux_result], 1)

        return out
