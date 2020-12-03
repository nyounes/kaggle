import torch
from torch import nn
from torch.nn import functional as F


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class Cnn(nn.Module):

    def __init__(self, embedding_matrix, max_features, num_aux_targets, kernel_num=128, kernel_sizes=[1, 2, 3, 4],
                 dense_units1=512, dense_units2=128):
        super(Cnn, self).__init__()

        embed_size = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        self.max_pool = nn.MaxPool1d(2, stride=2)

        self.conv1_3 = nn.Conv1d(embed_size, kernel_num, kernel_size=1, padding=1)
        self.conv1_4 = nn.Conv1d(embed_size, kernel_num, kernel_size=2, padding=2)
        self.conv1_5 = nn.Conv1d(embed_size, kernel_num, kernel_size=3, padding=2)
        self.conv1_6 = nn.Conv1d(embed_size, kernel_num, kernel_size=4, padding=3)

        self.conv2_2 = nn.Conv1d(kernel_num * 4, kernel_num * 2, kernel_size=2)
        self.conv2_3 = nn.Conv1d(kernel_num * 4, kernel_num * 2, kernel_size=3, padding=1)

        self.conv3_2 = nn.Conv1d(kernel_num * 4, kernel_num * 4, kernel_size=2)
        self.conv3_3 = nn.Conv1d(kernel_num * 4, kernel_num * 4, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv1d(kernel_num * 8, kernel_num * 8, kernel_size=1)

        self.dropout = nn.Dropout(0.6)

        self.linear1 = nn.Linear(kernel_num * 8, dense_units1)
        self.linear2 = nn.Linear(dense_units1, dense_units2)

        self.linear_out = nn.Linear(dense_units2, 1)
        self.linear_aux_out = nn.Linear(dense_units2, num_aux_targets)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, features):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)
        x = h_embedding.transpose(1, 2)

        x1_3 = F.relu(self.conv1_3(x))
        x1_3 = self.max_pool(x1_3)
        x1_4 = F.relu(self.conv1_4(x))
        x1_4 = self.max_pool(x1_4)
        x1_5 = F.relu(self.conv1_5(x))
        x1_5 = self.max_pool(x1_5)
        x1_6 = F.relu(self.conv1_6(x))
        x1_6 = self.max_pool(x1_6)
        x1 = torch.cat([x1_3, x1_4, x1_5, x1_6], 1)
        x1 = x1.squeeze()

        x2_2 = F.relu(self.conv2_2(x1))
        x2_2 = self.max_pool(x2_2)
        x2_3 = F.relu(self.conv2_3(x1))
        x2_3 = self.max_pool(x2_3)
        x2 = torch.cat([x2_2, x2_3], 1)

        x3_2 = F.relu(self.conv3_2(x2))
        x3_2 = nn.MaxPool1d(x3_2.size()[-1])(x3_2)
        x3_3 = F.relu(self.conv3_3(x2))
        x3_3 = nn.MaxPool1d(x3_3.size()[-1])(x3_3)
        x3 = torch.cat([x3_2, x3_3], 1)
        x3 = x3.view(-1, 1024)

        linear1 = F.relu(self.linear1(x3))
        linear2 = F.relu(self.linear2(linear1))
        result = self.linear_out(linear2)
        aux_result = self.linear_aux_out(linear2)
        out = torch.cat([result, aux_result], 1)

        return out
