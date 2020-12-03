import numpy as np


def calculate_loss_weight(train, columns):
    weights = np.ones((len(train),)) / 4
    # Subgroup
    weights += (train[columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int) / 4
    # Background Positive, Subgroup Negative
    weights += (((train['target'].values >= 0.5).astype(bool).astype(np.int) +
                (train[columns].fillna(0).values < 0.5).sum(
                    axis=1).astype(bool).astype(np.int)) > 1).astype(bool).astype(np.int) / 4
    # Background Negative, Subgroup Positive
    weights += (((train['target'].values < 0.5).astype(bool).astype(np.int) +
                (train[columns].fillna(0).values >= 0.5).sum(
                    axis=1).astype(bool).astype(np.int)) > 1).astype(bool).astype(np.int) / 4
    loss_weight = 1.0 / weights.mean()

    return weights, loss_weight
