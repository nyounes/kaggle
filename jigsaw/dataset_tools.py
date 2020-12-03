from logger import logger
import numpy as np
import torch
from fastai.train import DataBunch

from dataset import MultipleInputDataset


def create_full_train_datasets(x_train, y_train, y_aux_train, train_features, x_test, test_features):
    logger.info("Building datasets for full training ...")
    x_train_torch = torch.tensor(x_train, dtype=torch.long)
    x_train_features_torch = torch.tensor(train_features, dtype=torch.float32)
    y_train_torch = torch.tensor(np.hstack([y_train, y_aux_train]), dtype=torch.float32)

    x_test_torch = torch.tensor(x_test, dtype=torch.long)
    x_test_features_torch = torch.tensor(test_features, dtype=torch.float32)

    batch_size = 512

    train_dataset = MultipleInputDataset(x_train_torch, x_train_features_torch, y_train_torch)
    valid_dataset = MultipleInputDataset(x_train_torch[: batch_size], x_train_features_torch[: batch_size],
                                         y_train_torch[: batch_size])
    test_dataset = MultipleInputDataset(x_test_torch, x_test_features_torch)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    databunch = DataBunch(train_dl=train_loader, valid_dl=valid_loader)
    logger.info("Building datasets for full training => done")

    return databunch, test_dataset


def create_cross_val_datasets(x_train, y_train, y_aux_train, train_features, train_index, val_index, batch_size,
                              x_test, test_features):

    logger.info("Building datasets for cross val ...")

    x_train = torch.tensor(x_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_aux_train = torch.tensor(y_aux_train, dtype=torch.float32)
    x_train = torch.tensor(x_train, dtype=torch.long)
    train_features = torch.tensor(train_features, dtype=torch.float32)
    y_train_all = np.hstack([y_train, y_aux_train])
    x_test = torch.tensor(x_test, dtype=torch.long)
    test_features = torch.tensor(test_features, dtype=torch.float32)

    train_dataset = MultipleInputDataset(x_train[train_index], train_features[train_index], y_train_all[train_index])
    fake_valid_dataset = MultipleInputDataset(x_train[: batch_size], train_features[: batch_size],
                                              y_train_all[: batch_size])
    val_dataset = MultipleInputDataset(x_train[val_index], train_features[val_index])
    test_dataset = MultipleInputDataset(x_test, test_features)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    fake_valid_loader = torch.utils.data.DataLoader(fake_valid_dataset, batch_size=batch_size, shuffle=False)

    databunch = DataBunch(train_dl=train_loader, valid_dl=fake_valid_loader)
    logger.info("Building datasets for cross val => done")

    return databunch, val_dataset, test_dataset
