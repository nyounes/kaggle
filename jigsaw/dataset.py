from torch.utils.data import Dataset


class MultipleInputDataset(Dataset):

    def __init__(self, x, features, labels=None):
        self.x = x
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.labels is not None:
            return [self.x[index], self.features[index]], self.labels[index]
        else:
            return [self.x[index], self.features[index]]





