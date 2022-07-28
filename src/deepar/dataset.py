import torch
import os
import numpy as np


class TrainDataset(torch.utils.data.Dataset):
    """
    self.data: (num_data_points, window_size, feature_size)
    features: [count, covariates (day/month), cluster_id]

    self.label: (num_data_points, window_size)
    Query count at next timestamp.
    self.data[i, j+1, 0] == self.label[i, j]

    output data:
    (window_size, feature_size), cluster_id, (window_size, )
    """

    def __init__(self, data_path, data_name):
        self.data = np.load(os.path.join(data_path, f"train_zx_{data_name}.npy"))
        self.label = np.load(os.path.join(data_path, f"train_label_{data_name}.npy"))
        self.train_len = self.data.shape[0]

    def num_classes(self):
        unique_series_ids = np.unique(self.data[:, 0, -1].astype(int))
        return len(unique_series_ids)

    def __len__(self):
        return self.train_len

    def __getitem__(self, idx):
        # self.data: (num_data_points, window_size, 1 + num_covariates + 1)
        # At the last dimension: [count, covariates..., cluster id]

        # Return format
        # [count, covariates], series id, label
        return self.data[idx, :, :-1], int(self.data[idx, 0, -1]), self.label[idx]


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, data_name):
        self.data = np.load(os.path.join(data_path, f"test_zx_{data_name}.npy"))
        self.v = np.load(os.path.join(data_path, f"test_v_{data_name}.npy"))
        self.label = np.load(os.path.join(data_path, f"test_label_{data_name}.npy"))
        self.test_len = self.data.shape[0]

    def __len__(self):
        return self.test_len

    def __getitem__(self, idx):
        # [count, covariates], series id, [scaling factors v1, v2], label
        return self.data[idx, :, :-1], int(self.data[idx, 0, -1]), self.v[idx], self.label[idx]


class WeightedSampler(torch.utils.data.Sampler):
    """
    self.weights: (num_data_points, )
    Sample each data point based on its weight
    """

    def __init__(self, data_path, data_name, replacement=True):
        v = np.load(os.path.join(data_path, f"train_v_{data_name}.npy"))
        self.weights = torch.as_tensor(np.abs(v[:, 0]) / np.sum(np.abs(v[:, 0])), dtype=torch.double)
        self.num_samples = self.weights.shape[0]
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples
