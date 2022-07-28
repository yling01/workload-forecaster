import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

import utils
import loss_attention
from models.modelv1 import DeepARV1
from dataset import TrainDataset, TestDataset, WeightedSampler

"""
1. The params object needs to be the same as the model you are loading.

2. Run locally
python generate_prediction.py \
--data_dir ../../data/1year \
--data_name 1year \
--ckpt_path ./model_checkpoints/v4/v4_0

3. Run on colab

% cd 785-project/src/deepar/
!python generate_prediction.py \
--data_dir ../../data/1year \
--data_name 1year \
--ckpt_path ./model_checkpoints/v4/v4_0

"""

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
parser.add_argument("--data_name", required=True)
parser.add_argument("--ckpt_path", required=True, type=str)


cuda_available = torch.cuda.is_available()
device = torch.device("cuda") if cuda_available else torch.device("cpu")
num_workers = 2 if cuda_available else 0

save_dir = "./prediction/"

params = utils.Params()

params.device = device
params.cov_dim = 4
params.predict_start = params.test_predict_start = 112
params.predict_steps = 16
params.test_window = params.train_window = 128
params.batch_size = 64
assert params.predict_start + params.predict_steps == params.train_window
assert params.test_predict_start + params.predict_steps == params.test_window

params.lstm_dropout = 0.1
params.lstm_hidden_dim = 512
params.lstm_layers = 4
params.embedding_dim = 32

params.model_type = "basic"

params.sampling = True
params.sample_times = 10


def eval(model: nn.Module, loader: DataLoader, sampling=True):
    model.eval()

    batch_bar = tqdm(total=len(loader), dynamic_ncols=True, leave=False, position=0, desc="Eval")

    predictions = []
    with torch.no_grad():
        for i, (zxs, series_ids, vs, labels) in enumerate(loader):
            B = zxs.shape[0]

            zxs = zxs.permute(1, 0, 2).float().to(device)  # (B, T, 1+cov_dim) -> (T, B, 1+cov_dim)
            series_ids = series_ids.unsqueeze(0).to(device)  # (B, 1)
            vs = vs.to(device)  # (B, 1)
            labels = labels.float().to(device)  # (B, T)

            scaled_mu = torch.zeros(B, params.predict_start, device=device)
            scaled_sigma = torch.zeros(B, params.predict_start, device=device)
            hidden, cell = model.init_hidden(B), model.init_cell(B)

            # processing given data
            for t in range(params.predict_start):
                mu, sigma, hidden, cell = model(zxs[t].unsqueeze(0), series_ids, hidden, cell)

                # scale
                scaled_mu[:, t] = vs[:, 0] * mu + vs[:, 1]
                scaled_sigma[:, t] = (
                    vs[:, 0] * sigma
                ) + 1e-5  # TODO: This is a hack to ensure scaled_sigma is not zero

            # generate predictions and metrics
            if sampling:
                samples, pred_mu, pred_sigma = model.test(zxs, vs, series_ids, hidden, cell, sampling=True)
            else:
                pred_mu, pred_sigma = model.test(zxs, vs, series_ids, hidden, cell, sampling=False)

            pred_mu = pred_mu.unsqueeze(2)  # (B, 16, 1)
            predictions.append(pred_mu)
            batch_bar.update()

    batch_bar.close()

    predictions = torch.cat(predictions, dim=0)  # (num_data_points, 16, 1)
    return predictions.cpu().detach().numpy()


def main():
    args = parser.parse_args()

    train_set = TrainDataset(args.data_dir, args.data_name)
    test_set = TestDataset(args.data_dir, args.data_name)

    test_loader = DataLoader(test_set, batch_size=params.batch_size, num_workers=num_workers)

    model = DeepARV1(params, train_set.num_classes()).to(device)

    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt["model_state"], strict=False)

    save_path = os.path.join(save_dir, f"prediction.npy")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    predictions = eval(model, test_loader, sampling=params.sampling)

    with open(save_path, "wb") as f:
        np.save(f, predictions)


if __name__ == "__main__":
    main()
    actual = np.load("../../data/1year/test_label_1year.npy")
    print(actual[0, -16:-14])
    print(actual[64, -16:-14])

    print("TARGET")
    save_path = os.path.join(save_dir, f"prediction.npy")
    target = np.load(save_path)
    print(target.shape)
    print(target[0, :2, 0])
    print(target[64, :2, 0])

