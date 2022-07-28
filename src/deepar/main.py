import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
import numpy as np
from tqdm import tqdm
import os

import utils
import loss
import plot
from dataset import TrainDataset, TestDataset, WeightedSampler
from models.modelv1 import DeepARV1
from models.modelv2 import DeepARV2

"""
Run on colab:
python main.py \
--data_dir /content/785-project/data/1year \
--data_name 1year \
--colab true \
--ckpt_dir /content/drive/MyDrive/785_project/checkpoints/ \
--plot_dir /content/drive/MyDrive/785_project/plots/

Run locally:
python main.py \
--data_dir ../../data/2017_01 \
--data_name 2017_01 \
--colab false \
--ckpt_dir _ \
--plot_dir __

"""

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
parser.add_argument("--data_name", required=True)
parser.add_argument("--colab", required=True, choices=["true", "false"], help="if true, log metrics and save models")
parser.add_argument("--ckpt_dir", required=True, type=str)
parser.add_argument("--plot_dir", required=True, type=str)


cuda_available = torch.cuda.is_available()
device = torch.device("cuda") if cuda_available else torch.device("cpu")
num_workers = 2 if cuda_available else 0

data_dir = "../../data/1year"
data_name = "1year"

params = utils.Params()

# TODO: fix this hacking.
# Make sure params.predict_start + params.predict_stpes == params.train_window.
# Make sure the parameters are consistent in preprocess.py and main.py.
# preprocess.py     main.py
# window_size       params.train_window, params.test_window
# stride_size       params.predict_steps
params.device = device
params.cov_dim = 4
params.predict_start = params.test_predict_start = 112
params.predict_steps = 16
params.test_window = params.train_window = 128
params.batch_size = 64
assert params.predict_start + params.predict_steps == params.train_window
assert params.test_predict_start + params.predict_steps == params.test_window

params.num_epochs = 30
params.learning_rate = 1e-3
params.lstm_dropout = 0.1
params.model_version = "v2"
params.model_type = "basic"
params.lstm_hidden_dim = 512
params.lstm_layers = 4
params.embedding_dim = 32
params.relative_metrics = True
params.sample_times = 10

params.teacher_forcing = False
params.conditioning_range_loss = False


def save_checkpoint(ckpt_path, id, model, epoch, optimizer=None, scheduler=None):
    path = os.path.join(ckpt_path, f"{id}_{epoch}")

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    save_dict = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "id": id,
    }

    if optimizer != None:
        save_dict["optimizer_state"] = optimizer.state_dict()
    if scheduler != None:
        save_dict["scheduler_state"] = scheduler.state_dict()

    torch.save(save_dict, path)
    print(f"=> saved model to {path}")


def train(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, criterion, tf_rate):
    model.train()

    batch_bar = tqdm(total=len(loader), dynamic_ncols=True, leave=False, position=0, desc="Train")

    running_loss = 0
    for i, (zxs, series_ids, labels) in enumerate(loader):
        optimizer.zero_grad()

        B, T = zxs.shape[0], zxs.shape[1]

        zxs = zxs.permute(1, 0, 2).float().to(device)  # (B, T, 1+cov_dim) -> (T, B, 1+cov_dim)
        series_ids = series_ids.unsqueeze(0).to(device)  # (B, 1) -> (1, B, 1)
        labels = labels.permute(1, 0).float().to(device)  # (B, T) -> (T, B)

        if isinstance(model, DeepARV1):
            loss = 0
            hidden, cell = model.init_hidden(B), model.init_cell(B)
            for t in range(params.train_window):
                # if z_t is missing, replace it by mu from previous timestep
                # We don't need this - a zero is a zero!
                # zero_index = zxs[t, :, 0] == 0  # index where z_t == 0
                # if t > 0 and torch.sum(zero_index) > 0:
                #     zxs[t, zero_index, 0] = mu[zero_index]

                mu, sigma, hidden, cell = model(zxs[t].unsqueeze(0), series_ids, hidden, cell)

                # Use `mu` as input for next timestep, if not teacher forcing
                if params.teacher_forcing and np.random.random() >= tf_rate and t < params.train_window - 1:
                    zxs[t + 1, :, 0] = mu

                if t >= params.predict_start or params.conditioning_range_loss:
                    loss += criterion(mu, sigma, labels[t])
        elif isinstance(model, DeepARV2):
            mu, sigma = model(zxs, series_ids)
            loss = criterion(mu, sigma, labels.unsqueeze(2))

        loss.backward()
        optimizer.step()

        running_loss += loss.item() / params.train_window

        batch_bar.set_postfix(loss="{:.04f}".format(running_loss / (i + 1)))
        batch_bar.update()

    batch_bar.close()
    return running_loss / len(loader)


def eval(model: nn.Module, loader: DataLoader, criterion, epoch, sampling=True, save_plot=False, plot_dir=None):
    model.eval()

    batch_bar = tqdm(total=len(loader), dynamic_ncols=True, leave=False, position=0, desc="Eval")

    if save_plot:
        assert plot_dir is not None

    with torch.no_grad():
        raw_metrics = utils.init_metrics(sample=sampling)

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
                # if z_t is missing, replace it by mu from previous timestep
                # We don't need this - a zero is a zero!
                # zero_index = zxs[t, :, 0] == 0  # index where z_t == 0
                # if t > 0 and torch.sum(zero_index) > 0:
                #     zxs[t, zero_index, 0] = mu[zero_index]

                mu, sigma, hidden, cell = model(zxs[t].unsqueeze(0), series_ids, hidden, cell)

                # scale
                scaled_mu[:, t] = vs[:, 0] * mu + vs[:, 1]
                scaled_sigma[:, t] = (
                    vs[:, 0] * sigma
                ) + 1e-5  # TODO: This is a hack to ensure scaled_sigma is not zero

            # generate predictions and metrics
            if sampling:
                # samples: (sample times, B, predict steps)
                samples, pred_mu, pred_sigma = model.test(zxs, vs, series_ids, hidden, cell, sampling=True)
                raw_metrics = utils.update_metrics(
                    raw_metrics,
                    scaled_mu.cpu(),
                    scaled_sigma.cpu(),
                    pred_mu.cpu(),
                    pred_sigma.cpu(),
                    labels.cpu(),
                    params.predict_start,
                    samples.cpu(),
                    relative=params.relative_metrics,
                )
            else:
                pred_mu, pred_sigma = model.test(zxs, vs, series_ids, hidden, cell, sampling=False)
                raw_metrics = utils.update_metrics(
                    raw_metrics,
                    scaled_mu.cpu(),
                    scaled_sigma.cpu(),
                    pred_mu.cpu(),
                    pred_sigma.cpu(),
                    labels.cpu(),
                    params.predict_start,
                    relative=params.relative_metrics,
                )

            batch_bar.update()

            # Use len(loader)-2 to ensure a full batch of at least 10 datapoints
            if save_plot and i == len(loader) - 2:
                if sampling:
                    sample_metrics = utils.get_metrics(
                        pred_mu, labels, params.predict_start, samples, relative=params.relative_metrics
                    )
                else:
                    sample_metrics = utils.get_metrics(
                        pred_mu, labels, params.test_predict_start, relative=params.relative_metrics
                    )

                # select 10 from samples with highest error and 10 from the rest
                top_10_nd_sample = (-sample_metrics["ND"]).argsort()[: B // 10]  # hard coded to be 10
                chosen = set(top_10_nd_sample.tolist())
                all_samples = set(range(B))
                not_chosen = np.asarray(list(all_samples - chosen))
                if B < 100:  # make sure there are enough unique samples to choose top 10 from
                    random_sample_10 = np.random.choice(top_10_nd_sample, size=10, replace=True)
                else:
                    random_sample_10 = np.random.choice(top_10_nd_sample, size=10, replace=False)
                if B < 12:  # make sure there are enough unique samples to choose bottom 90 from
                    random_sample_90 = np.random.choice(not_chosen, size=10, replace=True)
                else:
                    random_sample_90 = np.random.choice(not_chosen, size=10, replace=False)
                combined_sample = np.concatenate((random_sample_10, random_sample_90))

                label_plot = labels[combined_sample].data.cpu().numpy()
                predict_mu = pred_mu[combined_sample].data.cpu().numpy()
                predict_sigma = pred_sigma[combined_sample].data.cpu().numpy()
                plot_mu = np.concatenate((scaled_mu[combined_sample].data.cpu().numpy(), predict_mu), axis=1)
                plot_sigma = np.concatenate((scaled_mu[combined_sample].data.cpu().numpy(), predict_sigma), axis=1)
                plot_metrics = {_k: _v[combined_sample] for _k, _v in sample_metrics.items()}
                plot.plot_eight_windows(
                    plot_dir,
                    plot_mu,
                    plot_sigma,
                    label_plot,
                    params.test_window,
                    params.test_predict_start,
                    epoch,
                    plot_metrics,
                    False,
                )

    batch_bar.close()
    summary_metric = utils.final_metrics(raw_metrics, sampling=sampling)
    return summary_metric


def main():
    args = parser.parse_args()
    args.colab = True if args.colab == "true" else False
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

    train_set = TrainDataset(args.data_dir, args.data_name)
    test_set = TestDataset(args.data_dir, args.data_name)
    sampler = WeightedSampler(args.data_dir, args.data_name)

    train_loader = DataLoader(train_set, batch_size=params.batch_size, sampler=sampler, num_workers=num_workers)
    test_loader = DataLoader(
        test_set, batch_size=params.batch_size, sampler=RandomSampler(test_set), num_workers=num_workers
    )

    if params.model_version == "v1":
        model = DeepARV1(params, train_set.num_classes()).to(device)
    elif params.model_version == "v2":
        model = DeepARV2(params, train_set.num_classes()).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=1, verbose=True, threshold=1e-2)
    criterion = loss.NLL_loss

    if args.colab:
        import wandb

        os.environ["WANDB_API_KEY"] = "f48e810b0a53cfababaa9de2f318850fe8e4e787"

        experiment_id = (
            f"model[{params.model_version}]"
            f"_lstm[{params.lstm_layers}_{params.lstm_hidden_dim}_{params.lstm_dropout}]"
            f"_embed[{params.embedding_dim}]"
            f"_sample[{params.sample_times}]"
            f"_lr[{params.learning_rate}]"
        )

        # Wandb project: https://wandb.ai/yy0125/785_project
        wandb.init(project="785_project", name=experiment_id, config=params.dict)

    print(params.dict)
    print(model)

    best_rmse = float("inf")
    tf_rate = 1.05
    for epoch in range(params.num_epochs):
        if epoch % 5 == 0 and params.teacher_forcing:
            tf_rate = max(tf_rate - 0.05, 0.7)

        train_loss = train(model, train_loader, optimizer, criterion, tf_rate)

        eval_metrics = eval(
            model, test_loader, criterion, epoch, sampling=True, save_plot=True, plot_dir=args.plot_dir
        )
        test_loss, test_pred_loss, ND, RMSE, rou50, rou90 = (
            eval_metrics["test_loss"],
            eval_metrics["test_pred_loss"],
            eval_metrics["ND"],
            eval_metrics["RMSE"],
            eval_metrics["rou50"],
            eval_metrics["rou90"],
        )

        print(
            f"Epoch {epoch}/{params.num_epochs}: Train loss {train_loss:.04f}, lr {optimizer.param_groups[0]['lr']:.04f}, tf_rate {tf_rate}"
        )
        print(
            f"Test loss: {test_loss:.04f}, test_pred_loss: {test_pred_loss:.04f}, ND: {ND:.04f}, RMSE: {RMSE:.04f}, rou50: {rou50:.04f}, rou90: {rou90:.04f}"
        )

        scheduler.step(RMSE)

        if args.colab:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "test_pred_loss": test_pred_loss,
                    "test_loss": test_loss,
                    "ND": ND,
                    "RMSE": RMSE,
                    "rou50": rou50,
                    "rou90": rou90,
                    "lr": optimizer.param_groups[0]["lr"],
                    "tf_rate": tf_rate,
                }
            )

            if RMSE < best_rmse:
                save_checkpoint(args.ckpt_dir, experiment_id, model, epoch, optimizer, scheduler)
                best_rmse = RMSE

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
