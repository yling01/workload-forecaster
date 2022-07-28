import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
import numpy as np
from tqdm import tqdm
import os

import utils_attention
import loss_attention
from models.cross_attention import DeepARSeq2Seq
from models.cross_attention_no_blstm import DeepARSeq2Seq as DeepARSeq2SeqNoBLSTM
from dataset import TrainDataset, TestDataset, WeightedSampler
import matplotlib.pyplot as plt
import seaborn as sns

"""
Run on colab:
python main.py \
--data_dir /content/785-project/data/1year \
--data_name 1year \
--colab true \
--ckpt_dir /content/drive/MyDrive/785_project/checkpoints/ \
--plot_dir /content/drive/MyDrive/785_project/plots/

Run locally:
python main_attention.py \
--data_dir ../../data/2017_01 \
--data_name 2017_01 \
--colab false \
--ckpt_dir ./model_checkpoints/v3 \
--plot_dir ./plots/v3

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

params = utils_attention.Params()

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

params.num_epochs = 100
params.learning_rate = 1e-3
params.lstm_dropout = 0.2
params.model_type = "cross_attention_v9"
params.lstm_hidden_dim = 128
params.plstm_layers = 2
params.lstm_layers = 2
params.cluster_embedding_dim = 128
params.relative_metrics = True
params.sample_times = 10
params.sampling = False

# Attention
params.key_value_size = 128
params.num_heads = 2
params.attention_dropout = 0.2
params.teacher_forcing = False


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


def plot_attention(attention):
    # utility function for debugging
    plt.clf()
    sns.heatmap(attention, cmap="GnBu")
    plt.show()


def plot_eight_windows(plot_dir, predict_values, predict_sigma, labels, plot_num):

    x = np.arange(params.predict_steps)
    f = plt.figure(figsize=(8, 42), constrained_layout=True)
    nrows = 20
    ncols = 1
    ax = f.subplots(nrows, ncols)

    for k in range(nrows):
        ax[k].plot(x, predict_values[k], color="b")
        ax[k].fill_between(
            x,
            predict_values[k, :] - 2 * predict_sigma[k, :],
            predict_values[k, :] + 2 * predict_sigma[k, :],
            color="blue",
            alpha=0.2,
        )
        ax[k].plot(x, labels[k, params.predict_start :], color="r")

        # plot_metrics_str = f'ND: {plot_metrics["ND"][k]: .3f} ' f'RMSE: {plot_metrics["RMSE"][k]: .3f}'
        # ax[k].set_title(plot_metrics_str, fontsize=10)

    f.savefig(os.path.join(plot_dir, str(plot_num) + ".png"))
    # plt.show()
    plt.close()


def train(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, criterion):
    model.train()

    batch_bar = tqdm(total=len(loader), dynamic_ncols=True, leave=False, position=0, desc="Train")

    running_loss = 0
    # attentions_mean = None
    for i, (zxs, series_ids, labels) in enumerate(loader):
        optimizer.zero_grad()

        zxs = zxs.float().to(device)  # (B, T, 1+cov_dim)
        series_ids = series_ids.to(device)  # (1, B)
        labels = labels.float().to(device)  # (B, T)

        pred_mu, pred_sigma, attentions = model(zxs, series_ids)  # (B, 16)
        # if attentions_mean == None:
        #     attentions_mean = attentions
        # else:
        #     attentions_mean += attentions
        pred_sigma += 1e-5
        loss = criterion(pred_mu, pred_sigma, labels[:, params.predict_start :])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()

        running_loss += loss.item() / params.predict_steps

        batch_bar.set_postfix(loss="{:.04f}".format(running_loss / (i + 1)))
        batch_bar.update()

    # plot_attention(attentions_mean / len(loader))
    plot_attention(attentions)

    batch_bar.close()
    return running_loss / len(loader)


def eval_train(model: nn.Module, loader: DataLoader, criterion, epoch, sampling=True, save_plot=False, plot_dir=None):
    model.eval()

    batch_bar = tqdm(total=len(loader), dynamic_ncols=True, leave=False, position=0, desc="Eval")

    if save_plot:
        assert plot_dir is not None

    with torch.no_grad():
        raw_metrics = utils_attention.init_metrics(sample=sampling)

        for i, (zxs, series_ids, labels) in enumerate(loader):
            if i >= 64:
                break

            zxs = zxs.float().to(device)  # (B, T, 1+cov_dim) -> (T, B, 1+cov_dim)
            series_ids = series_ids.to(device)  # (B, 1)
            labels = labels.float().to(device)  # (B, T)

            pred_mu, pred_sigma, _ = model(zxs, series_ids)
            raw_metrics = utils_attention.update_metrics(
                raw_metrics,
                None,
                None,
                pred_mu.cpu(),
                pred_sigma.cpu(),
                labels.cpu(),
                params.predict_start,
                relative=params.relative_metrics,
            )

            batch_bar.update()

    # x = np.arange(0, params.predict_steps, 1)
    # plt.plot(x, pred_mu[0].cpu().detach().numpy(), label="Prediction")
    # plt.plot(x, labels[0, params.predict_start :], label="Actual")
    # plt.legend()
    # plt.show()

    plot_eight_windows(plot_dir, pred_mu.cpu(), pred_sigma.cpu(), labels.cpu(), 0)

    batch_bar.close()
    summary_metric = utils_attention.final_metrics(raw_metrics, False)
    return summary_metric


def eval(model: nn.Module, loader: DataLoader, criterion, epoch, sampling=True, save_plot=False, plot_dir=None):
    model.eval()

    batch_bar = tqdm(total=len(loader), dynamic_ncols=True, leave=False, position=0, desc="Eval")

    if save_plot:
        assert plot_dir is not None

    with torch.no_grad():
        raw_metrics = utils_attention.init_metrics(sample=sampling)

        for i, (zxs, series_ids, vs, labels) in enumerate(loader):

            zxs = zxs.float().to(device)  # (B, T, 1+cov_dim) -> (T, B, 1+cov_dim)
            series_ids = series_ids.to(device)  # (B, 1)
            vs = vs.to(device)  # (B, 2)
            labels = labels.float().to(device)  # (B, T)

            # generate predictions and metrics
            if sampling:
                # samples: (sample times, B, predict steps)
                samples, pred_mu, pred_sigma = model.test(zxs, vs, series_ids, sampling=True)
                raw_metrics = utils_attention.update_metrics(
                    raw_metrics,
                    None,
                    None,
                    pred_mu.cpu(),
                    pred_sigma.cpu(),
                    labels.cpu(),
                    params.predict_start,
                    samples.cpu(),
                    relative=params.relative_metrics,
                )
            else:
                _, pred_mu, pred_sigma = model.test(zxs, vs, series_ids, sampling=False)
                raw_metrics = utils_attention.update_metrics(
                    raw_metrics,
                    None,
                    None,
                    pred_mu.cpu(),
                    pred_sigma.cpu(),
                    labels.cpu(),
                    params.predict_start,
                    relative=params.relative_metrics,
                )

            batch_bar.update()

    batch_bar.close()
    summary_metric = utils_attention.final_metrics(raw_metrics, sampling=sampling)
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

    # model = DeepARSeq2Seq(params, train_set.num_classes()).to(device)
    model = DeepARSeq2SeqNoBLSTM(params, train_set.num_classes()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=1, verbose=True, threshold=1e-2)
    criterion = loss_attention.NLL_loss

    # ckpt = torch.load(
    #     "/content/drive/MyDrive/CMU-11685/project/models/v5_attention/model[cross_attention_v1]_lstm[3_128_0.1]_attention[64_2_0.1]_embed[32]_sample[10]_lr[0.001]_teacher_forcing[false]_0"
    # )
    # model.load_state_dict(ckpt["model_state"], strict=False)

    if args.colab:
        import wandb

        os.environ["WANDB_API_KEY"] = "f48e810b0a53cfababaa9de2f318850fe8e4e787"

        experiment_id = (
            f"model[{params.model_type}]"
            f"_lstm[{params.plstm_layers}_{params.lstm_hidden_dim}_{params.lstm_dropout}]"
            f"_attention[{params.key_value_size}_{params.num_heads}_{params.attention_dropout}]"
            f"_embed[{params.cluster_embedding_dim}]"
            f"_sample[{params.sample_times}]"
            f"_lr[{params.learning_rate}]"
            f"_teacher_forcing[false]"
        )

        # Wandb project: https://wandb.ai/yy0125/785_project
        wandb.init(project="785_project", name=experiment_id, config=params.dict)

    print(params.dict)
    print(model)

    best_rmse = float("inf")
    for epoch in range(params.num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        # train_loss = 0

        train_metrics = eval_train(
            model, train_loader, criterion, epoch, sampling=params.sampling, save_plot=True, plot_dir=args.plot_dir
        )

        train_pred_loss, train_ND, train_RMSE = (
            train_metrics["test_pred_loss"],
            train_metrics["ND"],
            train_metrics["RMSE"],
        )

        print(f"Train_pred_loss: {train_pred_loss:.04f}, ND: {train_ND:.04f}, RMSE: {train_RMSE:.04f}")

        eval_metrics = eval(
            model, test_loader, criterion, epoch, sampling=params.sampling, save_plot=True, plot_dir=args.plot_dir
        )

        if params.sampling:
            test_pred_loss, ND, RMSE, rou50, rou90 = (
                eval_metrics["test_pred_loss"],
                eval_metrics["ND"],
                eval_metrics["RMSE"],
                eval_metrics["rou50"],
                eval_metrics["rou90"],
            )

            print(
                f"Epoch {epoch}/{params.num_epochs}: Train loss {train_loss:.04f}, lr {optimizer.param_groups[0]['lr']:.04f}"
            )

            print(
                f"Test_pred_loss: {test_pred_loss:.04f}, ND: {ND:.04f}, RMSE: {RMSE:.04f}, rou50: {rou50:.04f}, rou90: {rou90:.04f}"
            )

            if args.colab:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "test_pred_loss": test_pred_loss,
                        "ND": ND,
                        "RMSE": RMSE,
                        "rou50": rou50,
                        "rou90": rou90,
                        "lr": optimizer.param_groups[0]["lr"],
                        "train_ND": train_ND,
                        "train_RMSE": train_RMSE,
                    }
                )

                # TODO: fix
                if train_loss < best_rmse:
                    save_checkpoint(args.ckpt_dir, experiment_id, model, epoch, optimizer, scheduler)
                    best_rmse = train_loss
        else:
            test_pred_loss, ND, RMSE = (
                eval_metrics["test_pred_loss"],
                eval_metrics["ND"],
                eval_metrics["RMSE"],
            )

            print(
                f"Epoch {epoch}/{params.num_epochs}: Train loss {train_loss:.04f}, lr {optimizer.param_groups[0]['lr']:.04f}"
            )
            print(f"Test_pred_loss: {test_pred_loss:.04f}, ND: {ND:.04f}, RMSE: {RMSE:.04f}")

            if args.colab:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "test_pred_loss": test_pred_loss,
                        "ND": ND,
                        "RMSE": RMSE,
                        "lr": optimizer.param_groups[0]["lr"],
                        "train_ND": train_ND,
                        "train_RMSE": train_RMSE,
                    }
                )

                # TODO: fix
                if train_loss < best_rmse:
                    save_checkpoint(args.ckpt_dir, experiment_id, model, epoch, optimizer, scheduler)
                    best_rmse = train_loss

        scheduler.step(RMSE)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
