from torch.autograd import Variable
import torch
import numpy as np
import math


def NLL_loss(mu: Variable, sigma: Variable, labels: Variable):
    distribution = torch.distributions.normal.Normal(mu, sigma)
    likelihood = distribution.log_prob(labels)
    return -torch.mean(likelihood)


# if relative is set to True, metrics are not normalized by the scale of labels
def accuracy_ND(mu: torch.Tensor, labels: torch.Tensor, relative=False):
    """
    ND: (1-st order) Norm Distance
    norm(x) = |x|.
    norm_distance(x, y) = |x - y|
    """
    # TODO: remove these zero_indexes as well? The zero_index makes it so that
    # the metrics dont consider z_t's that are padded 0 with 0's

    # zero_index = labels != 0
    # if relative:
    #     diff = torch.mean(torch.abs(mu[zero_index] - labels[zero_index])).item()
    #     return [diff, 1]
    # else:
    #     diff = torch.sum(torch.abs(mu[zero_index] - labels[zero_index])).item()
    #     summation = torch.sum(torch.abs(labels[zero_index])).item()
    #     return [diff, summation]

    if relative:
        diff = torch.mean(torch.abs(mu - labels)).item()
        return [diff, 1]
    else:
        diff = torch.sum(torch.abs(mu - labels)).item()
        summation = torch.sum(torch.abs(labels)).item()
        return [diff, summation]


def accuracy_RMSE(mu: torch.Tensor, labels: torch.Tensor, relative=False):
    """
    RMSE: Root Mean Square Error

    Return: 
    relative=True:  square erorr sum, number of points, num of points
    relative=False: square error sum, total label sum (normalization), num of points
    """
    # zero_index = labels != 0
    # diff = torch.sum(torch.mul((mu[zero_index] - labels[zero_index]), (mu[zero_index] - labels[zero_index]))).item()
    # if relative:
    #     return [diff, torch.sum(zero_index).item(), torch.sum(zero_index).item()]
    # else:
    #     summation = torch.sum(torch.abs(labels[zero_index])).item()
    #     if summation == 0:
    #         print("summation denominator error! ")
    #     return [diff, summation, torch.sum(zero_index).item()]

    # mu: (B, T)
    diff = torch.sum(torch.mul((mu - labels), (mu - labels))).item()
    N = labels.numel()
    if relative:
        return [diff, N, N]
    else:
        summation = torch.sum(torch.abs(labels)).item()
        if summation == 0:
            print("summation denominator error! ")
        return [diff, summation, N]


def accuracy_ROU(rou: float, samples: torch.Tensor, labels: torch.Tensor, relative=False):
    numerator = 0
    denominator = 0
    pred_samples = samples.shape[0]
    for t in range(labels.shape[1]):
        rou_th = math.ceil(pred_samples * (1 - rou))
        rou_pred = torch.topk(samples[:, :, t], dim=0, k=rou_th)[0][-1, :]
        abs_diff = labels[:, t][:] - rou_pred
        numerator += (
            2
            * (
                torch.sum(rou * abs_diff[labels[:, t][:] > rou_pred])
                - torch.sum((1 - rou) * abs_diff[labels[:, t][:] <= rou_pred])
            ).item()
        )
        denominator += torch.sum(labels[:, t][:]).item()
    if relative:
        return [numerator, torch.sum(labels != 0).item()]
    else:
        return [numerator, denominator]


def accuracy_ND_(mu: torch.Tensor, labels: torch.Tensor, relative=False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mu[labels == 0] = 0.0

    diff = np.sum(np.abs(mu - labels), axis=1)
    if relative:
        summation = np.sum((labels != 0), axis=1)
        mask = summation == 0
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result
    else:
        summation = np.sum(np.abs(labels), axis=1)
        mask = summation == 0
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result


def accuracy_RMSE_(mu: torch.Tensor, labels: torch.Tensor, relative=False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = labels == 0
    mu[mask] = 0.0

    diff = np.sum((mu - labels) ** 2, axis=1)
    summation = np.sum(np.abs(labels), axis=1)
    mask2 = summation == 0
    if relative:
        div = np.sum(~mask, axis=1)
        div[mask2] = 1
        result = np.sqrt(diff / div)
        result[mask2] = -1
        return result
    else:
        summation[mask2] = 1
        result = (np.sqrt(diff) / summation) * np.sqrt(np.sum(~mask, axis=1))
        result[mask2] = -1
        return result


def accuracy_ROU_(rou: float, samples: torch.Tensor, labels: torch.Tensor, relative=False):
    samples = samples.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = labels == 0
    samples[:, mask] = 0.0

    pred_samples = samples.shape[0]
    rou_th = math.floor(pred_samples * rou)

    samples = np.sort(samples, axis=0)
    rou_pred = samples[rou_th]

    abs_diff = np.abs(labels - rou_pred)
    abs_diff_1 = abs_diff.copy()
    abs_diff_1[labels < rou_pred] = 0.0
    abs_diff_2 = abs_diff.copy()
    abs_diff_2[labels >= rou_pred] = 0.0

    numerator = 2 * (rou * np.sum(abs_diff_1, axis=1) + (1 - rou) * np.sum(abs_diff_2, axis=1))
    denominator = np.sum(labels, axis=1)

    mask2 = denominator == 0
    denominator[mask2] = 1
    result = numerator / denominator
    result[mask2] = -1
    return result
