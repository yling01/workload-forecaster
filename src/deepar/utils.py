import json
import numpy as np
import loss


class Params:
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```

    Note you cannot read from non-existing parameter. But you can create a new
    one easily. Example:
    ```
    params.new_attr = 'foo'        # Ok
    print(params.new_attr)         # foo
    print(params.nonexisting_attr) # Not ok
    ```
    """

    def __init__(self, json_path=None):
        if json_path is not None:
            with open(json_path) as f:
                params = json.load(f)
                self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by params.dict['learning_rate']"""
        return self.__dict__


def init_metrics(sample=True):
    metrics = {
        "ND": np.zeros(2),  # numerator, denominator
        "RMSE": np.zeros(3),  # numerator, denominator, time step count
        "test_loss": np.zeros(2),
        "test_pred_loss": np.zeros(2),
    }
    if sample:
        metrics["rou90"] = np.zeros(2)
        metrics["rou50"] = np.zeros(2)
    return metrics


def get_metrics(sample_mu, labels, predict_start, samples=None, relative=False):
    # Only used for plotting
    metric = dict()
    metric["ND"] = loss.accuracy_ND_(sample_mu, labels[:, predict_start:], relative=relative)
    metric["RMSE"] = loss.accuracy_RMSE_(sample_mu, labels[:, predict_start:], relative=relative)
    if samples is not None:
        metric["rou90"] = loss.accuracy_ROU_(0.9, samples, labels[:, predict_start:], relative=relative)
        metric["rou50"] = loss.accuracy_ROU_(0.5, samples, labels[:, predict_start:], relative=relative)
    return metric


def update_metrics(
    raw_metrics, input_mu, input_sigma, sample_mu, sample_sigma, labels, predict_start, samples=None, relative=False
):
    raw_metrics["ND"] = raw_metrics["ND"] + loss.accuracy_ND(sample_mu, labels[:, predict_start:], relative=relative)
    raw_metrics["RMSE"] = raw_metrics["RMSE"] + loss.accuracy_RMSE(
        sample_mu, labels[:, predict_start:], relative=relative
    )

    if input_mu is not None:
        input_time_steps = input_mu.numel()  # numel: total number of elements
        raw_metrics["test_loss"] = raw_metrics["test_loss"] + [
            loss.NLL_loss(input_mu, input_sigma, labels[:, :predict_start]) * input_time_steps,
            input_time_steps,
        ]

    sample_time_steps = sample_mu.numel()
    sample_sigma += 1e-9  # To ensure that sample_sigma > 0
    raw_metrics["test_pred_loss"] = raw_metrics["test_pred_loss"] + [
        loss.NLL_loss(sample_mu, sample_sigma, labels[:, predict_start:]) * sample_time_steps,
        sample_time_steps,
    ]

    if samples is not None:
        raw_metrics["rou90"] = raw_metrics["rou90"] + loss.accuracy_ROU(
            0.9, samples, labels[:, predict_start:], relative=relative
        )
        raw_metrics["rou50"] = raw_metrics["rou50"] + loss.accuracy_ROU(
            0.5, samples, labels[:, predict_start:], relative=relative
        )
    return raw_metrics


def final_metrics(raw_metrics, sampling=False):
    summary_metric = {}
    summary_metric["ND"] = raw_metrics["ND"][0] / raw_metrics["ND"][1]
    summary_metric["RMSE"] = np.sqrt(raw_metrics["RMSE"][0] / raw_metrics["RMSE"][2]) / (
        raw_metrics["RMSE"][1] / raw_metrics["RMSE"][2]
    )
    if "test_loss" in raw_metrics.keys():
        summary_metric["test_loss"] = (raw_metrics["test_loss"][0] / raw_metrics["test_loss"][1]).item()
    summary_metric["test_pred_loss"] = (raw_metrics["test_pred_loss"][0] / raw_metrics["test_pred_loss"][1]).item()
    if sampling:
        summary_metric["rou90"] = raw_metrics["rou90"][0] / raw_metrics["rou90"][1]
        summary_metric["rou50"] = raw_metrics["rou50"][0] / raw_metrics["rou50"][1]
    return summary_metric
