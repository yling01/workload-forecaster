"""
This file contains model template and implementation for Forecaster.
All forecasting models should inherit from
ForecastModel, and override the _do_fit and _do_predict abstract methods
"""

import copy
import logging
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import glob
import pandas as pd
import torch
import re
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset import ForecastDataset
from pathlib import Path


class ForecastModel(ABC):
    """Interface ftor all the forecasting models"""

    def __init__(self, horizon, interval, sequence_length):
        self._x_transformer = None
        self._y_transformer = None

        self.horizon = horizon
        self.interval = interval
        self.sequence_length = sequence_length

    @property
    def name(self):
        return self.__class__.__name__

    def fit(self, train_seqs: ForecastDataset) -> None:
        """Fit the model with training sequences

        Parameters
        ----------
        train_seqs :
            List of training sequences and the expected output label
            in a certain horizon
        """

        # Make sure that the training data matches what the model expects
        assert self.horizon == train_seqs.horizon
        assert self.interval == train_seqs.interval
        assert self.sequence_length == train_seqs.sequence_length

        self._x_transformer, self._y_transformer = self._get_transformers(
            train_seqs.raw_df.values)

        transformed = copy.deepcopy(train_seqs)
        transformed.set_transformers(
            (self._x_transformer, self._y_transformer))

        self._do_fit(transformed)

    @abstractmethod
    def _do_fit(self, trains_seqs: ForecastDataset) -> None:
        """Perform fitting.
        Should be overloaded by a specific model implementation.

        Parameters
        ----------
        train_seqs:
            List of training sequences and the expected output label in a
            certain horizon. Normalization would have been done if needed
        """
        raise NotImplementedError("Should be implemented by child classes")

    def predict(self, test_seq: np.ndarray) -> float:
        """Test a fitted model with a sequence.
        Parameters
        ----------
        test_seq:
            1D Test sequence

        Returns
        -------
        Predicted value at certain horizon
        """

        if self._x_transformer:
            test_seq = self._x_transformer.transform(test_seq)

        predict = self._do_predict(test_seq)
        if self._y_transformer:
            # Get the predicted scalar value back
            predict = self._y_transformer.inverse_transform(
                np.array([predict]).reshape(1, -1))[0][0]

        return predict

    @abstractmethod
    def _do_predict(self, test_seq: np.ndarray) -> float:
        """Perform prediction given input sequence.
        Should be overloaded by a specific model implementation.
        Parameters
        ----------
        test_seq
            1D Test sequence

        Returns
        -------
        Predicted value at certain horizon
        """
        raise NotImplementedError("Should be implemented by child classes")

    @abstractmethod
    def _get_transformers(self, data: np.ndarray) -> Tuple:
        """Get the data transformers
        Parameters
        ----------
        data :
            Training data

        Returns
        -------
        A tuple of x and y transformers
        """
        raise NotImplementedError(
            "Each model should have its own transformers")

    def save(self, path):
        self._do_save(path)

    @abstractmethod
    def _do_save(self, path):
        raise NotImplementedError("Should be implemented by child classes")

    @staticmethod
    def load(path):
        raise NotImplementedError("Should be implemented by child classes")


class LSTM(nn.Module, ForecastModel):
    """A simple LSTM model serves as a template for ForecastModel"""

    def __init__(
            self,
            horizon: pd.Timedelta = pd.Timedelta(seconds=60),
            interval: pd.Timedelta = pd.Timedelta(seconds=10),
            sequence_length: int = 5,
            input_size: int = 1,
            hidden_layer_size: int = 100,
            num_hidden_layers: int = 1,
            output_size: int = 1,
            lr: float = 0.001,
            epochs: int = 20,  # note: this is how the number of epochs can be controlled
    ):
        """
        Parameters
        ----------
        input_size :
            Dimension of data point that is fed into the LSTM each time.
        hidden_layer_size :
            How many cells in one layer of the LSTM.
        num_hidden_layers :
            How many layers in the stacked LSTM.
        output_size :
            Dimension of prediction output.
        lr :
            Learning rate while fitting.
        epochs :
            Number of epochs for fitting.
        """
        nn.Module.__init__(self)
        ForecastModel.__init__(self, horizon, interval, sequence_length)

        self._hidden_layer_size = hidden_layer_size
        self._num_hidden_layers = num_hidden_layers

        self._lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_layer_size,
            num_layers=num_hidden_layers,
        )

        self._linear = nn.Linear(hidden_layer_size, output_size)

        self._hidden_cell = (
            torch.zeros(self._num_hidden_layers, 1, self._hidden_layer_size),
            torch.zeros(self._num_hidden_layers, 1, self._hidden_layer_size),
        )

        self._epochs = epochs
        self._lr = lr

    def forward(self, input_seq: torch.FloatTensor) -> float:
        """Forward propagation Implements nn.Module.forward().

        Parameters
        ----------
        input_seq : 1D FloatTensor

        Returns
        -------
        A single value prediction
        """
        lstm_out, self._hidden_cell = self._lstm(
            input_seq.view(len(input_seq), 1, -1), self._hidden_cell)
        predictions = self._linear(lstm_out.view(len(input_seq), -1))
        # note: only take one prediction at a time
        # todo: can use all the predictions
        return predictions[-1]

    def _do_fit(self, train_seqs: Dataset) -> None:
        """
        Perform fitting.
        Should be overloaded by a specific model implementation.

        Parameters
        ----------
        train_seqs:
            List of training sequences and the expected output label in a
            certain horizon.
            Normalization has been done in ForecastModel.fit() with
            x and y transformers.
        """
        epochs = self._epochs
        lr = self._lr

        # Training specifics
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        logging.info(
            f"Training with {len(train_seqs)} samples, {epochs} epochs:")
        print(f"Training with {len(train_seqs)} samples, {epochs} epochs:")
        for i in range(epochs):
            # randomly shuffle training sequences
            arr = np.arange(len(train_seqs))
            np.random.shuffle(arr)
            batch_bar = tqdm(total=len(arr), dynamic_ncols=True, leave=False, position=0, desc="Train")
            total_loss = 0
            for count, ind in enumerate(arr, start=1):
                seq, labels = train_seqs[ind]
                optimizer.zero_grad()

                self._hidden_cell = (
                    torch.zeros(self._num_hidden_layers, 1,
                                self._hidden_layer_size),
                    torch.zeros(self._num_hidden_layers, 1,
                                self._hidden_layer_size),
                )
                # todo: this is only generating one data point to train
                seq = seq.view(-1)
                labels = labels.view(-1)

                # this will call forward
                y_pred = self(seq)

                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()

                total_loss += single_loss.detach()
                avg_loss = total_loss / count
                batch_bar.set_postfix(loss=f"{avg_loss:10.10f}")
                batch_bar.update()
            batch_bar.close()
            print(f"[LSTM FIT]epoch: {i + 1:3} loss: {total_loss / len(arr):10.10f}")

    def _do_predict(self, seq: np.ndarray) -> float:
        """Use LSTM to predict based on input sequence.
        Parameters
        ----------
        test_seq
            1D Test sequence.

        Returns
        -------
        Predicted value at certain horizon.
        """
        # To tensor
        seq = torch.FloatTensor(seq).view(-1)

        with torch.no_grad():
            self._hidden_cell = (
                torch.zeros(self._num_hidden_layers, 1,
                            self._hidden_layer_size),
                torch.zeros(self._num_hidden_layers, 1,
                            self._hidden_layer_size),
            )
            pred = self(seq)

        return pred.item()

    def _get_transformers(self, data: np.ndarray) -> Tuple:
        """
        Get the transformers. In the case of the LSTM, it uses the same
        MinMaxScaler for both X and Y.
        Parameters
        ----------
        data :
            Training data

        Returns
        -------
        A tuple of x and y transformers
        """
        # todo: ask Jackie about the removal
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(data)

        # Time-series data shares the same transformer
        return scaler, scaler

    def _do_save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        return torch.load(path)


class ClusterForecaster:
    """
    Predict cluster in workload using trained LSTMs.

    Attributes
    ----------
    prediction_interval : pd.Timedelta
        Time interval to aggregate cluster counts by.
    prediction_horizon : pd.Timedelta
        The prediction horizon of the models to train.
    prediction_seqlen : int
        Number of intervals to feed the LSTM for a prediction.
    models : Dict[int, LSTM]
        Dictionary of trained models to perform inference by

    """

    MODEL_PREFIX = "model_"

    @staticmethod
    def cluster_to_file(path, cluster):
        """Generate model file path from cluster name"""
        return f"{path}/{ClusterForecaster.MODEL_PREFIX}{cluster}.pkl"

    @staticmethod
    def get_cluster_from_file(filename):
        """Infer cluster id from file name"""
        m = re.search(
            f"(?<={ClusterForecaster.MODEL_PREFIX})[^/]*(?=\\.pkl)", filename)
        if m is None:
            raise RuntimeError("Could not get cluster name")
        return m[0]

    def __init__(
            self,
            train_df,
            prediction_seqlen,
            prediction_interval,
            prediction_horizon,
            save_path,
            top_k=5,
            override=False,
    ):
        """Construct the ClusterForecaster object.
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data grouped by cluster and timestamp
        save_path : str
            Directory for loading/saving trained models
        top_k : int
            Only train models for the top k most common clusters.
        override : bool
            Determines whether we should (re)train models anyway, even if they are
            in the directory.
        """
        assert train_df.index.names[0] == "cluster"
        assert train_df.index.names[1] == "log_time_s"

        self.prediction_seqlen = prediction_seqlen
        self.prediction_interval = prediction_interval
        self.prediction_horizon = prediction_horizon
        self.models = {}

        if not override:
            model_files = glob.glob(
                str(Path(save_path) / f"{self.MODEL_PREFIX}*.pkl"))
            for filename in model_files:
                cluster_name = self.get_cluster_from_file(filename)
                self.models[int(cluster_name)] = LSTM.load(filename)
                print(f"loaded model for cluster {cluster_name}")
            print(f"Loaded {len(model_files)} models")

        if train_df is None:
            return

        # Only consider top k clusters.
        cluster_totals = train_df.groupby(
            level=0).sum().sort_values(by="count", ascending=False)
        labels = cluster_totals.index[:top_k]

        self.top_clusters = list(labels)
        print("Training on cluster time series..")

        mintime = train_df.index.get_level_values(1).min()
        maxtime = train_df.index.get_level_values(1).max()

        dtindex = pd.DatetimeIndex([mintime, maxtime])

        for cluster in labels:
            if cluster in self.models and not override:
                print(f"Already have model for cluster {cluster}, skipping")
                continue

            print(f"training model for cluster {cluster}")
            # obtain the time series data for the current cluster
            cluster_counts = train_df[train_df.index.get_level_values(
                0) == cluster].droplevel(0)

            # This zero-fills the start and ends of the cluster time series.
            cluster_counts = cluster_counts.reindex(
                cluster_counts.index.append(dtindex), fill_value=0)
            cluster_counts = cluster_counts.resample(prediction_interval).sum()
            self._train_cluster(cluster_counts, cluster, save_path)

    def _train_cluster(self, cluster_counts, cluster, save_path):
        dataset = ForecastDataset(
            cluster_counts,
            sequence_length=self.prediction_seqlen,
            horizon=self.prediction_horizon,
            interval=self.prediction_interval,
        )

        self.models[cluster] = LSTM(
            horizon=self.prediction_horizon,
            interval=self.prediction_interval,
            sequence_length=self.prediction_seqlen,
            epochs=1
        )

        # this will call _do_fit in the LSTM class above
        self.models[cluster].fit(dataset)
        self.models[cluster].save(self.cluster_to_file(save_path, cluster))

    def get_top_clusters(self):
        return self.top_clusters

    def predict_new(self, input_total, condition_len=112, prediction_len=16):
        num_windows, seq_length_total, num_cols = input_total.shape
        assert(seq_length_total == 128 and num_cols == 2)

        # note: make a deep copy for pretty plot
        predictions = copy.deepcopy(input_total)

        for w in range(num_windows):
            # note: this is assuming the first column is cluster id and the second column is count
            condition_window = input_total[w, :condition_len, 1]
            clusters = np.unique(input_total[w, :, 0], return_index=False, return_inverse=False, return_counts=False)

            assert(len(clusters) == 1)
            cluster = clusters[0]

            if cluster not in self.models.keys():
                print(f"Model for cluster {cluster} not found, returning ground truth")
                continue
            seq = torch.from_numpy(condition_window).view((-1, 1))
            for prediction_index in range(prediction_len):
                prediction = self.models[cluster].predict(seq)
                # note: preserve the cluster id column and only modify the count column
                predictions[w, prediction_index + condition_len, 1] = prediction
                seq = torch.cat((seq[1:], torch.FloatTensor([[prediction]])))

        return predictions



    def predict(self, cluster_df, cluster, start_time, end_time, autoregressive=True):
        """
        Given a cluster dataset, attempt to return prediction of query count
        from a cluster within the given time-range.
        """
        assert cluster_df.index.names[0] == "cluster"
        assert cluster_df.index.names[1] == "log_time_s"

        # Cluster not in the data.
        if cluster not in cluster_df.index.get_level_values(0):
            return None

        # No model for given cluster.
        if cluster not in self.models.keys():
            return None

        cluster_counts = cluster_df[cluster_df.index.get_level_values(
            0) == cluster].droplevel(0)

        if autoregressive:
            # note: round the start time to the nearest 10 minutes
            start_time = start_time.round(self.prediction_horizon)
            end_time = end_time.round(self.prediction_horizon)

            # note: changed this, need to check if this is correct. Suppose X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            #  sequence length is 2, and horizon is 10. If I want to predict the 7th point (0-indexed, ground truth is 7),
            #  the first time point I should feed in (inclusive) is 5.
            trunc_start = start_time - self.prediction_horizon - (self.prediction_seqlen - 1) * self.prediction_interval

            # note: the input data contains 10 points before the start time
            truncated = cluster_counts[(cluster_counts.index >= trunc_start) & (
                    cluster_counts.index < start_time)]

            seq = torch.FloatTensor(truncated.values)

            current_time = start_time - self.prediction_interval
            predictions = []
            while current_time < end_time:
                prediction = self.models[cluster].predict(seq)
                predictions.append(prediction)
                seq = torch.cat((seq[1:], torch.FloatTensor([[prediction]])))
                current_time += self.prediction_interval

            date_range = pd.date_range(start=start_time, end=end_time, freq=self.prediction_interval)
            assert(len(date_range) == len(predictions))

            pred_arr = [[date_range[i], pred]
                        for i, pred in enumerate(predictions)]

            pred_df = pd.DataFrame(pred_arr, columns=["log_time_s", "count"])
            pred_df.set_index("log_time_s", inplace=True)

            return pred_df

        else:
            # Truncate cluster_df to the time range necessary to generate prediction range.
            # todo: changed this, need to check if this is correct. Suppose X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            #  sequence length is 2, and horizon is 10. If I want to predict the 7th point (0-indexed, ground truth is 7),
            #  the first time point I should feed in (inclusive) is 5.
            trunc_start = start_time - self.prediction_horizon - \
                          (self.prediction_seqlen - 1) * self.prediction_interval
            trunc_end = end_time - self.prediction_horizon  # last ground truth to read

            truncated = cluster_counts[(cluster_counts.index >= trunc_start) & (
                    cluster_counts.index < trunc_end)]

            dataset = ForecastDataset(
                truncated,
                sequence_length=self.prediction_seqlen,
                horizon=self.prediction_horizon,
                interval=self.prediction_interval,
            )

            # generate predictions
            # note: this is feed one seq at a time and predict one more data point
            # note: i.e. it is not using the last prediction to autoregressively predict next data point
            # todo: find out the dataset that is used for the prediction
            # todo: concatenate the predicted results to use as the next input
            for seq, _ in dataset:
                print(seq)
                break
            predictions = [self.models[cluster].predict(seq) for seq, _ in dataset]

            # tag with timestamps
            pred_arr = [[dataset.get_y_timestamp(i), pred]
                        for i, pred in enumerate(predictions)]

            pred_df = pd.DataFrame(pred_arr, columns=["log_time_s", "count"])
            pred_df.set_index("log_time_s", inplace=True)
            return pred_df[start_time:]

