import pandas as pd
from torch.utils.data import Dataset
import torch


class ForecastDataset(Dataset):
    """Data loader for time-series forecasting. Inherits torch.utils.data.Dataset

    Attributes
    ----------
    raw_df : pd.Dataframe
        A time-indexed dataframe of the sequence of counts, aggregated by
        the interval.
    horizon : pd.Timedelta
        How far into the future to predict (the y label).
    interval : pd.Timedelta
        The time interval to aggregate the original timeseries into.
    sequence_length : int
        The number of data points to use for prediction.
    X : torch.FloatTensor
        The (optionally transformed) tensors representing the training sequence
    y : torch.FloatTensor.
        The (optionally transformed) tensors expecting the expected value at the
        specified horizon.
    """

    def __init__(
            self,
            df,
            horizon=pd.Timedelta("1S"),
            interval=pd.Timedelta("1S"),
            sequence_length=5,
    ):
        self.horizon = horizon
        self.interval = interval
        self.sequence_length = sequence_length
        self.raw_df = df.resample(interval).sum()

        self.set_transformers((None, None))

    def __len__(self):
        # note: set the length of the dataset to be the length of y
        #  so that x will not be read off the frame
        return self.y.shape[0]

    def __getitem__(self, i):
        # note: no special logic necessary here!
        x = self.X[i: i + self.sequence_length]
        y = self.y[i].reshape((1, -1))
        return x, y

    def get_y_timestamp(self, ind):
        """For a (seq,label) pair in the dataset, return the
        corresponding timestamp for the label.

        Parameters
        ----------
        ind : int
            Index into the dataset of the label in question.

        Returns
        -------
        pd.Timestamp : The timestamp of the corresponding label.
        """
        return self.raw_df.index[ind] + self.horizon

    def set_transformers(self, transformers):
        """Transform the X and y tensors according to the supplied transformers.
        Currently, both transformers are MinMaxScalers

        Parameters
        ----------
        transformers : (None, None) or Tuple of scikit.preprocessing data transformers
        """
        x_transformer, y_transformer = transformers
        # note: now shift the original so that the window starting position is aligned with
        #  the first prediction label. Assuming sequence is 2, horizon is 1 and X = [1, 2, 3, 4, 5, 6, 7, 8],
        #  then                                                                 y = [3, 4, 5, 6, 7, 8]


        # note: --------
        #                   2020-01-01 00:00:00
        # note: --------
        #                   2020-01-01 00:10:00
        # note: --------
        #                   2020-01-01 00:20:00
        # note: --------


        label_start_time = self.raw_df.index[0] + (self.sequence_length - 1) * self.interval + self.horizon
        shifted = self.raw_df[self.raw_df.index >= label_start_time]
        if x_transformer is None or y_transformer is None:
            self.X = torch.FloatTensor(self.raw_df.values)
            self.y = torch.FloatTensor(shifted.values)
            return

        self.X = torch.FloatTensor(x_transformer.transform(self.raw_df.values))
        self.y = torch.FloatTensor(y_transformer.transform(shifted.values))
