import os
import pandas as pd
import numpy as np
from scipy import stats


def read_timeseries(data_dir):
    cluster_dfs = []

    for cluster_filename in os.listdir(data_dir):
        cluster = int(cluster_filename.split(".")[0])
        df = pd.read_csv(f"{data_dir}/{cluster_filename}", parse_dates=[0])  # parse_dates is important!
        df.columns = ["log_time", "count"]

        # Pad missing timestamps with 0s
        df_start = min(df.log_time)
        df_end = max(df.log_time)
        df_time_range = pd.date_range(df_start, df_end, freq="10T")
        df.set_index("log_time", inplace=True)
        df = df.reindex(df_time_range, fill_value=0)

        # Recover 'log_time'
        # df.reset_index(inplace=True)
        # df.columns = ["log_time", "count"]
        df.insert(0, "cluster", cluster)  # Insert a 'cluster' column

        assert df.shape[0] == len(df_time_range)
        cluster_dfs.append([cluster, df, df_time_range])

    return cluster_dfs


def read_timeseries_with_timestamp(data_dir):
    cluster_dfs = []

    for cluster_filename in os.listdir(data_dir):
        cluster = int(cluster_filename.split(".")[0])
        df = pd.read_csv(f"{data_dir}/{cluster_filename}", parse_dates=[0])  # parse_dates is important!
        df.columns = ["log_time", "count"]

        # Pad missing timestamps with 0s
        df_start = min(df.log_time)
        df_end = max(df.log_time)
        df_time_range = pd.date_range(df_start, df_end, freq="10T")
        df.set_index("log_time", inplace=True)
        df = df.reindex(df_time_range, fill_value=0)

        # Recover 'log_time'
        df.reset_index(inplace=True)
        df.columns = ["log_time", "count"]
        df.insert(0, "cluster", cluster)  # Insert a 'cluster' column

        assert df.shape[0] == len(df_time_range)
        cluster_dfs.append([cluster, df, df_time_range])

    return cluster_dfs


def merge(data_dir, train_ratio, return_time_range=False):
    """
    1. Load data from {cluster}.csv in `data_dir`.
    2. Add padding and fill holes in date range.
    3. Concatecate into a big dataframe

    Args:
        data_dir: str
    """

    # cluster, log_time, count
    cluster_dfs = read_timeseries_with_timestamp(data_dir)

    train_dfs = []
    test_dfs = []

    for i, (cluster_id, df, df_time_range) in enumerate(cluster_dfs):
        train_df, test_df = split_train_test(df, train_ratio)
        train_dfs.append(train_df)
        test_dfs.append(test_df)

    train = pd.concat([l for l in train_dfs])  # Concate dataframes
    test = pd.concat([l for l in test_dfs])

    return train, test


def split_train_test(df, train_ratio):
    """
    Format data before feeding into model.
    Then split into train and test set.
    """

    # 1. Copy 'log_time' to 'log_time_s'
    # ForecastDataset expects 'log_time_s', at second granularity, and resample
    # at specified interval. Our dataset is already at 10-min granularity. But
    # it works fine.
    df['log_time_s'] = df['log_time']
    df = df.drop(columns=['log_time'])

    # # 2. Split into train and test
    start_time = min(df['log_time_s'])
    end_time = max(df['log_time_s'])
    split_time = start_time + train_ratio * (end_time - start_time)
    train_df = df[df['log_time_s'] < split_time]
    test_df = df[df['log_time_s'] >= split_time]

    # 3. Set index
    # train_df.set_index(['cluster', 'log_time_s'], inplace=True)
    # test_df.set_index(['cluster', 'log_time_s'], inplace=True)

    assert not any(train_df.index.duplicated())
    assert not any(test_df.index.duplicated())

    return train_df, test_df


def preprocess_LSTM_train(raw_dir, output_dir):
    train_df, _ = merge(raw_dir, 0.8)
    train_df.to_csv(f"{output_dir}/train.csv", index=None)


def preprocess_LSTM_test(test_zxs_npy, test_labels_npy, save_path, idx2cluster):
    test_zxs = np.load(test_zxs_npy)
    test_labels = np.load(test_labels_npy)

    assert test_zxs.shape[0] == test_labels.shape[0]

    N = test_zxs.shape[0]
    test_data = np.zeros((N, 128, 2))

    # [num windows, 128, [cnt_{t-1}, ..., cluster_id]]
    for i in range(N):
        test_data[i, :, 0] = idx2cluster[test_zxs[i, 0, -1]]
        test_data[i, :, 1] = test_labels[i]

    np.save(save_path, test_data)


def gen_covariates(timepoints, num_covariates):
    assert num_covariates >= 4

    covariates = np.zeros([timepoints.shape[0], num_covariates])
    for i, timepoint in enumerate(timepoints):
        covariates[i, 1] = timepoint.weekday()
        covariates[i, 2] = timepoint.hour
        covariates[i, 3] = timepoint.month

    for i in range(1, num_covariates):
        covariates[:, i] = stats.zscore(covariates[:, i])

    return covariates


def gen_data(
    data, covariates, num_covariates, total_date_range, window_size, stride_size, save_path, save_name, train=True,
):
    """
    Args:
        data (List): (num_data_points{train/test}, num_clusters) --> Train or test data
        covariates (np.ndarray): (num_data_points, 4)
        num_covariates (int)
        total_date_range: date range of ALL clusters
        window_size (int)
        stride_size (int)
        train (bool): generating data for train or test
    """
    # Compute how many data points there are in total
    num_clusters = len(data)
    input_size = window_size - stride_size  # stride_size == prediction horizon length
    total_windows = 0
    windows_per_series = []
    for i in range(num_clusters):
        time_len = data[i].shape[0]
        num_window = (time_len - input_size) // stride_size
        windows_per_series.append(num_window)
        total_windows += num_window

    x_input = np.zeros(
        (total_windows, window_size, 1 + num_covariates + 1), dtype="float32"
    )  # (total_windows, window_size, num_features)
    label = np.zeros((total_windows, window_size), dtype="float32")

    # v is the scaling factor
    v_input = np.zeros((total_windows, 2), dtype="float32")  # (total_windows, 2)

    cov_age_df = pd.DataFrame(index=total_date_range.copy(), data=stats.zscore(np.arange(len(total_date_range))))
    count = 0
    for cluster in range(num_clusters):
        cluster_start = min(data[cluster].log_time)
        cluster_end = max(data[cluster].log_time)
        time_len = data[cluster].shape[0]

        mask = (cov_age_df.index >= cluster_start) & (cov_age_df.index <= cluster_end)

        # covariates are for range [train_start, test_end]
        # For training data, we only need [train_start, train_end]
        # For testing data, we only need [test_start, test_end].
        # That's why we need a mask.
        if train:
            covariates[cluster][:time_len, 0] = cov_age_df.loc[mask].values.squeeze(1)
        else:
            covariates[cluster] = covariates[cluster][-time_len:]
            covariates[cluster][:, 0] = cov_age_df.loc[mask].values.squeeze(1)

        # Loop through all the windows for this cluster
        for i in range(windows_per_series[cluster]):
            window_start = stride_size * i
            window_end = window_start + window_size

            label[count, :] = data[cluster].loc[window_start: window_end - 1, "count"]  # loc is inclusive

            # [cnt, cov_1, ... cov_{num_covariates}, series id]
            x_input[count, 1:, 0] = data[cluster].loc[window_start: (window_end - 1) - 1, "count"]
            x_input[count, :, 1: 1 + num_covariates] = covariates[cluster][
                window_start:window_end
            ]  # numpy array is not inclusive, no need for window_end-1
            x_input[count, :, -1] = cluster

            nonzero_sum = (x_input[count, 1:input_size, 0] != 0).sum()
            if nonzero_sum == 0:
                v_input[count, 0] = 0
            else:
                v_input[count, 0] = np.true_divide(x_input[count, 1:input_size, 0].sum(), nonzero_sum) + 1
                x_input[count, :, 0] = x_input[count, :, 0] / v_input[count, 0]
                if train:
                    label[count, :] = label[count, :] / v_input[count, 0]

            count += 1

    prefix = os.path.join(save_path, "train_" if train else "test_")
    np.save(prefix + "zx_" + save_name, x_input)
    np.save(prefix + "v_" + save_name, v_input)
    np.save(prefix + "label_" + save_name, label)


def preprocess_deepAR(
    raw_dir,
    save_path,
    save_name,
    num_covariates=4,
    window_size=128,
    stride_size=16,
    train_ratio=0.8,
    overlap_ratio=0.1,
):
    cluster_dfs = read_timeseries(raw_dir)  # [(cluster1, df1, time_range1), ...]

    # Generate covariates for each cluster. Split each cluster_df into train and test numpy arrays
    cluster2idx = {}
    train_data = []  # [train_df1, train_df2, ...]
    test_data = []  # [test_df1, test_df2, ...]
    covariates = []  # [covariates1, covariates2, ...]
    df_start_times = []
    df_end_times = []
    for i, (cluster_id, df, df_time_range) in enumerate(cluster_dfs):
        cluster2idx[cluster_id] = i
        covariates.append(gen_covariates(df_time_range, num_covariates))  # (num_data_points, 4)

        # Split into train and test
        train_start = min(df_time_range)
        test_end = max(df_time_range)
        total_range = test_end - train_start
        train_end = train_start + train_ratio * total_range
        test_start = train_end

        df = df.reset_index()
        df.columns = ["log_time", "cluster", "count"]

        # Exact same splitting logic as for LSTM baseline
        train_df = df[df['log_time'] < train_end]
        train_df.columns = ["log_time", "cluster", "count"]
        train_df.reset_index(inplace=True)
        train_data.append(train_df)

        test_df = df[df['log_time'] >= test_start]
        test_df.columns = ["log_time", "cluster", "count"]
        test_df.reset_index(inplace=True)
        test_data.append(test_df)

        # Keep track of start and end times of each df
        df_start_times.append(train_start)
        df_end_times.append(test_end)

    # Get total date_range for gen_data function
    total_date_range = pd.date_range(min(df_start_times), max(df_end_times), freq="10T")

    gen_data(
        train_data,
        covariates,
        num_covariates,
        total_date_range,
        window_size,
        stride_size,
        save_path,
        save_name,
        train=True,
    )
    gen_data(
        test_data,
        covariates,
        num_covariates,
        total_date_range,
        window_size,
        stride_size,
        save_path,
        save_name,
        train=False,
    )

    # TODO: Store cluster2idx for potential future use?
    return cluster2idx

if __name__ == "__main__":
    raw_dir = "./1year/raw"
    output_dir = "./1year"

    # Generates training and testing data for DeepAR, using raw *.csv
    # train_*.npy, test_*.npy
    cluster2idx = preprocess_deepAR(raw_dir, output_dir, "1year")
    idx2cluster = {v: k for k, v in cluster2idx.items()}

    # Generates training data for LSTM baseline: train.csv
    preprocess_LSTM_train(raw_dir, output_dir)

    # Generate testing data for LSTM baseline: test.npy
    preprocess_LSTM_test("./1year/test_zx_1year.npy", "./1year/test_label_1year.npy", "./1year/test.npy", idx2cluster)

    # Check
    # train_input = np.load(f"{output_dir}/train_zx_1year.npy")
    # test_input = np.load(f"{output_dir}/test_zx_1year.npy")
    # print(train_input.shape, test_input.shape)

    test = np.load("./1year/test.npy")
    print(test[0])
