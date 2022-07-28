import pandas as pd


def split_train_test(df, train_ratio, return_timestamps):
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

    # 2. Split into train and test
    start_time = min(df['log_time_s'])
    end_time = max(df['log_time_s'])
    split_time = start_time + train_ratio * (end_time - start_time)
    train_df = df[df['log_time_s'] < split_time]
    test_df = df[df['log_time_s'] >= split_time]

    # 3. Set index
    train_df.set_index(['cluster', 'log_time_s'], inplace=True)
    test_df.set_index(['cluster', 'log_time_s'], inplace=True)

    assert not any(train_df.index.duplicated())
    assert not any(test_df.index.duplicated())

    if return_timestamps:
        return train_df, test_df, (start_time, split_time, end_time)
    else:
        return train_df, test_df


def load_data(data_path, train_ratio=0.8, return_timestamps=False):
    """
    1. Load data from `data_path`
    2. Split into train and test set

    Args:
        data_dir: str
        train_ratio: float
        return_split_time: return split time if True
    """
    df = pd.read_csv(data_path, parse_dates=['log_time'])
    return split_train_test(df, train_ratio, return_timestamps)
