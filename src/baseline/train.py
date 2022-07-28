from model import ClusterForecaster
import pandas as pd
import sys
import numpy as np

sys.path.append('../')

DATA_PATH = "../../data/1year"
MODEL_DIR = "../../models"
PLOT_DIR = "../../figures/2016_9_2017_1"
PREDICTION_DIR = "prediction"


# test_df (num_windows, 128, 2)

def main():
    train_df = pd.read_csv(DATA_PATH + "/train.csv", parse_dates=['log_time_s'])
    train_df.set_index(['cluster', 'log_time_s'], inplace=True)
    test = np.load(DATA_PATH + "/test.npy")

    forecaster = ClusterForecaster(train_df,
                                   # note: the sequence length is changed to 116
                                   prediction_seqlen=116,
                                   prediction_interval=pd.Timedelta(minutes=10),
                                   # note: changed prediction horizon to 10m to allow autoregressive
                                   prediction_horizon=pd.Timedelta(minutes=10),
                                   save_path=MODEL_DIR,
                                   top_k=1,
                                   override=False)

    top_clusters = forecaster.get_top_clusters()
    np.savetxt(PREDICTION_DIR + "/cluster.txt", top_clusters, fmt="%d")
    pred = forecaster.predict_new(test)
    pred_filtered = pred[:, -16:, :]
    assert(pred.shape == test.shape)
    np.save(PREDICTION_DIR + "/prediction.npy", pred_filtered)




if __name__ == '__main__':
    main()
