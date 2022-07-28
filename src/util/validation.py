import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

LSTM_PRED_PATH = "../baseline/prediction/prediction.npy"
LSTM_CLUSTER_PATH = "../baseline/prediction/cluster.txt"
DEEPAR_PRED_PATH = "../deepar/prediction/prediction.npy"
TRUE_LABEL_PATH = "../../data/1year/test_label_1year.npy"


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def validation(true_labels, predicted_labels, clusters_file=None):
    # true_labels: (num_of_points * 128) where each point is the ground truth
    # predicted_labels: (num_of_points * 16 * 1)
    num_points = true_labels.shape[0]
    pred_len = predicted_labels.shape[1]
    assert num_points == predicted_labels.shape[0]
    if clusters_file:
        clusters = np.loadtxt(clusters_file)
        y_true = np.empty((0, 1))
        y_pred = np.empty((0, 1))
        for i in range(num_points):
            cluster_id = predicted_labels[i][0][0]
            if cluster_id not in clusters:
                continue
            y_true = np.vstack((y_true, true_labels[i, -pred_len:].reshape(-1, 1)))
            y_pred = np.vstack((y_pred, predicted_labels[i, :, -1].reshape((-1, 1))))

    else:
        y_true = true_labels[:, -pred_len:].reshape((num_points * pred_len, 1))
        y_pred = predicted_labels[:, :, -1].reshape((num_points * pred_len, 1))

    rmse_score = rmse(y_true, y_pred)
    r2_score = r2(y_true, y_pred)
    return rmse_score, r2_score, np.log10(rmse_score)


def test_validation():
    true = np.array([[[1, 1], [2, 1], [3, 0]], [[5, 1], [6, 0], [7, 2]]])
    pred = np.array([[[2, 0], [3, 1]], [[6, 1], [7, 2]]])
    print(validation(true, pred))


if __name__ == "__main__":
    lstm_pred = np.load(LSTM_PRED_PATH)
    deepar_pred = np.load(DEEPAR_PRED_PATH)
    true_labels = np.load(TRUE_LABEL_PATH)

    print(f"LSTM score: {validation(true_labels, lstm_pred, LSTM_CLUSTER_PATH)}")
    print(f"DeepAR score: {validation(true_labels, deepar_pred)}")
