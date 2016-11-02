import numpy as np
from sklearn.metrics import roc_auc_score

from cnn_embedding import get_embedded_train_val_split
from data_provider import pairs_generator, load_organized_data_info
from utils import pairs_dot

IMGS_DIM_1D = 256
CNNS_WEIGHTS = {
    'cnn_2_9069_vl.h5': 11,
    'cnn_2_9273_vl.h5': 11,
    'cnn_2_9330_vl.h5': 11,
    'cnn_2_9549_vl.h5': 10,
    'cnn_2_9675_vl.h5': 10,
    'cnn_2_9678_vl.h5': 9,
    'cnn_2_9755_vl.h5': 9,
    'cnn_2_9806_vl.h5': 9,
    'cnn_2_9924_vl.h5': 9,
    'cnn_2_9979_vl.h5': 9,
    'cnn_3_0069_vl.h5': 9,
    'cnn_3_0236_vl.h5': 8,
    'cnn_3_0256_vl.h5': 8,
    'cnn_3_0416_vl.h5': 7,
    'cnn_3_0453_vl.h5': 7,
    'cnn_3_0456_vl.h5': 6,
    'cnn_3_0743_vl.h5': 4,
    'cnn_3_0752_vl.h5': 4,
}


def _softmax_dot():
    data_info = load_organized_data_info(IMGS_DIM_1D)
    X_avg, y_val = _average_embedded_val_data(data_info)

    batches_val = _create_pairs_generator(
        X_avg, y_val, lambda u, v: [u, v],
        num_groups=32,
        batch_size=1000000)

    y_pred, y_true = np.array([]), np.array([])
    for X, y in batches_val:
        y_pred = np.hstack((y_pred, pairs_dot(X)))
        y_true = np.hstack((y_true, y))

    print("Validation AUC: {:.4f}".format(roc_auc_score(y_true, y_pred)))


def _average_embedded_val_data(data_info):
    X_avg, y_val =\
        np.zeros((data_info['num_val'], data_info['num_distinct_cls'])), None

    for model, weight in CNNS_WEIGHTS.items():
        print("Model: {:s}".format(model))
        split = get_embedded_train_val_split('softmax', model_name=model)
        _, _, _, X_val, y_val, _ = split
        X_avg += weight * X_val

    X_avg /= sum(CNNS_WEIGHTS.values())
    return X_avg, y_val


def _create_pairs_generator(X, y, pairs_func, num_groups, batch_size):
    return pairs_generator(
        X, y,
        batch_size=batch_size,
        pair_func=pairs_func,
        num_groups=num_groups)


if __name__ == '__main__':
    _softmax_dot()
