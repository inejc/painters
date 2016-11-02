from os.path import join

import numpy as np

from cnn_embedding import get_embedded_test_set
from data_provider import SUBMISSION_INFO_FILE, DATA_DIR
from data_provider import load_organized_data_info
from utils import append_to_file
from utils import read_lines_in_batches
from validation import CNNS_WEIGHTS

IMGS_DIM_1D = 256
SUBMISSION_FILE = join(DATA_DIR, 'submission.csv')
BATCH_SIZE = 1000000
FILES_TO_AVG = {}


def _create_submission_file_avg_cnns():
    data_info = load_organized_data_info(IMGS_DIM_1D)
    X_avg, names = _average_embedded_test_data(data_info)
    features_lookup = {n: f for n, f in zip(names, X_avg)}
    _create_submission_file(
        BATCH_SIZE, features_lookup, _calculate_batch_prediction_dot)


def _average_embedded_test_data(data_info):
    X_avg, names =\
        np.zeros((data_info['num_te'], data_info['num_distinct_cls'])), None
    for model, weight in CNNS_WEIGHTS.items():
        X, names = get_embedded_test_set('softmax', model_name=model)
        X_avg += weight * X

    X_avg /= sum(CNNS_WEIGHTS.values())
    return X_avg, names


def _calculate_batch_prediction_dot(lines, features_lookup):
    y_pred, submission_indices = [], []

    for line in lines:
        submission_indices.append(line[0])
        image_feature_a = features_lookup[line[1]]
        image_feature_b = features_lookup[line[2]]
        y_pred.append(np.dot(image_feature_a, image_feature_b))

    return y_pred, submission_indices


def _create_submission_file(batch_size, features_lookup, batch_predict_func):
    append_to_file(["index,sameArtist\n"], SUBMISSION_FILE)

    for batch in read_lines_in_batches(SUBMISSION_INFO_FILE, batch_size):
        y_pred, indices = batch_predict_func(batch, features_lookup)
        lines = ["{:s},{:f}\n".format(i, p) for i, p in zip(indices, y_pred)]
        append_to_file(lines, SUBMISSION_FILE)


def _average_submission_files():
    lines_gens, weights = [], []

    for file_name, weight in FILES_TO_AVG.items():
        file_path = join(DATA_DIR, file_name)
        lines_gen = read_lines_in_batches(file_path, batch_size=BATCH_SIZE)
        lines_gens.append(lines_gen)
        weights.append(weight)

    append_to_file(["index,sameArtist\n"], SUBMISSION_FILE)

    while True:
        try:
            _average_write_next_batch(lines_gens, weights)
        except StopIteration:
            return


def _average_write_next_batch(lines_gens, weights):
    separated_lines = [next(lg) for lg in lines_gens]
    merged_lines = zip(*separated_lines)

    result_lines = []

    for same_example_lines in merged_lines:
        example_index = same_example_lines[0][0]
        preds = [float(l[1]) for l in same_example_lines]
        pred_avg = sum(w * p for w, p in zip(weights, preds)) / sum(weights)
        result_lines.append("{:s},{:f}\n".format(example_index, pred_avg))

    append_to_file(result_lines, SUBMISSION_FILE)


if __name__ == '__main__':
    _create_submission_file_avg_cnns()
    # _average_submission_files()
