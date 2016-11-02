import pickle
from csv import reader
from itertools import islice
from os import makedirs
from os.path import isdir

import numpy as np
from keras.preprocessing.image import load_img, img_to_array


def read_lines(file_name):
    with open(file_name, 'r') as f:
        return list(f)


def read_lines_in_batches(file_name, batch_size, num_skip=1):
    with open(file_name, 'r') as f:
        list(islice(f, num_skip))

        reader_ = reader(f)
        while True:
            batch = list(islice(reader_, batch_size))
            yield batch

            if len(batch) < batch_size:
                return


def save_pickle(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def append_to_file(lines, file_name):
    with open(file_name, 'a') as f:
        f.writelines(lines)


def mkdirs_if_not_exist(path):
    if not isdir(path):
        makedirs(path)


def load_img_arr(p):
    return img_to_array(load_img(p))


def pairs_dot(X):
    return np.sum(X[:, 0] * X[:, 1], axis=1)
