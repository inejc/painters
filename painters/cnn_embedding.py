from os.path import join, isfile, basename, splitext, dirname

import numpy as np

from data_provider import DATA_DIR, load_organized_data_info
from data_provider import MODELS_DIR, testing_generator
from data_provider import init_directory_generator
from train_cnn import load_trained_cnn_feature_maps_layer, PENULTIMATE_SIZE
from train_cnn import load_trained_cnn_penultimate_layer, LAST_FEATURE_MAPS_SIZE
from train_cnn import load_trained_cnn_softmax_layer, SOFTMAX_SIZE
from utils import mkdirs_if_not_exist

IMGS_DIM_1D = 256
MODEL_NAME = 'cnn_2_9069_vl.h5'
BATCH_SIZE = 512

LAYER_RESULT_FUNCS = {
    'feature_maps': load_trained_cnn_feature_maps_layer,
    'penultimate': load_trained_cnn_penultimate_layer,
    'softmax': load_trained_cnn_softmax_layer
}

LAYER_SIZES = {
    'feature_maps': LAST_FEATURE_MAPS_SIZE,
    'penultimate': PENULTIMATE_SIZE,
    'softmax': SOFTMAX_SIZE
}


def get_embedded_train_val_split(layer, model_name=MODEL_NAME):
    assert layer in LAYER_RESULT_FUNCS.keys()

    model_path = join(MODELS_DIR, model_name)
    model_name_no_ext, _ = splitext(model_name)
    embedded_data_dir = join(
        DATA_DIR, 'embedding_{:s}'.format(model_name_no_ext))
    train_val_split_file = join(
        embedded_data_dir, 'train_val_split_{:s}.npz'.format(layer))

    if isfile(train_val_split_file):
        split = np.load(train_val_split_file)
        return split['arr_0'], split['arr_1'],\
            split['arr_2'], split['arr_3'],\
            split['arr_4'], split['arr_5']
    else:
        return _create_embedded_train_val_split(
            layer, model_path, train_val_split_file)


def _create_embedded_train_val_split(layer, model_path, train_val_split_file):
    data_info = load_organized_data_info(IMGS_DIM_1D)
    dir_tr, num_tr = data_info['dir_tr'], data_info['num_tr']
    dir_val, num_val = data_info['dir_val'], data_info['num_val']

    model = LAYER_RESULT_FUNCS[layer](model_path)
    gen = testing_generator(dir_tr=dir_tr)

    X_tr, y_tr, names_tr = _create_embedded_data_from_dir(
        model, gen, dir_tr, num_tr, LAYER_SIZES[layer])

    X_val, y_val, names_val = _create_embedded_data_from_dir(
        model, gen, dir_val, num_val, LAYER_SIZES[layer])

    _save_np_compressed_data(
        train_val_split_file, X_tr, y_tr, names_tr, X_val, y_val, names_val)

    return X_tr, y_tr, names_tr, X_val, y_val, names_val


def get_embedded_test_set(layer, model_name=MODEL_NAME):
    assert layer in LAYER_RESULT_FUNCS.keys()

    model_path = join(MODELS_DIR, model_name)
    model_name_no_ext, _ = splitext(model_name)
    embedded_data_dir = join(
        DATA_DIR, 'embedding_{:s}'.format(model_name_no_ext))
    test_set_file = join(embedded_data_dir, 'test_set_{:s}.npz'.format(layer))

    if isfile(test_set_file):
        test_set = np.load(test_set_file)
        return test_set['arr_0'], test_set['arr_1']
    else:
        return _create_embedded_test_set(layer, model_path, test_set_file)


def _create_embedded_test_set(layer, model_path, test_set_file):
    data_info = load_organized_data_info(IMGS_DIM_1D)
    dir_te, num_te = data_info['dir_te'], data_info['num_te']
    dir_tr = data_info['dir_tr']

    model = LAYER_RESULT_FUNCS[layer](model_path)
    gen = testing_generator(dir_tr=dir_tr)

    X_te, names = _create_embedded_data_from_dir(
        model, gen, dir_te, num_te, LAYER_SIZES[layer], is_test_set=True)

    _save_np_compressed_data(test_set_file, X_te, names)
    return X_te, names


def _create_embedded_data_from_dir(
        model, gen, dir_, num_samples, layer_size, is_test_set=False):
    gen = init_directory_generator(
        gen, dir_, BATCH_SIZE, class_mode='sparse', shuffle_=False)

    X, y = _create_embedded_data_from_gen(model, gen, num_samples, layer_size)
    names = [basename(p) for p in gen.filenames]

    if is_test_set:
        return X, names
    return X, y, names


def _create_embedded_data_from_gen(model, data_gen, num_samples, layer_size):
    num_full_epochs = num_samples // BATCH_SIZE
    last_batch_size = num_samples - (num_full_epochs * BATCH_SIZE)

    if isinstance(layer_size, int):
        X = np.empty((0, layer_size))
    else:
        X = np.empty((0, layer_size[0], layer_size[1], layer_size[2]))
    y = np.empty((0,)).astype(int)

    for i in range(num_full_epochs + 1):
        X_batch, y_batch = next(data_gen)

        if i == num_full_epochs:
            X_batch = X_batch[:last_batch_size]
            y_batch = y_batch[:last_batch_size]

        X = np.vstack((X, model(X_batch)))
        y = np.hstack((y, y_batch.astype(int)))

    return X, y


def _save_np_compressed_data(file_name, *args):
    mkdirs_if_not_exist(dirname(file_name))
    np.savez_compressed(file_name, *args)


if __name__ == '__main__':
    get_embedded_train_val_split('penultimate')
    get_embedded_test_set('penultimate')
    get_embedded_train_val_split('softmax')
    get_embedded_test_set('softmax')
    get_embedded_train_val_split('feature_maps')
    get_embedded_test_set('feature_maps')
