import threading
from itertools import combinations, chain
from json import load, dump
from math import ceil
from os import listdir
from os.path import join, dirname, isfile, abspath, isdir, basename
from random import shuffle

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from utils import read_lines, load_img_arr

DATA_DIR = join(dirname(dirname(__file__)), 'data')
TEST_DIR = join(DATA_DIR, 'test')
TRAIN_DIR = join(DATA_DIR, 'train')
TRAIN_INFO_FILE = join(DATA_DIR, 'train_info.csv')
SUBMISSION_INFO_FILE = join(DATA_DIR, 'submission_info.csv')
ORGANIZED_DATA_INFO_FILE = 'organized_data_info_.json'
MODELS_DIR = join(dirname(dirname(__file__)), 'models')
MISC_DIR = join(dirname(dirname(__file__)), 'misc')


def train_val_dirs_generators(
        batch_size, dir_tr, dir_val, target_size=(256, 256)):
    gen_tr = _train_generator()
    gen_val = _val_generator()

    sample = apply_to_images_in_subdirs(dir_tr, load_img_arr, num_per_cls=10)
    sample = np.array(sample)
    gen_tr.fit(sample)
    gen_val.fit(sample)

    gen_tr = init_directory_generator(
        gen_tr, dir_tr, batch_size, target_size=target_size)
    gen_val = init_directory_generator(
        gen_val, dir_val, batch_size, target_size=target_size)
    return gen_tr, gen_val


def _train_generator():
    return ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=180,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect')


def testing_generator(dir_tr):
    gen = _val_generator()
    sample = apply_to_images_in_subdirs(dir_tr, load_img_arr, num_per_cls=10)
    sample = np.array(sample)
    gen.fit(sample)
    return gen


def _val_generator():
    return ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True)


def apply_to_images_in_subdirs(parent_dir, func, num_per_cls=None, **kwargs):
    results = []
    for cls_dir_name in listdir(parent_dir):
        cls_dir = abspath(join(parent_dir, cls_dir_name))
        r = _apply_to_first_n_in_dir(func, cls_dir, num_per_cls, **kwargs)
        results += r
    return results


def _apply_to_first_n_in_dir(func, dir_, num_per_cls, **kwargs):
    if not isdir(dir_):
        return []
    results = []
    for path in listdir(dir_)[:num_per_cls]:
        abspath_ = abspath(join(dir_, path))
        result = func(abspath_, **kwargs)
        results.append(result)
    return results


def init_directory_generator(
        gen, dir_, batch_size, target_size=(256, 256),
        class_mode='categorical', shuffle_=True):

    return gen.flow_from_directory(
        dir_,
        class_mode=class_mode,
        batch_size=batch_size,
        target_size=target_size,
        shuffle=shuffle_)


def train_val_pairs_dirs_generators(
        batch_size, dir_tr, dir_val, num_groups_tr,
        num_groups_val, num_samples_per_cls_val=None):

    gen_tr = PairsImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=180,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect')
    gen_val = PairsImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True)

    sample = apply_to_images_in_subdirs(dir_tr, load_img_arr, num_per_cls=10)
    sample = np.array(sample)
    gen_tr.fit(sample)
    gen_val.fit(sample)

    gen_tr = gen_tr.flow_from_directory(
        dir_tr, batch_size=batch_size, num_groups=num_groups_tr)
    gen_val = gen_val.flow_from_directory(
        dir_val, batch_size=batch_size, num_groups=num_groups_val,
        num_samples_per_cls=num_samples_per_cls_val)
    return gen_tr, gen_val


class PairsImageDataGenerator(ImageDataGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg',
             num_groups=43, num_samples_per_cls=None):

        raise NotImplementedError

    def flow_from_directory(
            self, dir_, target_size=(256, 256), color_mode='rgb',
            classes=None, class_mode='categorical', batch_size=32,
            shuffle=True, seed=None, save_to_dir=None, save_prefix='',
            save_format='jpeg', num_groups=43, num_samples_per_cls=None):

        return PairsDirectoryIterator(
            dir_, num_groups, self, batch_size, num_samples_per_cls)


def inf_pairs_generator(batch_size, X, y, num_groups, num_samples_per_cls=None):
    return PairsNumpyArrayIterator(
        X, y, num_groups, batch_size, num_samples_per_cls)


class PairsNumpyArrayIterator(object):

    def __init__(self, X, y, num_groups, batch_size=32, num_per_cls=None):
        if num_per_cls:
            self.X, self.y = self._select_num_per_cls_samples(X, y, num_per_cls)
        else:
            self.X, self.y = X, y
        self.num_groups = num_groups
        self.batch_size = batch_size
        self._init_pairs_generator()
        self.lock = threading.Lock()

    def _init_pairs_generator(self):
        self.pairs_generator = pairs_generator(
            self.X, self.y, self.batch_size,
            lambda a, b: [a, b], self.num_groups)

    @staticmethod
    def _select_num_per_cls_samples(X, y, num_per_cls):
        X_sub, y_sub = np.empty((0,) + X.shape[1:]), np.empty((0,))
        for cls in set(y):
            X_sub = np.vstack((X_sub, X[y == cls][:num_per_cls]))
            y_sub = np.hstack((y_sub, y[y == cls][:num_per_cls]))
        return X_sub, y_sub

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            try:
                X_batch, y_batch = next(self.pairs_generator)
            except StopIteration:
                # todo: implement this properly :)
                self._init_pairs_generator()
                X_batch, y_batch = next(self.pairs_generator)
        return [X_batch[:, 0], X_batch[:, 1]], y_batch


class PairsDirectoryIterator(object):

    def __init__(self, dir_, num_groups, image_data_generator,
                 batch_size=32, num_samples_per_cls=None):

        paths, y = self._get_paths_labels_from_dir(dir_, num_samples_per_cls)
        self.paths = paths
        self.y = y
        self.num_groups = num_groups
        self.batch_size = batch_size
        self._init_pairs_generator()
        self.image_data_generator = image_data_generator
        self.lock = threading.Lock()

    @staticmethod
    def _get_paths_labels_from_dir(dir_, num_per_cls):
        def path_label(p): return [p, basename(dirname(p))]
        paths_labels = apply_to_images_in_subdirs(dir_, path_label, num_per_cls)
        paths_labels = np.array(paths_labels)
        return paths_labels[:, 0], paths_labels[:, 1].astype(int)

    def _init_pairs_generator(self):
        self.pairs_generator = pairs_generator(
            self.paths, self.y, self.batch_size,
            lambda a, b: [a, b], self.num_groups)

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            try:
                paths_batch, y_batch = next(self.pairs_generator)
            except StopIteration:
                # todo: implement this properly :)
                self._init_pairs_generator()
                paths_batch, y_batch = next(self.pairs_generator)

        X_batch = []
        for path_a, path_b in paths_batch:
            image_a, image_b = load_img_arr(path_a), load_img_arr(path_b)
            image_a = self._std_random_transform_img(image_a)
            image_b = self._std_random_transform_img(image_b)
            X_batch.append([image_a, image_b])
        X_batch = np.array(X_batch)

        return [X_batch[:, 0], X_batch[:, 1]], y_batch

    def _std_random_transform_img(self, img):
        img = self.image_data_generator.random_transform(img)
        return self.image_data_generator.standardize(img)


def pairs_generator(X, y, batch_size, pair_func, num_groups):
    grouped_indices = _split_into_groups(y, num_groups)
    merged_combinations = _merge_within_groups_combinations(grouped_indices)

    while True:
        X_batch, y_batch = [], []

        for _ in range(batch_size):
            try:
                pair_indices = next(merged_combinations)
            except StopIteration:
                return

            index_a, index_b = int(pair_indices[0]), int(pair_indices[1])
            X_batch.append(pair_func(X[index_a], X[index_b]))
            y_batch.append(int(y[index_a] == y[index_b]))

        yield np.array(X_batch), np.array(y_batch)


def _split_into_groups(y, num_groups):
    groups = [[] for _ in range(num_groups)]
    group_index = 0

    for cls in set(y):
        this_cls_indices = np.where(y == cls)[0]
        num_cls_samples = len(this_cls_indices)

        num_cls_split_groups = ceil(num_cls_samples / 500)
        split = np.array_split(this_cls_indices, num_cls_split_groups)

        for cls_group in split:
            groups[group_index] = np.hstack((groups[group_index], cls_group))
            group_index = (group_index + 1) % num_groups

    return groups


def _merge_within_groups_combinations(grouped_indices):
    for gi in grouped_indices:
        shuffle(gi)
    group_combinations = [combinations(gi, 2) for gi in grouped_indices]
    shuffle(group_combinations)
    return chain.from_iterable(group_combinations)


def load_train_info():
    train_info = read_lines(TRAIN_INFO_FILE)[1:]
    parsed_train_info = {}
    # filename,artist,title,style,genre,date
    for l in train_info:
        split = l.split(',')
        parsed_train_info[split[0]] = split[1]
    return parsed_train_info


def load_organized_data_info(imgs_dim_1d, multi_crop=False):
    if not isfile(_organized_data_info_file_dim(imgs_dim_1d, multi_crop)):
        raise FileNotFoundError('Run data_dirs_organizer first')
    with open(_organized_data_info_file_dim(imgs_dim_1d, multi_crop), 'r') as f:
        return load(f)


def save_organized_data_info(info, imgs_dim_1d, multi_crop=False):
    with open(_organized_data_info_file_dim(imgs_dim_1d, multi_crop), 'w') as f:
        dump(info, f)


def _organized_data_info_file_dim(imgs_dim_1d, multi_crop=False):
    split = ORGANIZED_DATA_INFO_FILE.split('.')
    split[0] += str(imgs_dim_1d)
    if multi_crop:
        split[0] += '_multi_crop'
    return join(DATA_DIR, '.'.join(split))
