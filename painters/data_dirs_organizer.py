from os import mkdir, listdir, makedirs
from os.path import join, abspath, basename, splitext, dirname

import numpy as np
from PIL.Image import LANCZOS
from PIL.ImageOps import fit
from keras.preprocessing.image import load_img
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from data_provider import TRAIN_DIR, TEST_DIR, DATA_DIR
from data_provider import load_train_info
from data_provider import save_organized_data_info, load_organized_data_info

IMGS_DIM_2D = (256, 256)
NEW_TRAIN_DIR = join(DATA_DIR, 'train_{:d}'.format(IMGS_DIM_2D[0]))
NEW_VAL_DIR = join(DATA_DIR, 'val_{:d}'.format(IMGS_DIM_2D[0]))
NEW_TEST_DIR = join(DATA_DIR, 'test_{:d}'.format(IMGS_DIM_2D[0]))
NEW_TEST_DIR = join(NEW_TEST_DIR, 'all')
MULTI_CROP = '_multi_crop'
VAL_SIZE = 0.1


def _organize_train_test_dir():
    _organize_train_dir()
    _organize_test_dir()


def _organize_train_dir():
    paths, labels = _load_paths_labels_from_train_dir()
    ind_tr, ind_val, classes = _train_val_split_indices(labels)
    _save_images_to_dir(NEW_TRAIN_DIR, paths[ind_tr], labels[ind_tr], classes)
    _save_images_to_dir(
        NEW_VAL_DIR, paths[ind_val], labels[ind_val], classes, multi_crop=False)


def _load_paths_labels_from_train_dir():
    labels_lookup = load_train_info()
    paths, labels = [], []
    for name in listdir(TRAIN_DIR):
        abspath_ = abspath(join(TRAIN_DIR, name))
        paths.append(abspath_)
        labels.append(labels_lookup[name])

    return np.array(paths), LabelEncoder().fit_transform(labels)


def _train_val_split_indices(labels):
    split = StratifiedShuffleSplit(
        labels, n_iter=1, test_size=VAL_SIZE, random_state=42)
    indices_tr, indices_val = next(iter(split))

    _save_organized_data_info(
        split.classes, indices_tr, indices_val, multi_crop=False)
    _save_organized_data_info(
        split.classes, indices_tr, indices_val, multi_crop=True)
    return indices_tr, indices_val, split.classes


def _save_organized_data_info(classes, indices_tr, indices_val, multi_crop):
    dir_tr = NEW_TRAIN_DIR + MULTI_CROP if multi_crop else NEW_TRAIN_DIR
    num_tr = 5 * len(indices_tr) if multi_crop else len(indices_tr)
    info = {
        'dir_tr': dir_tr,
        'num_tr': num_tr,
        'dir_val': NEW_VAL_DIR,
        'num_val': len(indices_val),
        'num_distinct_cls': len(classes),
        'dir_te': dirname(NEW_TEST_DIR)
    }
    save_organized_data_info(info, IMGS_DIM_2D[0], multi_crop)


def _save_images_to_dir(
        dest_dir, src_paths, labels, distinct_classes, multi_crop=True):

    _make_dir_tree(dest_dir, distinct_classes)
    if multi_crop:
        _make_dir_tree(dest_dir + MULTI_CROP, distinct_classes)

    for src_path, label in zip(src_paths, labels):
        dest_path = join(join(dest_dir, str(label)), basename(src_path))
        scaled_cropped_image = _save_scaled_cropped_img(src_path, dest_path)

        if multi_crop:
            _save_multi_cropped_to_dir(
                src_path, dest_dir, label, scaled_cropped_image)


def _make_dir_tree(dir_, classes):
    mkdir(dir_)
    for class_ in classes:
        class_dir = join(dir_, str(class_))
        mkdir(class_dir)


def _save_multi_cropped_to_dir(src_path, dest_dir, label, scaled_cropped_image):
    multi_crop_dir = join(dest_dir + MULTI_CROP, str(label))
    dest_path_multi_crop = join(multi_crop_dir, basename(src_path))
    scaled_cropped_image.save(dest_path_multi_crop)
    _save_multi_cropped_imgs(src_path, dest_path_multi_crop)


def _save_multi_cropped_imgs(src, dest):
    image = load_img(src)
    image, crop_coordinates = _prepare_image_for_cropping(image)

    dest_no_ext, ext = splitext(dest)
    for i, crop_position in enumerate(crop_coordinates):
        dest_i = "{:s}_{:d}{:s}".format(dest_no_ext, i, ext)
        cropped_img = image.crop(box=crop_position)

        assert cropped_img.size == IMGS_DIM_2D, \
            'Cropped image dimension is {:s}, instead of {:s}'\
            .format(cropped_img.size, IMGS_DIM_2D)

        cropped_img.save(dest_i)


def _prepare_image_for_cropping(image):
    width, height = image.size

    fixed_width = IMGS_DIM_2D[0] if width < IMGS_DIM_2D[0] else width
    fixed_height = IMGS_DIM_2D[1] if height < IMGS_DIM_2D[1] else height
    if (fixed_width, fixed_height) != image.size:
        image = image.resize((fixed_width, fixed_height), resample=LANCZOS)

    crop_coordinates = [
        (0, 0, IMGS_DIM_2D[0], IMGS_DIM_2D[1]),
        (fixed_width - IMGS_DIM_2D[0], 0, fixed_width, IMGS_DIM_2D[1]),
        (0, fixed_height - IMGS_DIM_2D[1], IMGS_DIM_2D[0], fixed_height),
        (fixed_width - IMGS_DIM_2D[0], fixed_height - IMGS_DIM_2D[1],
         fixed_width, fixed_height),
    ]

    return image, crop_coordinates


def _organize_test_dir():
    makedirs(NEW_TEST_DIR)

    num_test_samples = 0
    for name in listdir(TEST_DIR):
        src_path = abspath(join(TEST_DIR, name))
        dest_path = join(NEW_TEST_DIR, name)
        _save_scaled_cropped_img(src_path, dest_path)
        num_test_samples += 1

    _append_num_te_to_organized_data_info(num_test_samples, multi_crop=False)
    _append_num_te_to_organized_data_info(num_test_samples, multi_crop=True)


def _save_scaled_cropped_img(src, dest):
    image = load_img(src)
    image = fit(image, IMGS_DIM_2D, method=LANCZOS)
    image.save(dest)
    return image


def _append_num_te_to_organized_data_info(num_test_samples, multi_crop):
    data_info = load_organized_data_info(IMGS_DIM_2D[0], multi_crop=multi_crop)
    data_info['num_te'] = num_test_samples
    save_organized_data_info(data_info, IMGS_DIM_2D[0], multi_crop=multi_crop)


if __name__ == '__main__':
    _organize_train_test_dir()
