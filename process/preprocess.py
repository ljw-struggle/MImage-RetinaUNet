# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

def write_hdf5(data, out_file):
    with h5py.File(out_file, 'w') as file:
        file.create_dataset('data', data=data, dtype=data.dtype)

def get_data(original_image_dir, ground_truth_dir, border_mask_dir):
    original_images = []
    ground_truths = []
    border_masks = []

    _, _, original_image_paths = list(os.walk(original_image_dir))[0]
    _, _, ground_truth_paths = list(os.walk(ground_truth_dir))[0]
    _, _, border_mask_paths = list(os.walk(border_mask_dir))[0]

    for i in tqdm(range(len(original_image_paths)), ascii=True):
        original_image = Image.open(original_image_dir + original_image_paths[i])
        ground_truth = Image.open(ground_truth_dir + ground_truth_paths[i])
        border_mask = Image.open(border_mask_dir + border_mask_paths[i])
        original_images.append(np.asarray(original_image))
        ground_truths.append(np.asarray(ground_truth))
        border_masks.append(np.asarray(border_mask))

    height, width = 584, 565
    original_images = np.reshape(original_images, (-1, height, width, 3))
    ground_truths = np.reshape(ground_truths, (-1, height, width, 1))
    border_masks = np.reshape(border_masks, (-1, height, width, 1))
    return original_images, ground_truths, border_masks

if __name__ == '__main__':
    preprocessed_dir = './data/DRIVE_preprocessed/'
    train_original_image_dir = './data/DRIVE/training/images/'
    train_ground_truth_dir = './data/DRIVE/training/1st_manual/'
    train_border_mask_dir = './data/DRIVE/training/mask/'
    test_original_image_dir = './data/DRIVE/test/images/'
    test_ground_truth_dir = './data/DRIVE/test/1st_manual/'
    test_border_mask_dir = './data/DRIVE/test/mask/'

    # 1\ Make a dir to save preprocessed data.
    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir)

    # 2\ Save train data.
    train_original_image, train_ground_truth, train_border_mask = get_data(
        train_original_image_dir, train_ground_truth_dir, train_border_mask_dir)
    write_hdf5(train_original_image, preprocessed_dir + 'DRIVE_train_original_image.hdf5')
    write_hdf5(train_ground_truth, preprocessed_dir + 'DRIVE_train_ground_truth.hdf5')
    write_hdf5(train_border_mask, preprocessed_dir + 'DRIVE_train_border_mask.hdf5')
    print('There are {} images for training.'.format(len(train_original_image), ))

    # 3\ Save test data.
    test_original_image, test_ground_truth, test_border_mask = get_data(
        test_original_image_dir, test_ground_truth_dir, test_border_mask_dir)
    write_hdf5(test_original_image, preprocessed_dir + 'DRIVE_test_original_image.hdf5')
    write_hdf5(test_ground_truth, preprocessed_dir + 'DRIVE_test_ground_truth.hdf5')
    write_hdf5(test_border_mask, preprocessed_dir + 'DRIVE_test_border_mask.hdf5')
    print('There are {} images for testing.'.format(len(test_original_image)))