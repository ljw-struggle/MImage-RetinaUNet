# -*- coding: utf-8 -*-
import os, configparser, argparse, random

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from model import get_unet_model
from loader import loader
from utils import load_hdf5, recompose, evaluate_metric

random.seed(10)

def train(config):
    name_experiment      = config.get('Experiment Name', 'name')
    train_original_image = config.get('Data Attribute', 'train_original_image')
    train_ground_truth   = config.get('Data Attribute', 'train_ground_truth')
    train_border_mask    = config.get('Data Attribute', 'train_border_mask')
    patch_height         = config.getint('Data Attribute', 'patch_height')
    patch_width          = config.getint('Data Attribute', 'patch_width')
    num_patch            = config.getint('Train Setting', 'num_patch')
    num_epoch            = config.getint('Train Setting', 'num_epoch')
    batch_size           = config.getint('Train Setting', 'batch_size')
    inside_FOV           = config.getboolean('Train Setting', 'inside_FOV')

    if not os.path.exists('./result/' + name_experiment):
        os.makedirs('./result/' + name_experiment, exist_ok=False)

    patches_img_train, patches_gt_train = loader.get_data_training(
        original_image_path=train_original_image, ground_truth_path=train_ground_truth, border_mask_path=train_border_mask,
        patch_height=patch_height, patch_width=patch_width, num_patch=num_patch, inside_FOV=inside_FOV)

    model = get_unet_model(patch_height, patch_width, 1)

    check_pointer = ModelCheckpoint(filepath='./result/' + name_experiment + '/best_weights.h5',
                                    verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
    lr_drop = LearningRateScheduler(lambda epoch: 0.0005 if epoch > 100 else 0.001)
    model.fit(patches_img_train, patches_gt_train, epochs=num_epoch, batch_size=batch_size, shuffle=True,
              validation_split=0.1, verbose=1, callbacks=[check_pointer, lr_drop])

    model.save_weights('./result/' + name_experiment + '/last_weights.h5', overwrite=True)

def test(config):
    name_experiment     = config.get('Experiment Name', 'name')
    test_original_image = config.get('Data Attribute', 'test_original_image')
    test_ground_truth   = config.get('Data Attribute', 'test_ground_truth')
    test_border_mask    = config.get('Data Attribute', 'test_border_mask')
    patch_height        = config.getint('Data Attribute', 'patch_height')
    patch_width         = config.getint('Data Attribute', 'patch_width')
    stride_height       = config.getint('Test Setting', 'stride_height')
    stride_width        = config.getint('Test Setting', 'stride_width')
    best_last           = config.get('Test Setting', 'best_last')

    patches_img_test, n_h, n_w, num_image = loader.get_data_testing_overlap(
        original_image_path=test_original_image, patch_height=patch_height,
        patch_width=patch_width, stride_height=stride_height, stride_width=stride_width)

    model = get_unet_model(patch_height, patch_width, 1)
    model.load_weights('./result/' + name_experiment + '/' + best_last + '_weights.h5')

    pred_patches = model.predict(patches_img_test, batch_size=32, verbose=1)
    pred_image = recompose(pred_patches, patch_height, patch_width, stride_height, stride_width,
                                 n_h, n_w, num_image, 584, 565)

    ground_truth = load_hdf5(test_ground_truth)
    pred_image = pred_image
    original_image = load_hdf5(test_original_image)
    border_mask = load_hdf5(test_border_mask)
    evaluate_metric(ground_truth, pred_image, original_image, border_mask,
                    threshold=0.5, path_experiment='./result/' + name_experiment)

if __name__ == '__main__':
    # 1\ Argument Parse
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('-e', '--exe_mode', default='train', help='The execution mode.(train/test)')
    parser.add_argument('-c', '--config', default='./config/config_0.ini', help='The config file of experiment.')
    args = parser.parse_args()

    # 2\ Configuration Parse
    config = configparser.ConfigParser()
    config.read(args.config)

    # 3\ Select the execution mode.
    if args.exe_mode == 'train':
        train(config)
    elif args.exe_mode == 'test':
        test(config)
    else:
        print('No mode named {}.'.format(args.exe_mode))