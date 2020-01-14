# -*- coding: utf-8 -*-
import os, configparser, argparse, random
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from model import get_unet_model
from loader import loader
from utils import recompose, evaluate_metric

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

    if not os.path.exists('./result/' + name_experiment):
        os.makedirs('./result/' + name_experiment, exist_ok=False)

    patches_img_train, patches_gt_train = loader.get_data_training(
        original_image_path=train_original_image, ground_truth_path=train_ground_truth, border_mask_path=train_border_mask,
        patch_height=patch_height, patch_width=patch_width, num_patch=num_patch, inside_mask=False)
    exit(0)
    model = get_unet_model(patch_height, patch_width, 1)

    check_pointer = ModelCheckpoint(filepath='./result/' + name_experiment + '/best_weights.h5',
                                    verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
    # lr_drop = LearningRateScheduler(lambda epoch: 0.0001*(0.9**epoch))
    lr_drop = ReduceLROnPlateau(patience=5)
    early_stop = EarlyStopping(patience=10)

    image_data = np.concatenate((np.concatenate(patches_img_train[0:10, :, :, :], axis=1),
                                 np.concatenate(patches_gt_train[0:10, :, :, :], axis=1)), axis=0)
    image_data = np.repeat((image_data*255).astype(np.uint8), 3, axis=-1)
    img = Image.fromarray(image_data)
    img.save('./result/' + name_experiment + '/input_sample.png')

    model.fit(patches_img_train, patches_gt_train, epochs=num_epoch, batch_size=batch_size, shuffle=True,
              validation_split=0.1, verbose=1, callbacks=[check_pointer, lr_drop, early_stop])

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

    if not os.path.exists('./result/' + name_experiment):
        os.makedirs('./result/' + name_experiment, exist_ok=False)

    patches_img_test, n_h, n_w, num_image = loader.get_data_testing_overlap(
        original_image_path=test_original_image, patch_height=patch_height,
        patch_width=patch_width, stride_height=stride_height, stride_width=stride_width)

    model = get_unet_model(patch_height, patch_width, 1)
    model.load_weights('./result/' + name_experiment + '/' + best_last + '_weights.h5')

    pred_patches = model.predict(patches_img_test, batch_size=32, verbose=1)
    pred_image = recompose(pred_patches, patch_height, patch_width, stride_height, stride_width,
                                 n_h, n_w, num_image, 584, 565)

    original_image = loader.load_hdf5(test_original_image)
    ground_truth = loader.load_hdf5(test_ground_truth)
    border_mask = loader.load_hdf5(test_border_mask)
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

    # 3\ Configure the Check the Environment.
    tf.debugging.set_log_device_placement(False)
    tf.config.set_soft_device_placement(True)
    cpu_devices = tf.config.experimental.list_physical_devices('CPU')
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    print('Check the Deep learning Environment:', flush=True)
    print('GPU count:{}, Memory growth:{}, Soft device placement:{} ...'.format(len(gpu_devices),True,True), flush=True)

    # 3\ Select the execution mode.
    if args.exe_mode == 'train':
        train(config)
    elif args.exe_mode == 'test':
        test(config)
    else:
        print('No mode named {}.'.format(args.exe_mode))