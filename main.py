# -*- coding: utf-8 -*-
import os, shutil, random, configparser, argparse
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model as plot
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from model.unet_func import get_unet_model
from utils.loader import *
from utils.utils import *
from utils.metric import *

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
random.seed(10)
shutil.copy('file', 'file_dir')
num_image, height, width, channel = 20, 584, 565, 3

def train(config):
    name_experiment      = config.get('Experiment Name', 'name')
    train_original_image = config.get('Data Attribute', 'train_original_image')
    train_ground_truth   = config.get('Data Attribute', 'train_ground_truth')
    train_border_mask   = config.get('Data Attribute', 'train_border_mask')
    patch_height         = config.getint('Data Attribute', 'patch_height')
    patch_width          = config.getint('Data Attribute', 'patch_width')
    num_patch            = config.getint('Training Setting', 'num_patch')
    num_epoch            = config.getint('Training Setting', 'num_epoch')
    batch_size           = config.getint('Training Setting', 'batch_size')
    inside_FOV           = config.getboolean('Training Setting', 'inside_FOV')

    patches_img_train, patches_gt_train = loader.get_data_training(
        original_image_path=train_original_image, ground_truth_path=train_ground_truth, border_mask_path=train_border_mask,
        patch_height=patch_height, patch_width=patch_width, num_patch=num_patch, inside_FOV=inside_FOV)

    visualize(group_images(patches_img_train[0:40, :, :, :], 5), './result/' + name_experiment + '/sample_input_img')
    visualize(group_images(patches_gt_train[0:40, :, :, 0:1], 5), './result/' + name_experiment + '/sample_input_gt')

    model = get_unet_model(patch_height, patch_width, 1)
    model.to_json(fp = open('./result/' + name_experiment + '/architecture.json', 'w'))
    plot(model, to_file='./result/' + name_experiment + '/model.png')

    check_pointer = ModelCheckpoint(filepath='./result/' + name_experiment + '/best_weights.h5',
                                    verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
    lr_drop = LearningRateScheduler(lambda epoch: 0.005 if epoch > 100 else 0.001)
    model.fit(patches_img_train, patches_gt_train, epochs=num_epoch, batch_size=batch_size, shuffle=True,
              validation_split=0.1, verbose=1, callbacks=[check_pointer, lr_drop])
    model.save_weights('./' + name_experiment + '/last_weights.h5', overwrite=True)

def test(config):
    name_experiment     = config.get('Experiment Name', 'name')
    test_original_image = config.get('Data Attribute', 'test_original_image')
    test_ground_truth   = config.get('Data Attribute', 'test_ground_truth')
    test_border_mask    = config.get('Data Attribute', 'test_border_mask')
    patch_height        = config.getint('Data Attribute', 'patch_height')
    patch_width         = config.getint('Data Attribute', 'patch_width')
    best_last           = config.get('Test Setting', 'best_last')
    average_mode        = config.getboolean('Test Setting', 'average_mode')
    stride_height       = config.getint('Test Setting', 'stride_height')
    stride_width        = config.getint('Test Setting', 'stride_width')

    if average_mode == True:
        gt_img = load_hdf5(test_ground_truth)
        patches_img_test, n_h, n_w, num = loader.get_data_testing_overlap(
            original_image_path=test_original_image, patch_height=patch_height,
            patch_width=patch_width, stride_height=stride_height, stride_width=stride_width)
        model = model_from_json(open('./result/' + name_experiment + '/architecture.json').read())
        model.load_weights('./result/' + name_experiment + '/' + best_last + '_weights.h5')
        pred_patches = model.predict(patches_img_test, batch_size=32, verbose=2)
        pred_patches = int(pred_patches[:, :, :, 1] > 0.5)
        pred_img = recompose_overlap(pred_patches, patch_height, patch_width, stride_height, stride_width,
                                     n_h, n_w, num, 584, 565)
    else:
        gt_img = load_hdf5(test_ground_truth)
        patches_img_test, n_h, b_w, num = loader.get_data_testing(
            original_image_path=test_original_image, patch_height=patch_height, patch_width=patch_width)
        model = model_from_json(open('./result/' + name_experiment + '/architecture.json').read())
        model.load_weights('./result/' + name_experiment + '/' + best_last + '_weights.h5')
        pred_patches = model.predict(patches_img_test, batch_size=32, verbose=2)
        pred_patches= int(pred_patches[:, :, :, 1] > 0.5)
        pred_img = recompose(pred_patches, patch_height, patch_width, n_h, n_w, num, 584, 565)

    test_border_mask = load_hdf5(test_border_mask)

    def kill_border(data, original_imgs_border_masks):
        height = data.shape[2]
        width = data.shape[3]
        for i in range(data.shape[0]):
            for x in range(width):
                for y in range(height):
                    if inside_FOV_DRIVE(i, x, y, original_imgs_border_masks) == False:
                        data[i, :, y, x] = 0.0

    kill_border(pred_img, test_border_mask)
    y_score, y_true = pred_only_FOV(pred_img[:, 0:584, 0:565, :], gt_img[:, 0:584, 0:565, :], test_border_mask)
    evaluate_metric(y_true, y_score, './result/' + name_experiment)

def pred_only_FOV(data_imgs,data_masks,original_imgs_border_masks):
    height = data_imgs.shape[2]
    width = data_imgs.shape[3]
    new_pred_imgs = []
    new_pred_masks = []
    for i in range(data_imgs.shape[0]):
        for x in range(width):
            for y in range(height):
                if inside_FOV_DRIVE(i,x,y,original_imgs_border_masks)==True:
                    new_pred_imgs.append(data_imgs[i,:,y,x])
                    new_pred_masks.append(data_masks[i,:,y,x])
    new_pred_imgs = np.asarray(new_pred_imgs)
    new_pred_masks = np.asarray(new_pred_masks)
    return new_pred_imgs, new_pred_masks

def inside_FOV_DRIVE(i, x, y, DRIVE_masks):
    if (x >= DRIVE_masks.shape[3] or y >= DRIVE_masks.shape[2]):
        return False
    if (DRIVE_masks[i,0,y,x]>0):
        return True
    return False

if __name__ == '__main__':
    # 1\ Argument Parse
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('-e', '--exe_mode', default='train', help='The execution mode.(train/test)')
    parser.add_argument('-c', '--config', default='./config/configuration.txt', help='The config file of experiment.')
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