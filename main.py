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

def train(config):
    name_experiment = config.get('Experiment Name', 'name')
    train_original_image = config.get('Data Attribute', 'train_original_image')
    train_ground_truth = config.get('Data Attribute', 'train_ground_truth')
    patch_height = config.getint('Data Attribute', 'patch_height')
    patch_width = config.getint('Data Attribute', 'patch_width')
    patch_channel = config.getint('Data Attribute', 'num_channel')
    num_patch = config.getint('Training Setting', 'num_patch')
    inside_FOV = config.getboolean('Training Setting', 'inside_FOV')

    patches_img_train, patches_gt_train = loader.get_data_training(train_original_image=train_original_image,
                                                            train_ground_truth=train_ground_truth,
                                                            patch_height=patch_height,
                                                            patch_width=patch_width,
                                                            num_patch=num_patch,
                                                            inside_FOV=inside_FOV)

    visualize(group_images(patches_img_train[0:40, :, :, :], 5), './result/' + name_experiment + '/sample_input_img')
    visualize(group_images(patches_gt_train[0:40, :, :, :], 5), './result/' + name_experiment + '/sample_input_gt')
    patches_gt_train = masks_Unet(patches_gt_train)

    model = get_unet_model(patch_height, patch_width, patch_channel)
    model.to_json(fp = open('./' + name_experiment + '/' + name_experiment + '_architecture.json', 'w'))
    plot(model, to_file='./result/' + name_experiment + '/model.png')

    num_epoch = config.getint('Training Setting', 'num_epoch')
    batch_size = config.getint('Training Setting', 'batch_size')
    check_pointer = ModelCheckpoint(filepath='./result/' + name_experiment + '/best_weights.h5',
                                    verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
    lr_drop = LearningRateScheduler(lambda epoch: 0.005 if epoch>100 else 0.001)

    model.fit(patches_img_train, patches_gt_train, verbose=1,
              epochs=num_epoch, batch_size=batch_size, shuffle=True, validation_split=0.1,
              callbacks=[check_pointer, lr_drop])
    model.save_weights('./' + name_experiment + '/last_weights.h5', overwrite=True)

def test(config):
    DRIVE_test_imgs_original = config.get('Data Attribute', 'test_original_image')
    test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)
    full_img_height = test_imgs_orig.shape[2]
    full_img_width = test_imgs_orig.shape[3]
    DRIVE_test_border_masks = config.get('Data Attribute', 'test_border_mask')
    test_border_masks = load_hdf5(DRIVE_test_border_masks)
    patch_height = config.getint('Data Attribute', 'patch_height')
    patch_width = config.getint('Data Attribute', 'patch_width')
    patch_channel = config.getint('Data Attribute', 'num_channel')
    stride_height = config.getint('Test Setting', 'stride_height')
    stride_width = config.getint('Test Setting', 'stride_width')
    name_experiment = config.get('Experiment Name', 'name')
    path_experiment = './' + name_experiment + '/'
    N_visual = config.getint('Test Setting', 'N_group_visual')
    average_mode = config.getboolean('Test Setting', 'average_mode')
    gtruth= config.get('Data Attribute', 'test_ground_truth')
    img_truth= load_hdf5(gtruth)
    visualize(group_images(test_imgs_orig[0:20,:,:,:],5),'original')#.show()
    visualize(group_images(test_border_masks[0:20,:,:,:],5),'borders')#.show()
    visualize(group_images(img_truth[0:20,:,:,:],5),'gtruth')#.show()

    new_height = None
    new_width = None
    masks_test = None
    patches_masks_test = None
    if average_mode == True:
        patches_imgs_test, new_height, new_width, masks_test = loader.get_data_testing_overlap(
            DRIVE_test_imgs_original=DRIVE_test_imgs_original,
            DRIVE_test_groudTruth=config.get('Data Attribute', 'test_ground_truth'),
            Imgs_to_test=20,
            patch_height=patch_height,
            patch_width=patch_width,
            stride_height=stride_height,
            stride_width=stride_width)
    else:
        patches_imgs_test, patches_masks_test = get_data_testing(
            DRIVE_test_imgs_original=DRIVE_test_imgs_original,
            DRIVE_test_groudTruth=config.get('Data Attribute', 'test_ground_truth'),
            Imgs_to_test=20,
            patch_height=patch_height,
            patch_width=patch_width)

    best_last = config.get('Test Setting', 'best_last')
    model = model_from_json(open(path_experiment + name_experiment + '_architecture.json').read())
    model.load_weights(path_experiment + name_experiment + '_' + best_last + '_weights.h5')
    predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)
    print('predicted images size :' + str(predictions.shape))
    score = model.evaluate(patches_imgs_test, masks_Unet(patches_masks_test), verbose=0)
    print('Test score:', score[0], 'Test accuracy:', score[1])
    pred_patches = pred_to_imgs(predictions, patch_height, patch_width, 'original')

    # Elaborate and visualize the predicted images
    if average_mode == True:
        pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)  # predictions
        orig_imgs = loader.preprocess(test_imgs_orig[0:pred_imgs.shape[0], :, :, :])  # originals
        gtruth_masks = masks_test  # ground truth masks
    else:
        pred_imgs = recompone(pred_patches, 13, 12)  # predictions
        orig_imgs = recompone(patches_imgs_test, 13, 12)  # originals
        gtruth_masks = recompone(patches_masks_test, 13, 12)  # masks
    # Apply the DRIVE masks on the predictions  # set everything outside the FOV to zero!!
    kill_border(pred_imgs, test_border_masks)  # DRIVE MASK  #only for visualization
    orig_imgs = orig_imgs[:, :, 0:full_img_height, 0:full_img_width]
    pred_imgs = pred_imgs[:, :, 0:full_img_height, 0:full_img_width]
    gtruth_masks = gtruth_masks[:, :, 0:full_img_height, 0:full_img_width]
    print('Orig imgs shape: ' + str(orig_imgs.shape))
    print('pred imgs shape: ' + str(pred_imgs.shape))
    print('Gtruth imgs shape: ' + str(gtruth_masks.shape))
    visualize(group_images(orig_imgs, N_visual), path_experiment + 'all_originals')
    visualize(group_images(pred_imgs, N_visual), path_experiment + 'all_predictions')
    visualize(group_images(gtruth_masks, N_visual), path_experiment + 'all_groundTruths')

    # Visualize results comparing mask and prediction
    N_predicted = orig_imgs.shape[0]
    group = N_visual
    for i in range(int(N_predicted / group)):
        orig_stripe = group_images(orig_imgs[i * group:(i * group) + group, :, :, :], group)
        masks_stripe = group_images(gtruth_masks[i * group:(i * group) + group, :, :, :], group)
        pred_stripe = group_images(pred_imgs[i * group:(i * group) + group, :, :, :], group)
        total_img = np.concatenate((orig_stripe, masks_stripe, pred_stripe), axis=0)
        visualize(total_img, path_experiment + name_experiment + '_Original_GroundTruth_Prediction' + str(i))  # .show()

    # Evaluate the results
    y_score, y_true = pred_only_FOV(pred_imgs, gtruth_masks, test_border_masks)  # returns data only inside the FOV
    print('Calculating results only inside the FOV:')
    print('y scores pixels: ' + str(y_score.shape[0]) + ' (radius 270: 270*270*3.14==228906), including background around retina: '
          + str(pred_imgs.shape[0] * pred_imgs.shape[2] * pred_imgs.shape[3]) + ' (584*565==329960)')
    print('y true pixels: ' + str(y_true.shape[0]) + ' (radius 270: 270*270*3.14==228906), including background around retina: '
          + str(gtruth_masks.shape[2] * gtruth_masks.shape[3] * gtruth_masks.shape[0]) + ' (584*565==329960)')
    evaluate_metric(y_true, y_score, path_experiment)

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