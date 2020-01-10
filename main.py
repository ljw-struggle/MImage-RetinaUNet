# -*- coding: utf-8 -*-
import os
import random
import configparser
from model.unet_func import get_unet
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils.vis_utils import plot_model as plot
from keras.models import model_from_json
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve

from utils.loader import recompone
from utils.loader import recompone_overlap
from utils.loader import paint_border
from utils.loader import kill_border
from utils.loader import pred_only_FOV
from utils.loader import get_data_training
from utils.loader import get_data_testing
from utils.loader import get_data_testing_overlap
from utils.loader import my_PreProc
from utils.utils import *

from matplotlib import pyplot as plt

random.seed(10)
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def get_cofig():
    config = configparser.RawConfigParser()
    config.read('configuration.txt')
    path_data = config.get('data paths', 'path_local')
    name_experiment = config.get('experiment name', 'name')
    N_epochs = int(config.get('training settings', 'N_epochs'))
    batch_size = int(config.get('training settings', 'batch_size'))

def train():
    config = configparser.RawConfigParser()
    config.read('configuration.txt')
    path_data = config.get('data paths', 'path_local')
    name_experiment = config.get('experiment name', 'name')
    N_epochs = int(config.get('training settings', 'N_epochs'))
    batch_size = int(config.get('training settings', 'batch_size'))

    patches_imgs_train, patches_masks_train = get_data_training(
        DRIVE_train_imgs_original=path_data + config.get('data paths', 'train_imgs_original'),
        DRIVE_train_groudTruth=path_data + config.get('data paths', 'train_groundTruth'),  # masks
        patch_height=int(config.get('data attributes', 'patch_height')),
        patch_width=int(config.get('data attributes', 'patch_width')),
        N_subimgs=int(config.get('training settings', 'N_subimgs')),
        inside_FOV=config.getboolean('training settings','inside_FOV'))

    # Save a sample of what you're feeding to the neural network
    N_sample = min(patches_imgs_train.shape[0], 40)
    visualize(group_images(patches_imgs_train[0:N_sample, :, :, :], 5), './' + name_experiment + '/' + "sample_input_imgs")  # .show()
    visualize(group_images(patches_masks_train[0:N_sample, :, :, :], 5), './' + name_experiment + '/' + "sample_input_masks")  # .show()

    n_ch = patches_imgs_train.shape[1]
    patch_height = patches_imgs_train.shape[2]
    patch_width = patches_imgs_train.shape[3]
    model = get_unet(n_ch, patch_height, patch_width)
    print("Check: final output of the network:")
    print(model.output_shape)
    plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')
    json_string = model.to_json()
    open('./' + name_experiment + '/' + name_experiment + '_architecture.json', 'w').write(json_string)

    check_pointer = ModelCheckpoint(filepath='./' + name_experiment + '/' + name_experiment + '_best_weights.h5',
                                   verbose=1, monitor='val_loss', mode='auto',
                                   save_best_only=True)

    def step_decay(epoch):
        if epoch==100:
            return 0.005
        return 0.01

    lr_drop = LearningRateScheduler(step_decay)

    patches_masks_train = masks_Unet(patches_masks_train)  # reduce memory consumption
    model.fit(patches_imgs_train, patches_masks_train, epochs=N_epochs, batch_size=batch_size, verbose=1, shuffle=True,
              validation_split=0.1, callbacks=[check_pointer, lr_drop])
    model.save_weights('./' + name_experiment + '/' + name_experiment + '_last_weights.h5', overwrite=True)
    # score = model.evaluate(patches_imgs_test, masks_Unet(patches_masks_test), verbose=0)
    # print('Test score:', score[0], 'Test accuracy:', score[1])

def test():
    config = configparser.RawConfigParser()
    config.read('configuration.txt')
    path_data = config.get('data paths', 'path_local')
    DRIVE_test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
    test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)
    full_img_height = test_imgs_orig.shape[2]
    full_img_width = test_imgs_orig.shape[3]
    DRIVE_test_border_masks = path_data + config.get('data paths', 'test_border_masks')
    test_border_masks = load_hdf5(DRIVE_test_border_masks)
    patch_height = int(config.get('data attributes', 'patch_height'))
    patch_width = int(config.get('data attributes', 'patch_width'))
    stride_height = int(config.get('testing settings', 'stride_height'))
    stride_width = int(config.get('testing settings', 'stride_width'))
    assert (stride_height < patch_height and stride_width < patch_width)
    name_experiment = config.get('experiment name', 'name')
    path_experiment = './' + name_experiment + '/'
    Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
    N_visual = int(config.get('testing settings', 'N_group_visual'))
    average_mode = config.getboolean('testing settings', 'average_mode')
    gtruth= path_data + config.get('data paths', 'test_groundTruth')
    img_truth= load_hdf5(gtruth)
    visualize(group_images(test_imgs_orig[0:20,:,:,:],5),'original')#.show()
    visualize(group_images(test_border_masks[0:20,:,:,:],5),'borders')#.show()
    visualize(group_images(img_truth[0:20,:,:,:],5),'gtruth')#.show()

    # Load the data and divide in patches
    new_height = None
    new_width = None
    masks_test = None
    patches_masks_test = None
    if average_mode == True:
        patches_imgs_test, new_height, new_width, masks_test = get_data_testing_overlap(
            DRIVE_test_imgs_original=DRIVE_test_imgs_original,  # original
            DRIVE_test_groudTruth=path_data + config.get('data paths', 'test_groundTruth'),  # masks
            Imgs_to_test=int(config.get('testing settings', 'full_images_to_test')),
            patch_height=patch_height,
            patch_width=patch_width,
            stride_height=stride_height,
            stride_width=stride_width)
    else:
        patches_imgs_test, patches_masks_test = get_data_testing(
            DRIVE_test_imgs_original=DRIVE_test_imgs_original,  # original
            DRIVE_test_groudTruth=path_data + config.get('data paths', 'test_groundTruth'),  # masks
            Imgs_to_test=int(config.get('testing settings', 'full_images_to_test')),
            patch_height=patch_height,
            patch_width=patch_width)

    # Run the prediction of the patches
    best_last = config.get('testing settings', 'best_last')
    # Load the saved model
    model = model_from_json(open(path_experiment + name_experiment + '_architecture.json').read())
    model.load_weights(path_experiment + name_experiment + '_' + best_last + '_weights.h5')
    # Calculate the predictions
    predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)
    print("predicted images size :")
    print(predictions.shape)

    # Convert the prediction arrays in corresponding images
    pred_patches = pred_to_imgs(predictions, patch_height, patch_width, "original")

    # Elaborate and visualize the predicted images
    if average_mode == True:
        pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)  # predictions
        orig_imgs = my_PreProc(test_imgs_orig[0:pred_imgs.shape[0], :, :, :])  # originals
        gtruth_masks = masks_test  # ground truth masks
    else:
        pred_imgs = recompone(pred_patches, 13, 12)  # predictions
        orig_imgs = recompone(patches_imgs_test, 13, 12)  # originals
        gtruth_masks = recompone(patches_masks_test, 13, 12)  # masks
    # Apply the DRIVE masks on the predictions #set everything outside the FOV to zero!!
    kill_border(pred_imgs, test_border_masks)  # DRIVE MASK  #only for visualization
    # Back to original dimensions
    orig_imgs = orig_imgs[:, :, 0:full_img_height, 0:full_img_width]
    pred_imgs = pred_imgs[:, :, 0:full_img_height, 0:full_img_width]
    gtruth_masks = gtruth_masks[:, :, 0:full_img_height, 0:full_img_width]
    print("Orig imgs shape: " + str(orig_imgs.shape))
    print("pred imgs shape: " + str(pred_imgs.shape))
    print("Gtruth imgs shape: " + str(gtruth_masks.shape))
    visualize(group_images(orig_imgs, N_visual), path_experiment + "all_originals")  # .show()
    visualize(group_images(pred_imgs, N_visual), path_experiment + "all_predictions")  # .show()
    visualize(group_images(gtruth_masks, N_visual), path_experiment + "all_groundTruths")  # .show()

    # Visualize results comparing mask and prediction
    assert (orig_imgs.shape[0] == pred_imgs.shape[0] and orig_imgs.shape[0] == gtruth_masks.shape[0])
    N_predicted = orig_imgs.shape[0]
    group = N_visual
    assert (N_predicted % group == 0)
    for i in range(int(N_predicted / group)):
        orig_stripe = group_images(orig_imgs[i * group:(i * group) + group, :, :, :], group)
        masks_stripe = group_images(gtruth_masks[i * group:(i * group) + group, :, :, :], group)
        pred_stripe = group_images(pred_imgs[i * group:(i * group) + group, :, :, :], group)
        total_img = np.concatenate((orig_stripe, masks_stripe, pred_stripe), axis=0)
        visualize(total_img, path_experiment + name_experiment + "_Original_GroundTruth_Prediction" + str(i))  # .show()

    # Evaluate the results
    y_scores, y_true = pred_only_FOV(pred_imgs, gtruth_masks, test_border_masks)  # returns data only inside the FOV
    print("Calculating results only inside the FOV:")
    print("y scores pixels: " + str(
        y_scores.shape[0]) + " (radius 270: 270*270*3.14==228906), including background around retina: " + str(
        pred_imgs.shape[0] * pred_imgs.shape[2] * pred_imgs.shape[3]) + " (584*565==329960)")
    print("y true pixels: " + str(
        y_true.shape[0]) + " (radius 270: 270*270*3.14==228906), including background around retina: " + str(
        gtruth_masks.shape[2] * gtruth_masks.shape[3] * gtruth_masks.shape[0]) + " (584*565==329960)")

    # Area under the ROC curve
    fpr, tpr, thresholds = roc_curve((y_true), y_scores)
    AUC_ROC = roc_auc_score(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig(path_experiment + "ROC.png")

    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision, recall)
    plt.figure()
    plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig(path_experiment + "Precision_recall.png")

    # Confusion matrix
    threshold_confusion = 0.5
    print("\nConfusion matrix:  Custom threshold (for positive) of " + str(threshold_confusion))
    y_pred = np.empty((y_scores.shape[0]))
    for i in range(y_scores.shape[0]):
        if y_scores[i] >= threshold_confusion:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)
    accuracy = 0
    specificity = 0
    sensitivity =0
    precision = 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    if float(confusion[1, 1] + confusion[1, 0]) != 0:
        sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    if float(confusion[1, 1] + confusion[0, 1]) != 0:
        precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
    jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    with open(path_experiment + 'performances.txt', 'w') as file_perf:
        file_perf.write("Area under the ROC curve: " + str(AUC_ROC)
                        + "\nArea under Precision-Recall curve: " + str(AUC_prec_rec)
                        + "\nJaccard similarity score: " + str(jaccard_index)
                        + "\nF1 score (F-measure): " + str(F1_score)
                        + "\n\nConfusion matrix:"
                        + str(confusion)
                        + "\nACCURACY: " + str(accuracy)
                        + "\nSENSITIVITY: " + str(sensitivity)
                        + "\nSPECIFICITY: " + str(specificity)
                        + "\nPRECISION: " + str(precision))