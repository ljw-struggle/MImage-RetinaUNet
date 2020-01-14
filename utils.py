# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, \
    confusion_matrix, jaccard_score, f1_score
from matplotlib import pyplot as plt
from PIL import Image

def recompose(preds, patch_h, patch_w, stride_h, stride_w, n_h, n_w,
                      num_image=20, full_height=584, full_width=565):
    full_prob = np.zeros((num_image, patch_h+stride_h*(n_h-1), patch_w+stride_w*(n_w-1), 1))
    full_sum = np.zeros((num_image, patch_h+stride_h*(n_h-1), patch_w+stride_w*(n_w-1), 1))

    for i in range(num_image):
        for h in range(n_h):
            for w in range(n_w):
                full_prob[i, h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=preds[i*n_h*n_w+h*n_w+w]
                full_sum[i, h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=1

    return (full_prob/full_sum)[:, 0:full_height, 0:full_width, :]

def evaluate_metric(y_true, y_score, original_image, mask, threshold, path_experiment):
    """
    Evaluate.
    :param y_true: shape = (-1, 584, 565, 1)
    :param y_score: shape = (-1, 584, 565, 1)
    :param original_image: shape = (-1, 584, 565, 3)
    :param mask: shape = (-1, 584, 565, 1)
    :param threshold: threshold
    :param path_experiment: path of experiment.
    """
    true_image = np.repeat(y_true, 3, axis=-1)
    true_image[:, :, :, :][true_image >= threshold] = 255
    true_image[:, :, :, :][true_image < threshold] = 0
    score_image = np.repeat(y_score, 3, axis=-1)
    score_image[:, :, :, :][score_image >= threshold] = 255
    score_image[:, :, :, :][score_image < threshold] = 0
    score_image = score_image*mask
    image_data = np.concatenate((np.concatenate(original_image[0:5].astype(np.uint8), axis=1),
                                 np.concatenate(true_image[0:5].astype(np.uint8), axis=1),
                                 np.concatenate(score_image[0:5].astype(np.uint8), axis=1)), axis=0)
    img = Image.fromarray(image_data)
    img.save(path_experiment + '/result_image.png')

    # 1\ Flatten the y_true, y_score and Get the masked y_score.
    y_score = y_score[:, :, :, 0]
    y_true = y_true[:, :, :, 0]
    mask = mask[:, :, :, 0]

    new_y_score = []
    new_y_true = []
    for i in range(y_true.shape[0]):
        for j in range(y_true.shape[1]):
            for k in range(y_true.shape[2]):
                if mask[i, j, k] == 1:
                    new_y_score.append(y_score[i, j, k])
                    new_y_true.append(y_true[i, j, k])

    new_y_true = np.array(new_y_true)
    new_y_score = np.array(new_y_score)

    # 2\ Get the AUROC and AUPR.
    AUROC = plot_roc_curve(new_y_true, new_y_score, path_experiment)
    AUPR = plot_pr_curve(new_y_true, new_y_score, path_experiment)

    # 3\ Get the confusion matrix.
    y_pred = np.zeros((new_y_score.shape))
    y_pred[new_y_score>=threshold] = 1

    confusion = confusion_matrix(new_y_true, y_pred)
    jaccard_index = jaccard_score(new_y_true, y_pred)
    f1_score_value = f1_score(new_y_true, y_pred)
    accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])

    metric_str = 'Area under ROC curve: ' + str(AUROC) + '\n' + \
                 'Area under PR curve: ' + str(AUPR) + '\n\n' + \
                 'For threshold: ' + str(threshold) + '\n' + \
                 '  Confusion Matrix: ' + '\n' + str(confusion) + '\n' + \
                 '  Jaccard similarity score: ' + str(jaccard_index) + '\n' + \
                 '  F1 score (F-measure): ' + str(f1_score_value) + '\n' + \
                 '  Accuracy: ' + str(accuracy) + '\n' + \
                 '  Sensitivity: ' + str(sensitivity) + '\n' + \
                 '  Specificity: ' + str(specificity) + '\n' + \
                 '  Precision: ' + str(precision)
    with open(path_experiment + '/performances.txt', 'w') as file:
        file.write(metric_str)

def plot_roc_curve(y_true, y_score, path_experiment):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    AUROC = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, '-', label='AUROC = %0.4f' % AUROC)
    plt.title('ROC curve')
    plt.xlabel('FPR (False Positive Rate)')
    plt.ylabel('TPR (True Positive Rate)')
    plt.legend(loc='lower right')
    plt.savefig(path_experiment + '/roc.png')
    return AUROC

def plot_pr_curve(y_true, y_score, path_experiment):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    AUPR = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, '-', label='AUPR = %0.4f' % AUPR)
    plt.title('Precision - Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower right')
    plt.savefig(path_experiment + '/pr.png')
    return AUPR