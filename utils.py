# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, \
    jaccard_score, f1_score, accuracy_score, recall_score, precision_score

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
    # 1\ Visual the result with original image and ground truth.
    true_image = np.repeat(y_true, 3, axis=-1)
    true_image[:, :, :, :][true_image >= threshold] = 255
    true_image[:, :, :, :][true_image < threshold] = 0
    score_image = np.repeat(y_score, 3, axis=-1)
    score_image[:, :, :, :][score_image >= threshold] = 255
    score_image[:, :, :, :][score_image < threshold] = 0
    score_image = score_image * mask
    image_data = np.concatenate((np.concatenate(original_image[0:5].astype(np.uint8), axis=1),
                                 np.concatenate(true_image[0:5].astype(np.uint8), axis=1),
                                 np.concatenate(score_image[0:5].astype(np.uint8), axis=1)), axis=0)
    img = Image.fromarray(image_data)
    img.save(path_experiment + '/result_image.png')

    # 2\ Get the masked y_score and masked y_true.
    y_score = y_score[:, :, :, 0]
    y_true = y_true[:, :, :, 0]
    mask = mask[:, :, :, 0]

    new_y_true = y_true[mask==1]
    new_y_score = y_score[mask==1]

    # 3\ Get the auroc and aupr.
    auroc = roc_auc_score(new_y_true, new_y_score)
    aupr = average_precision_score(new_y_true, new_y_score)

    # 3\ Get the threshold-specify metric.
    y_pred = np.zeros((new_y_score.shape))
    y_pred[new_y_score>=threshold] = 1

    jaccard = jaccard_score(new_y_true, y_pred)
    f1 = f1_score(new_y_true, y_pred)
    accuracy = accuracy_score(new_y_true, y_pred)
    precision = precision_score(new_y_true, y_pred)
    recall = recall_score(new_y_true, y_pred) # sensitivity

    # 4\ Write the performance.
    metric_str = 'Area under ROC curve: ' + str(auroc) + '\n' + 'Area under PR curve: ' + str(aupr) + '\n\n' + \
                 'For threshold: ' + str(threshold) + '\n' + \
                 'Jaccard similarity score: ' + str(jaccard) + '\n' + 'F1 score (F-measure): ' + str(f1) + '\n' + \
                 'Accuracy: ' + str(accuracy) + '\n' + 'Precision: ' + str(precision) + '\n' + \
                 'Recall: ' + str(recall)
    with open(path_experiment + '/performances.txt', 'w') as file:
        file.write(metric_str)