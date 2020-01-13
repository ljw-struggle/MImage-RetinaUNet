# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, \
    confusion_matrix, jaccard_score, f1_score, classification_report
from matplotlib import pyplot as plt

def plot_roc_curve(y_true, y_score, path_experiment):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    AUROC = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, '-', label='AUROC = %0.4f' % AUROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig(path_experiment + "ROC.png")
    return AUROC

def plot_pr_curve(y_true, y_score, path_experiment):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    AUPR = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, '-', label='AUPR = %0.4f' % AUPR)
    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig(path_experiment + "PR.png")
    return AUPR

def write_metric(str, path_experiment):
    with open(path_experiment + 'performances.txt', 'w') as file:
        file.write(str)

def evaluate_metric(y_true, y_score, mask, threshold, path_experiment):
    """
    Evaluate.
    :param y_true: shape = (-1, 584, 565, 1)
    :param y_score: shape = (-1, 584, 565, 2)
    :param mask: shape = (-1, 584, 565, 1)
    :param threshold:
    :param path_experiment:
    :return:
    """
    # 1\ Get the masked y_score.
    y_score = y_score[:, :, :, 1]
    y_true = y_true[:, :, :, 0]
    mask = mask[:, :, :, 0]/255
    new_y_score = []
    new_y_true = []
    for i in range(y_true.shape[0]):
        temp_y_score = []
        temp_y_true = []
        for j in range(y_true.shape[1]):
            for k in range(y_true.shape[2]):
                if mask[i, j, k] == 255:
                    temp_y_score.append(y_score[i, j, k])
                    temp_y_true.append(y_true[i, j, k])

        new_y_score.append(temp_y_score)
        new_y_true.append(temp_y_true)

    # 2\ Get the AUROC and AUPR.
    AUROC = plot_roc_curve(y_true, y_score, path_experiment)
    AUPR = plot_pr_curve(y_true, y_score, path_experiment)

    # 3\ Get the confusion matrix.
    y_pred = np.zeros((y_score.shap))
    y_pred[y_score>threshold] = 1

    confusion = confusion_matrix(y_true, y_pred)
    jaccard_index = jaccard_score(y_true, y_pred)
    f1_score_value = f1_score(y_true, y_pred)
    accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])

    metric_str = 'Area under ROC curve: ' + str(AUROC) + '\n' + \
                 'Area under PR curve: ' + str(AUPR) + '\n\n\n' + \
                 'For threshold: ' + str(threshold) + '\n' + \
                 'Confusion Matrix: ' + str(confusion) + '\n' + \
                 'Jaccard similarity score: ' + str(jaccard_index) + '\n' + \
                 'F1 score (F-measure): ' + str(f1_score_value) + '\n' + \
                 'Accuracy: ' + str(accuracy) + '\n' + \
                 'Sensitivity: ' + str(sensitivity) + '\n' + \
                 'Specificity: ' + str(specificity) + '\n' + \
                 'Precision: ' + str(precision)
    write_metric(metric_str, path_experiment)