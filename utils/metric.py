# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, \
    confusion_matrix, jaccard_score, f1_score, roc_curve
from matplotlib import pyplot as plt

def plot_roc_curve(y_true, y_scores, path_experiment):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    AUROC = roc_auc_score(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, '-', label='AUROC = %0.4f' % AUROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig(path_experiment + "ROC.png")
    return AUROC

def plot_pr_curve(y_true, y_scores, path_experiment):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    precision = np.fliplr([precision])[0]
    recall = np.fliplr([recall])[0]
    AUPR = np.trapz(precision, recall)
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

def evaluate_metric(y_true, y_score, path_experiment):
    AUROC = plot_roc_curve(y_true, y_score, path_experiment)
    AUPR = plot_pr_curve(y_true, y_score, path_experiment)

    threshold_confusion = 0.5
    y_pred = np.zeros((y_score.shape[0]))
    y_pred[y_score>threshold_confusion] = 1

    confusion = confusion_matrix(y_true, y_pred)
    jaccard_index = jaccard_score(y_true, y_pred)
    f1_score_value = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    accuracy, specificity, sensitivity, precision = 0, 0, 0, 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    if float(confusion[1, 1] + confusion[1, 0]) != 0:
        sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    if float(confusion[1, 1] + confusion[0, 1]) != 0:
        precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])

    metric_str = 'Area under ROC curve: ' + str(AUROC) + '\n' + \
                 'Area under PR curve: ' + str(AUPR) + '\n\n\n' + \
                 'For threshold: ' + str(threshold_confusion) + '\n' + \
                 'Confusion Matrix: ' + str(confusion) + '\n' + \
                 'Jaccard similarity score: ' + str(jaccard_index) + '\n' + \
                 'F1 score (F-measure): ' + str(f1_score_value) + '\n' + \
                 'Accuracy: ' + str(accuracy) + '\n' + \
                 'Sensitivity: ' + str(sensitivity) + '\n' + \
                 'Specificity: ' + str(specificity) + '\n' + \
                 'Precision: ' + str(precision)
    write_metric(metric_str, path_experiment)