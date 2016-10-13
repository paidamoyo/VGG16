#!/usr/bin/env python

import numpy as np
from sklearn import metrics
import logging

from functions.aux import make_directory


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == labels) /
        predictions.shape[0])


def auc_roc(predictions, labels):
    try:
        fpr, tpr, _ = metrics.roc_curve(np.array(labels), np.argmax(predictions, 1), pos_label=1)
    except ValueError:  # if all predicted labels are the same
        print('All predicted labels are the same')
        return -1, -1, -1
    if type(fpr) is int and type(tpr) is int:
        auc = metrics.auc(fpr, tpr)
        return auc, fpr, tpr
    else:
        print('fpr and tpr were not returned properly.')
        return -1, -1, -1



def print_log(string, logging):
    print(string)
    logging.info(string)


def record_metrics(loss, acc, batch_y, logging, step, split, params):
    if step is not None or loss is not None:
        print_log("Batch Number " + str(step) + ", Image Loss= " + "{:.6f}".format(loss), logging)
    auc, fpr, tpr = auc_roc(acc, batch_y)
    print_log("Error: %.1f%%" % error_rate(acc, batch_y) + ", AUC= %.3f" % auc + ", FPR= %.3f" % fpr +
              ", TPR= %.3f" % tpr, logging)
    print("Predicted Labels: ", np.argmax(acc, 1).tolist(), logging)
    print("True Labels: ", batch_y)
    if split is not None:
        print_log("Training Split: ", split)
    print_log("Fraction of Positive Predictions: %d / %d" %
              (np.count_nonzero(np.argmax(acc, 1)), params['batch_size']), logging)


def setup_metrics(flags, aux_filenames, folder):
    flags['restore_directory'] = flags['aux_directory'] + flags['model_directory']
    flags['logging_directory'] = flags['restore_directory'] + folder
    make_directory(flags['logging_directory'])
    logging.basicConfig(filename=flags['logging_directory'] + aux_filenames + '.log', level=logging.INFO)
    return logging