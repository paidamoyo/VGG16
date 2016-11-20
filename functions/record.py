#!/usr/bin/env python

import numpy as np
from sklearn import metrics
import logging
import datetime
import os

from functions.aux import check_str


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])


def auc_roc(predictions, labels):  # must input np.arrays
    total = len(labels)
    predictions = np.argmax(predictions, 1)
    total_pos = np.count_nonzero(labels)
    total_neg = total - total_pos
    tpv = np.sum([np.equal(p, l) for p, l in zip(predictions, labels) if l == 1])
    fpv = total_pos - tpv
    tnv = np.sum([np.equal(p, l) for p, l in zip(predictions, labels) if l == 0])
    fnv = total_neg - tnv
    fpr, tpr, _ = metrics.roc_curve(labels, predictions, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc, tpv/total, fpv/total, tnv/total, fnv/total, total


def print_log(string):
    print(string)
    logging.info(string)


def record_metrics(loss, acc, batch_y, step, split, flags):
    if step is not None or loss is not None:
        print_log("Batch Number: " + str(step) + ", Image Loss= " + "{:.6f}".format(loss))
    if batch_y is not None or acc is not None:
        print_log(np.squeeze(batch_y))
        print_log(np.argmax(acc, 1))
        auc, tp, fp, tn, fn, total = auc_roc(acc, batch_y)
        print_log("Error: %.1f%%" % error_rate(acc, batch_y) + ", AUC= %.3f" % auc + ", TP= %.3f" % tp +
                  ", FP= %.3f" % fp + ", TN= %.3f" % tn + ", FN= %.3f" % fn)
    if split is not None:
        print("Training Split: ", split)




def make_directory(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)