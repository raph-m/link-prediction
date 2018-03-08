import numpy as np


def binary_error(preds, train_data):
    labels = train_data.get_label()
    return 'error', np.mean(labels != (preds > 0.5)), False


def f1_score_lgbm(preds, train_data):
    labels = train_data.get_label()
    tp = np.sum(labels[labels == 1] == (preds[labels == 1] > 0.5))
    tn = np.sum(labels[labels == 0] == (preds[labels == 0] > 0.5))
    fp = np.sum(labels[labels == 1] != (preds[labels == 1] > 0.5))
    fn = np.sum(labels[labels == 0] != (preds[labels == 0] > 0.5))
    p = tp / (tp + fp)
    r = tp / (tp + fn)

    return 'f1 score', 2 * p * r / (p + r), False


def f1_score(preds, labels):
    tp = np.sum(labels[labels == 1] == preds[labels == 1])
    tn = np.sum(labels[labels == 0] == preds[labels == 0])
    fp = np.sum(labels[labels == 1] != preds[labels == 1])
    fn = np.sum(labels[labels == 0] != preds[labels == 0])
    p = tp / (tp + fp)
    r = tp / (tp + fn)

    return 2 * p * r / (p + r)


