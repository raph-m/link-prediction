import numpy as np
import pandas as pd


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


def load_data(my_features_string):
    # path
    path_to_data = "data/"

    # feature tracking utils
    my_features_index = []
    my_features_dic = {}

    # load raw data
    training = pd.read_csv(path_to_data + "training_features.txt")
    testing = pd.read_csv(path_to_data + "testing_features.txt")

    del training["my_index"]
    del testing["my_index"]

    # track features and target
    target = 0
    for i in range(len(training.columns)):
        if training.columns[i] == "target":
            target = i

    Y_train = training.values[:, target].astype(int)

    del training["target"]

    for i in range(len(training.columns)):
        if training.columns[i] in my_features_string:
            my_features_dic.update({i: training.columns[i]})
            my_features_index.append(i)

    # separating features and labels
    training_val = training.values
    testing_val = testing.values
    X_train = training_val[:, my_features_index].astype(float)
    X_test = testing_val[:, my_features_index]

    del training_val
    del testing_val

    print(training.head())
    print(testing.head())

    return X_train, X_test, Y_train, my_features_index, my_features_dic


