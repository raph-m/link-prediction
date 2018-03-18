import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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


# plotting feature importances
def plot_importance(rf, features_dict, features_index, name):
    # plot settings
    sns.set_style("darkgrid")
    mpl.rcParams['figure.dpi'] = 200
    # mpl.rcParams['figure.tight_layout'] = True
    path_to_plot = "models/plots/"

    # fetch mean importances
    importances = rf.feature_importances_
    # compute std using each estimator in the forest
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
                 axis=0)
    # argsort the values
    index = list(map(int, np.argsort(importances)[::-1]))
    # Plot the feature importances of the rf
    plt.figure()
    # get axis
    fig, ax = plt.subplots(figsize=(6, 3))
    # add space for x labels
    plt.subplots_adjust(bottom=0.30)
    plt.title("Feature importances")
    # get number of features
    nb_features = len(features_dict)
    # plot with error bars
    plt.bar(range(nb_features), importances[index],
            color="r", yerr=std[index], align="center")
    # create x axis tickers
    plt.xticks(range(nb_features), index)
    # get feature names in right order
    index_features_sorted = np.array(features_index)[index]
    feature_names = list(map(lambda x: features_dict[x], index_features_sorted))
    # font dict to control x tickers labels
    ax.set_xticklabels(feature_names, rotation=40, fontsize=9, ha='right')
    plt.xlim([-1, nb_features])
    plt.ylim([0, 0.8])
    plt.savefig(path_to_plot + name)
    plt.show()
