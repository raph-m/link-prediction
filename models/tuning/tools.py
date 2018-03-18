import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Function to help plotting the results of our cross-validation for the random forest algorithm.
def triple_ticker_grid(tuned_parameters, parameter_1, parameter_2, parameter_3):
    ticker_labels = []
    for i in tuned_parameters[parameter_1]:
        for j in tuned_parameters[parameter_2]:
            for k in tuned_parameters[parameter_3]:
                ticker_labels.append(str((i, j, k)))
    return ticker_labels


def double_ticker_grid(tuned_parameters, parameter_1, parameter_2):
    ticker_labels = []
    for i in tuned_parameters[parameter_1]:
        for j in tuned_parameters[parameter_2]:
            ticker_labels.append(str((i, j)))
    return ticker_labels


# fucntion for plotting the results of the grid search
def plot_grid(metrics, params, param_names, index, name):
    # plot settings
    sns.reset_orig()
    mpl.rcParams['figure.dpi'] = 200
    path_to_plot = "models/tuning/plots/"

    # For the test set
    plt.figure(figsize=(30, 30))
    f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
    x = range(len(metrics['params']))
    ax1.plot(x, list(metrics['mean_test_precision']), '--',
             label='mean test precision', color='g')
    ax1.plot(x, list(metrics['mean_test_recall']), '-.',
             label='mean test recall', color='r')
    ax1.plot(x, list(metrics['mean_test_roc_auc']), '-o',
             label='mean test roc auc', color='b')
    ax1.plot(x, list(metrics['mean_test_accuracy']), '-*',
             label='mean test accuracy', color='purple')
    ax1.plot(x, list(metrics['mean_test_f1']), '-',
             label='mean test f1', color='orange')
    y_13 = np.arange(0.9, 1.05, 0.05)
    x_13 = np.repeat(index, len(y_13))

    ax1.plot(x_13, y_13, '-.', color='pink', lw=2.0)
    plt.ylim([0.95, 1.0])
    ax1.legend(bbox_to_anchor=(-0.2, 0.2), loc=4, borderaxespad=0., fontsize=7)
    ax1.set_ylabel('Test metrics')
    plt.subplots_adjust(left=0.40, bottom=0.15)
    plt.title(" ".join(param_names) + " Grid Search")

    # Setting the labels for the x-axis (gridsearch combination)
    # x_ticks_labels = double_parameter_cross_validation(params, 
    #                                                    'max_depth', 
    #                                                    'min_samples_leaf', 
    #                                                    'n_estimators')
    # Set number of ticks for x-axis
    ax1.set_xticks([])

    # Set ticks labels for x-axis
    # ax1.set_xticklabels(x_ticks_labels, rotation=70, fontsize=6);

    # For the train set
    ax2.plot(x, list(metrics['mean_train_precision']), '--',
             label='mean train precision', color='c')
    ax2.plot(x, list(metrics['mean_train_recall']), '-.',
             label='mean train recall', color='m')
    ax2.plot(x, list(metrics['mean_train_roc_auc']), '-o',
             label='mean train roc auc', color='y')
    ax2.plot(x, list(metrics['mean_train_accuracy']), '-*',
             label='mean train accuracy', color='k')
    ax2.plot(x, list(metrics['mean_train_f1']), '-',
             label='mean train f1', color='orange')
    ax2.plot(x_13, y_13, '-.', color='pink', lw=2.0)
    ax2.legend(bbox_to_anchor=(-0.2, 0.2), loc=4, borderaxespad=0., fontsize=6)
    plt.ylim([0.95, 1.0])
    if len(param_names) == 2:
        x_ticks_labels = double_ticker_grid(params,
                                            param_names[0],
                                            param_names[1])
    if len(param_names) == 3:
        x_ticks_labels = triple_ticker_grid(params,
                                            param_names[0],
                                            param_names[1],
                                            param_names[2])
    # Set number of ticks for x-axis
    ax2.set_xticks(x)
    # Set ticks labels for x-axis
    ax2.set_xticklabels(x_ticks_labels, rotation=70, fontsize=7, ha='right')
    ax2.set_ylabel('Train metrics')
    plt.savefig(path_to_plot + name)
    plt.show()
