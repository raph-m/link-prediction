import time

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from skopt import gp_minimize

from code.models.tools import f1_score
from code.models.tuning.objective_function import ObjectiveFunction

# path
path_to_data = "../../../data/"
path_to_submissions = "../../../submissions/"

# load data
training = pd.read_csv(path_to_data + "training_features.txt")
testing = pd.read_csv(path_to_data + "testing_features.txt")
del training["my_index"]
del testing["my_index"]

# replace inf in shortest_path with -1
training['shortest_path'] = training['shortest_path'].replace([float('inf')], [-1])
testing['shortest_path'] = testing['shortest_path'].replace([float('inf')], [-1])

my_features_string = [
    "overlap_title",
    "date_diff",
    "common_author",
    "journal_similarity",
    "overlapping_words_abstract",
    "cosine_distance",
    "shortest_path",
    "jaccard",
    "adar",
    "preferential_attachment",
    "resource_allocation_index",
    "out_neighbors",
    "in_neighbors",
    "common_neighbors"
]
my_features_index = []
my_features_dic = {}

target = 0
for i in range(len(training.columns)):
    if training.columns[i] == "target":
        target = i
    elif training.columns[i] in my_features_string:
        my_features_dic.update({len(my_features_index): training.columns[i]})
        my_features_index.append(i)

# separating features and labels
training_val = training.values
testing_val = testing.values
X_train, Y_train = training_val[:, my_features_index].astype(float), training_val[:, target].astype(int)
X_test = testing_val[:, my_features_index]


# function to optimize (too costly --> feature selection, subsampling)
def objective_svm(x):
    C_in, gamma_in = x[0] ** 2, x[1] ** 2
    svm_classifier = SVC(C=C_in, cache_size=200,
                         class_weight=None,
                         coef0=0.0,
                         decision_function_shape='ovr',
                         degree=3, gamma=gamma_in,
                         kernel='rbf',
                         max_iter=-1,
                         probability=False,
                         random_state=None,
                         shrinking=True,
                         tol=0.001,
                         verbose=False)
    k = 5
    kf = StratifiedKFold(k)
    i = 0
    score = 0
    for train_index, test_index in kf.split(X_train, Y_train):
        svm_classifier.fit(X_train[train_index], Y_train[train_index])
        Y_pred = svm_classifier.predict(X_train[test_index])
        score += f1_score(Y_train[test_index], Y_pred)
        i += 1
    return score


# Bayesian Optimization (too costly --> feature selection, subsampling)
f_bo = ObjectiveFunction(objective_svm)
t0 = time.time()
res = gp_minimize(f_bo, [(10**(-9), 10), (10**(-9), 0.1)], n_jobs=4)
t1 = time.time()
print("The total time with BO is : " + str(t1 - t0) + " seconds")
print('best score BO :', -res.fun)
print('best parameters BO:', res.x)


