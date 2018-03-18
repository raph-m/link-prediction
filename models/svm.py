import datetime

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from models.tools import f1_score, load_data

# path
path_to_data = "data/"
path_to_submissions = "submissions/"
path_to_stacking = "stacking/"
path_to_plots = "models/plots"

# used features

my_features_string = [
    "date_diff",
    "overlap_title",
    "common_author",
    # "score_1_2",
    # "score_2_1",
    "cosine_distance",
    # "journal_similarity",
    # # "overlapping_words_abstract",
    # "jaccard",
    # "adar",
    "preferential_attachment",
    # "resource_allocation_index",
    "out_neighbors",
    "in_neighbors",
    "common_neighbors",
    # # "shortest_path",
    # "popularity",
    # # "paths_of_length_one"
    # "katz"
    # "katz_2"
]

my_features_acronym = ["_".join(list(map(lambda x: x[0], string.split('_')))) for string in my_features_string]

# load data

(X_train,
 X_test,
 Y_train,
 my_features_index,
 my_features_dic) = load_data(my_features_string)

# normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# tuned hyperparameters
parameters = {
    'C': 0.1,
    'gamma': 0.01,
    'kernel': "linear"
}

# print user info
now = datetime.datetime.now()
print("date: " + str(now))
print("features: " + str(my_features_string))
print("model: SVM")
print("parameters:")
print(parameters)
print("cross validation:")

# instantiate classifier
svm_classifier = svm.SVC(C=parameters['C'],
                         gamma=parameters['gamma'],
                         kernel=parameters['kernel'],
                         probability=True,
                         verbose=1)

# instantiate Kfold and predictions placeholder
k = 2
kf = StratifiedKFold(k)
predictions = np.zeros((X_test.shape[0], k))
predictions_test = np.zeros((X_test.shape[0], k))
predictions_train = np.zeros(X_train.shape[0])
i = 0

# for each fold store predictions on test set and print validation results
for train_index, test_index in kf.split(X_train, Y_train):
    svm_classifier.fit(X_train[train_index], Y_train[train_index])
    Y_pred = svm_classifier.predict(X_train[test_index])
    Y_pred_train = svm_classifier.predict(X_train[train_index])
    predictions[:, i] = svm_classifier.predict(X_test)
    predictions_test[:, i] = svm_classifier.predict_proba(X_test)[:, 1]
    predictions_train[test_index] = svm_classifier.predict_proba(X_train[test_index])[:, 1]
    print("train: " + str(f1_score(Y_train[train_index], Y_pred_train)))
    print("test: " + str(f1_score(Y_train[test_index], Y_pred)))
    i += 1

# save submission file
Y_test = (np.sum(predictions, axis=1) > 2.5).astype(int)
submission = pd.DataFrame(Y_test)
submission.to_csv(
    path_or_buf=path_to_submissions + "-".join(my_features_acronym) + "SVM.csv",
    index=True,
    index_label="id",
    header=["category"]
)

# save probabilities for stacking
stacking_logits_test = np.sum(predictions_test, axis=1)
stacking_test = pd.DataFrame(stacking_logits_test)
stacking_test.to_csv(
    path_or_buf=path_to_stacking + "svmlinear_test" + ".csv",
    index=True,
    index_label="id",
    header=["category"]
)

stacking_train = pd.DataFrame(predictions_train)
stacking_train.to_csv(
    path_or_buf=path_to_stacking + "svmlinear_train" + ".csv",
    index=True,
    index_label="id",
    header=["category"]
)
