import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold

from models.tools import f1_score

# path
path_to_data = "data/"
path_to_submissions = "submissions/"

# parameters
parameters = {
    "max_iter": 100,
    "tol": 1e-6,
    "penalty": "l2"
}

# load data
training = pd.read_csv(path_to_data + "training_features.txt")
testing = pd.read_csv(path_to_data + "testing_features.txt")
del training["my_index"]
del testing["my_index"]

# replace inf in shortest_path with -1
training['shortest_path'] = training['shortest_path'].replace([float('inf')], [-1])
testing['shortest_path'] = testing['shortest_path'].replace([float('inf')], [-1])

my_features_string = [
    "date_diff",
    "overlap_title",
    "common_author",
    "score_1_2",
    # "score_2_1",
    # "cosine_distance",
    # "journal_similarity",
    # "overlapping_words_abstract",
    # "jaccard",
    # "adar",
    # "preferential_attachment",
    # "resource_allocation_index",
    # "out_neighbors",
    # "in_neighbors",
    # "common_neighbors",
    "shortest_path",
    "popularity"
]
my_features_index = []
my_features_dic = {}
my_features_acronym = ["_".join(list(map(lambda x: x[0], string.split('_')))) for string in my_features_string]

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

now = datetime.datetime.now()
print("date: " + str(now))
print("features: " + str(my_features_string))
print("model: Random Forest")
print("parameters:")
print(parameters)
print("cross validation:")

LogReg = LogisticRegressionCV(max_iter=parameters['max_iter'],
                              tol=parameters['tol'],
                              penalty=parameters['penalty'])
k = 5
kf = KFold(k)
predictions = np.zeros((X_test.shape[0], k))
i = 0

for train_index, test_index in kf.split(X_train, Y_train):
    LogReg.fit(X_train[train_index], Y_train[train_index])
    Y_pred = LogReg.predict(X_train[test_index])
    Y_pred_train = LogReg.predict(X_train[train_index])
    predictions[:, i] = LogReg.predict(X_test)
    print("train: " + str(f1_score(Y_train[train_index], Y_pred_train)))
    print("test: " + str(f1_score(Y_train[test_index], Y_pred)))
    i += 1

Y_test = (np.sum(predictions, axis=1) > 2.5).astype(int)

# submission = pd.DataFrame(Y_test)
# submission.to_csv(
#     path_or_buf=path_to_submissions+"-".join(my_features_string)+"LogReg.csv",
#     index=True,
#     index_label="id",
#     header=["category"]
# )
