from sklearn import svm
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from models.tools import f1_score

# path
path_to_data = "data/"
path_to_submissions = "submissions/"
path_to_stacking = "stacking/"
path_to_plots = "models/plots"

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
    "score_2_1",
    "cosine_distance",
    "journal_similarity",
    # "overlapping_words_abstract",
    "jaccard",
    "adar",
    "preferential_attachment",
    "resource_allocation_index",
    "out_neighbors",
    "in_neighbors",
    "common_neighbors",
    # "shortest_path",
    "popularity",
    # "paths_of_length_one"
    "katz"
    "katz_2"
]

my_features_index = []
my_features_dic = {}
my_features_acronym = ["_".join(list(map(lambda x: x[0], string.split('_')))) for string in my_features_string]

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

now = datetime.datetime.now()
print("date: " + str(now))
print("features: " + str(my_features_string))
print("model: SVM")
print("parameters:")
print(parameters)
print("cross validation:")

svm_classifier = svm.SVC(C=parameters['C'],
                         gamma=parameters['gamma'],
                         kernel=parameters['kernel'] ,
                         verbose=1)
k = 2
kf = StratifiedKFold(k)
predictions = np.zeros((X_test.shape[0], k))
predictions_train = np.zeros(X_train.shape[0])
i = 0

for train_index, test_index in kf.split(X_train, Y_train):
    svm_classifier.fit(X_train[train_index], Y_train[train_index])
    Y_pred = svm_classifier.predict(X_train[test_index])
    Y_pred_train = svm_classifier.predict(X_train[train_index])
    predictions[:, i] = svm_classifier.predict(X_test)
    predictions_train[test_index] = Y_pred
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
stacking_logits_test = np.sum(predictions, axis=1)
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
