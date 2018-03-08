from sklearn import svm
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from tools import f1_score

# path
path_to_data = "../../data/"
path_to_submissions = "../../submissions/"
path_to_stacking = "../../stacking"
path_to_plots = "../../plots"

# tuned hyperparameters
parameters = {
    'C': 0.1,
    'gamma': 0.01,
    'kernel': "rbf"
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

now = datetime.datetime.now()
print("date: " + str(now))
print("features: " + str(my_features_string))
print("model: Random Forest")
print("parameters:")
print(parameters)
print("cross validation:")

svm_classifier = svm.SVC(C=0.1, gamma=0.01, kernel="rbf")
k = 5
kf = KFold(k)
predictions = np.zeros((X_test.shape[0], k))
i = 0

for train_index, test_index in kf.split(X_train, Y_train):
    svm_classifier.fit(X_train[train_index], Y_train[train_index])
    Y_pred = svm_classifier.predict(X_train[test_index])
    Y_pred_train = svm_classifier.predict(X_train[train_index])
    predictions[:, i] = svm_classifier.predict(X_test)
    print("train: " + str(f1_score(Y_train[train_index], Y_pred_train)))
    print("test: " + str(f1_score(Y_train[test_index], Y_pred)))
    i += 1

# save submission file
Y_test = (np.sum(predictions, axis=1) > 2.5).astype(int)
submission = pd.DataFrame(Y_test)
submission.to_csv(
    path_or_buf=path_to_submissions + "-".join(my_features_string) + "SVM.csv",
    index=True,
    index_label="id",
    header=["category"]
)

# save probabilities for stacking
stacking_logits = np.sum(predictions, axis=1)
submission = pd.DataFrame(stacking_logits)
submission.to_csv(
    path_or_buf=path_to_stacking + "-".join(my_features_acronym) + "lgbm" + ".csv",
    index=True,
    index_label="id",
    header=["category"]
)