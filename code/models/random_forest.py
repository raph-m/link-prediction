import datetime
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# path
path_to_data = "../data/"
path_to_submissions = "../submissions/"

# parameters


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
    # "cosine_distance",
    "shortest_path"
]
my_features_index = []

target = 0
for i in range(len(training.columns)):
    if training.columns[i] == "target":
        target = i
    elif training.columns[i] in my_features_string:
        my_features_index.append(i)

# separating features and labels
training_val = training.values
testing_val = testing.values
X_train, Y_train = training_val[:, my_features_index].astype(float), training_val[:, target].astype(int)
X_test = testing_val[:, my_features_index]

now = datetime.datetime.now()
print("date: "+str(now))
print("features: "+str(my_features_string))
print("model: Random Forest")
print("parameters: default")
print("cross validation:")

RF = RandomForestClassifier(n_estimators=30)
k = 5
kf = KFold(k)
predictions = np.zeros((X_test.shape[0], k))
i = 0

for train_index, test_index in kf.split(X_train, Y_train):
    RF.fit(X_train[train_index], Y_train[train_index])
    Y_pred = RF.predict(X_train[test_index])
    predictions[:, i] = RF.predict(X_test)
    print(accuracy_score(Y_train[test_index], Y_pred))
    i += 1

Y_test = (np.sum(predictions, axis=1) > 2.1).astype(int)

submission = pd.DataFrame(Y_test)
submission.to_csv(
    path_or_buf=path_to_submissions+"-".join(my_features_string)+".csv",
    index=True,
    index_label="id",
    header=["category"]
)
print("kaggle score: ")
