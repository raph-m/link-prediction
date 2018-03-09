from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

from models.tools import f1_score

# path
path_to_data = "data/"
path_to_submissions = "submissions/"
path_to_stacking = "stacking"
path_to_plots = "plots"

# tuned hyper-parameters

parameters = {
    "n_estimators": 150,
    "criterion": "entropy",  # default = gini
    "max_depth": 9,
    "min_samples_leaf": 10,
    "bootstrap": True,
    "n_jobs": -1
}

# load data
training = pd.read_csv(path_to_data + "training_features.txt")
del training["my_index"]

# replace inf in shortest_path with -1

my_features_string = [
    "overlap_title",
    "date_diff",
    "common_author",
    "score_1_2",
    "score_2_1",
    "cosine_distance",
    "journal_similarity",
    "overlapping_words_abstract",
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

Y_train = training.values[:, target].astype(int)

del training["target"]

for i in range(len(training.columns)):
    if training.columns[i] in my_features_string:
        my_features_dic.update({len(my_features_index): training.columns[i]})
        my_features_index.append(i)

features_to_keep = []
for u in range(len(my_features_index)):
    features_to_keep_names = [my_features_dic[i] for i in features_to_keep]
    print("new round !")
    print("u = " + str(u) + ", current features are: " + str(features_to_keep_names))
    best_test_score = 0.0
    best_train_score = 0.0
    best_index = 0
    for i, f in my_features_dic.items():
        # separating features and labels
        print("testing additional feature: " + f)
        current_features = features_to_keep + [i]

        print(current_features)

        X_train = training.values[:, current_features].astype(float)

        RF = RandomForestClassifier(
            n_estimators=parameters["n_estimators"],
            criterion=parameters["criterion"],
            max_depth=parameters["max_depth"],
            min_samples_leaf=parameters["min_samples_leaf"],
            bootstrap=parameters["bootstrap"],
            n_jobs=parameters["n_jobs"]
        )
        k = 5
        kf = KFold(k)
        train_score = 0.0
        test_score = 0.0

        for train_index, test_index in kf.split(X_train, Y_train):
            RF.fit(X_train[train_index], Y_train[train_index])
            Y_pred = RF.predict(X_train[test_index])
            Y_pred_train = RF.predict(X_train[train_index])
            train_score += f1_score(Y_train[train_index], Y_pred_train)
            test_score += f1_score(Y_train[test_index], Y_pred)

        train_score /= k
        test_score /= k

        if test_score > best_test_score:
            best_index = i
            best_train_score = train_score

        print("train score: "+str(train_score))
        print("test score: " + str(test_score))
        print("")

    print("for this round, the best feature was " + my_features_dic[best_index])
    print("the scores obtained were: ")
    print("train score: " + str(best_train_score))
    print("test score: " + str(best_test_score))
    print("\n\n\n\n")

# # print feature importances
# for i in range(len(RF.feature_importances_)):
#     print(str(my_features_dic[i]) + ": " + str(RF.feature_importances_[i]))
