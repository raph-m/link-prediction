import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from models.tools import f1_score

# path
path_to_data = "data/"
path_to_submissions = "submissions/"
path_to_stacking = "stacking"
path_to_plots = "plots"

# tuned hyper-parameters

parameters = {
    "n_estimators": 100,
    "criterion": "entropy",  # default = gini
    "max_depth": 20,
    "min_samples_leaf": 10,
    "bootstrap": True,
    "n_jobs": -1
}

# load data
training = pd.read_csv(path_to_data + "training_features.txt")
del training["my_index"]

# replace inf in shortest_path with -1
training['shortest_path'] = training['shortest_path'].replace([float('inf')], [-1])

my_features_string = [
    "date_diff",
    "overlap_title",
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
    "common_neighbors",
    "shortest_path",
    "popularity",
    "common_successors",
    "common_predecessors",
    "paths_of_length_one",
    "authors_citation",
    "normalized_authors_citation",
    "best_authors_citation",
    "coauthor_score",
    "normalized_coauthor_score",
    "best_coauthor_score",
    "authors_in_neighbors",
    "normalized_authors_in_neighbors",
    "best_authors_in_neighbors"
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

already_computed_names = []
already_computed = []

for i in range(len(training.columns)):
    if training.columns[i] in my_features_string:
        my_features_dic.update({i: training.columns[i]})
        my_features_index.append(i)
    if training.columns[i] in already_computed_names:
        already_computed.append(i)

features_to_keep = []
for u in range(len(my_features_index)):

    try:
        features_to_keep.append(already_computed[u])
        print("added already computed feature " + str(my_features_dic[already_computed[u]]))
    except:

        features_to_keep_names = [my_features_dic[i] for i in features_to_keep]
        print("new round !")
        print("u = " + str(u) + ", current features are: " + str(features_to_keep_names))
        best_test_score = 0.0
        best_train_score = 0.0
        best_index = 0
        for i, f in my_features_dic.items():
            if i not in features_to_keep:
                # separating features and labels
                print("testing additional feature: " + f)
                current_features = features_to_keep + [i]

                X_train = training.values[:, current_features]

                RF = RandomForestClassifier(
                    n_estimators=parameters["n_estimators"],
                    criterion=parameters["criterion"],
                    max_depth=parameters["max_depth"],
                    min_samples_leaf=parameters["min_samples_leaf"],
                    bootstrap=parameters["bootstrap"],
                    n_jobs=parameters["n_jobs"]
                )
                k = 2
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
                    best_test_score = test_score

                print("train score: " + str(train_score))
                print("test score: " + str(test_score))
                print("")

        print("for this round, the best feature was " + my_features_dic[best_index])
        features_to_keep.append(best_index)
        print("the scores obtained were: ")
        print("train score: " + str(best_train_score))
        print("test score: " + str(best_test_score))
        print("\n\n\n\n")

# # print feature importances
# for i in range(len(RF.feature_importances_)):
#     print(str(my_features_dic[i]) + ": " + str(RF.feature_importances_[i]))
