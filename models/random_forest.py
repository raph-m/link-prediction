import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from models.tools import f1_score, plot_importance, load_data

# path
path_to_data = "data/"
path_to_submissions = "submissions/"
path_to_stacking = "stacking/"
path_to_plots = "plots/"

# tuned hyper-parameters

parameters = {
    "n_estimators": 150,
    "criterion": "entropy",  # default = gini
    "max_depth": 20,  # 9
    "min_samples_leaf": 10,  # 10
    "bootstrap": True,
    "n_jobs": -1
}

# used features

my_features_string = [
    "date_diff",
    # "overlap_title",
    "common_author",
    # "score_1_2",
    # "score_2_1",
    "cosine_distance",
    # "journal_similarity",
    # "overlapping_words_abstract",
    # "jaccard",
    # "adar",
    "preferential_attachment",
    # "resource_allocation_index",
    # "out_neighbors",
    "in_neighbors",
    "common_neighbors",
    # "shortest_path",
    # "popularity",
    # "common_successors",
    # "common_predecessors",
    # "paths_of_length_one",
    "authors_citation",
    # "coauthor_score",
    # "katz",
    # "katz_2"
]

my_features_acronym = ["_".join(list(map(lambda x: x[0], string.split('_')))) for string in my_features_string]

# load data

(X_train,
 X_test,
 Y_train,
 my_features_index,
 my_features_dic) = load_data(my_features_string)

# print user info
now = datetime.datetime.now()
print("date: " + str(now))
print("features: " + str(my_features_string))
print("model: Random Forest")
print("parameters:")
print(parameters)
print("cross validation:")

# instantiate classifier
RF = RandomForestClassifier(
    n_estimators=parameters["n_estimators"],
    criterion=parameters["criterion"],
    max_depth=parameters["max_depth"],
    min_samples_leaf=parameters["min_samples_leaf"],
    bootstrap=parameters["bootstrap"],
    n_jobs=parameters["n_jobs"]
)

# instantiate Kfold and predictions placeholder
k = 2
kf = KFold(k)
predictions = np.zeros((X_test.shape[0], k))
predictions_test = np.zeros((X_test.shape[0], k))
predictions_train = np.zeros(X_train.shape[0])
i = 0

# for each fold store predictions on test set and print validation results
test_score = 0.0
for train_index, test_index in kf.split(X_train, Y_train):
    RF.fit(X_train[train_index], Y_train[train_index])
    Y_pred = RF.predict(X_train[test_index])
    Y_pred_train = RF.predict(X_train[train_index])
    predictions[:, i] = RF.predict(X_test)
    predictions_test[:, i] = RF.predict_proba(X_test)[:, 1]
    predictions_train[test_index] = RF.predict_proba(X_train[test_index])[:, 1]
    current_test_score = f1_score(Y_train[test_index], Y_pred)
    test_score += current_test_score
    print("train: " + str(f1_score(Y_train[train_index], Y_pred_train)))
    print("test: " + str(current_test_score))
    i += 1

print("CV test score: " + str(test_score / k))
# save submission file
Y_test = (np.sum(predictions, axis=1) > 2.5).astype(int)
submission = pd.DataFrame(Y_test)
submission.to_csv(
    path_or_buf=path_to_submissions + "-".join(my_features_acronym) + "RF.csv",
    index=True,
    index_label="id",
    header=["category"]
)

# save probabilities for stacking
stacking_logits_test = np.sum(predictions_test, axis=1)
stacking_test = pd.DataFrame(stacking_logits_test)
stacking_test.to_csv(
    path_or_buf=path_to_stacking + "rf_test_2" + ".csv",
    index=True,
    index_label="id",
    header=["category"]
)

stacking_train = pd.DataFrame(predictions_train)
stacking_train.to_csv(
    path_or_buf=path_to_stacking + "rf_train_2" + ".csv",
    index=True,
    index_label="id",
    header=["category"]
)

# plot feature importances
plot_importance(RF,
                features_dict=my_features_dic,
                features_index=my_features_index,
                name='rf_importance')
