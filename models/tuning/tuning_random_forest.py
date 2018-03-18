from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from models.tools import load_data

# path
path_to_data = "data/"
path_to_submissions = "submissions/"
path_to_stacking = "stacking/"
path_to_plots = "plots/"

# tuned hyper-parameters

parameters = {
    "criterion": "entropy",  # default = gini
    "bootstrap": True,
    "n_jobs": -1
}

# used features

my_features_string = [
    "date_diff",
    "overlap_title",
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
    "out_neighbors",
    "in_neighbors",
    "common_neighbors",
    "shortest_path",
    "popularity",
    "common_successors",
    "common_predecessors",
    "paths_of_length_one"
    # "katz"
    # "katz_2"
]

# load data

(X_train,
 X_test,
 Y_train,
 my_features_index,
 my_features_dic) = load_data(my_features_string)


# GridSearchCV

# param grid

tuned_parameters = {
    "n_estimators": [150],
    "max_depth": [3, 6, 9, 12, 15, 20],
    "min_samples_leaf": [3, 5, 10, 20]
}

# tuning
rf = RandomForestClassifier(
    criterion=parameters["criterion"],
    bootstrap=parameters["bootstrap"],
    n_jobs=parameters["n_jobs"]
)

metrics = ["f1", "precision", "recall", "accuracy", "roc_auc"]
grid_RF = GridSearchCV(rf,
                       param_grid=tuned_parameters,
                       scoring=metrics,
                       refit='f1',
                       cv=5,
                       n_jobs=-1,
                       verbose=10
                       )
grid_RF.fit(X_train, Y_train)
print("GridSearch best parameters", grid_RF.best_params_)
