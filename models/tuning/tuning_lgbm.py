import warnings

from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

from models.tools import load_data
from models.tuning.tools import plot_grid

# deactivate deprecation warnings
warnings.simplefilter("ignore", DeprecationWarning)

n_jobs = 2

# path
path_to_data = "data/"
path_to_submissions = "submissions/"
path_to_stacking = "stacking/"
path_to_plots = "plots/"

# used features

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
    # "shortest_path",
    "popularity",
    "common_successors",
    "common_predecessors",
    # "paths_of_length_one",
    "authors_citation"
    "coauthor_score"
    # # "katz"
    # # "katz_2"
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
    # 'metric': {},
    'num_leaves': [150, 200, 250],
    "min_data_in_leaf": [2, 4, 6],
    "max_depth": [150, 200, 250]
}

# tuning
gbm = LGBMClassifier(
    boosting_type='gbdt',
    objective='binary',
    # 'metric': {},
    learning_rate=0.1,
    feature_fraction=0.4,
    bagging_fraction=0.6,
    bagging_freq=5,
    silent=True)
metrics = ["f1", "precision", "recall", "accuracy", "roc_auc"]
grid_lgbm = GridSearchCV(gbm,
                         param_grid=tuned_parameters,
                         scoring=metrics,
                         refit='f1',
                         cv=5,
                         n_jobs=n_jobs
                         )
grid_lgbm.fit(X_train, Y_train, verbose=-1)
print("GridSearch best parameters", grid_lgbm.best_params_)

# plot grid search results
best_params = grid_lgbm.best_params_
results = grid_lgbm.cv_results_
index = grid_lgbm.best_index_
plot_grid(metrics=results,
          params=tuned_parameters,
          index=index,
          param_names=list(tuned_parameters),
          name="grid_lgbm")
