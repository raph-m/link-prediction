import warnings
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV


# deactivate deprecation warnings
warnings.simplefilter("ignore", DeprecationWarning)

# path
path_to_data = "../../../data/"
path_to_submissions = "../../../submissions/"
path_to_plots = "../../plots"

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
                         n_jobs=2)
grid_lgbm.fit(X_train, Y_train, verbose=-1)
print("GridSearch best parameters", grid_lgbm.best_params_)
