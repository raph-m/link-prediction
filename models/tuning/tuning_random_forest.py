from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

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
