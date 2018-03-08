from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd

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
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 6, 9],
    "min_samples_leaf": [10, 20]
}

# tuning
rf = RandomForestClassifier(criterion="entropy", bootstrap=True)  # default = gini
metrics = ["f1", "precision", "recall", "accuracy", "roc_auc"]
grid_RF = GridSearchCV(rf,
                       param_grid=tuned_parameters,
                       scoring=metrics,
                       refit='f1',
                       cv=5,
                       n_jobs=2,
                       verbose=10)
grid_RF.fit(X_train, Y_train)
print("GridSearch best parameters", grid_RF.best_params_)
