import time

import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample

# path
path_to_data = "data/"
path_to_plots = "models/tuning/plots/"

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

# subsampling
X_train_sub, Y_train_sub = resample(X_train, Y_train, n_samples=500, random_state=42)
print(X_train_sub.shape, Y_train_sub.shape)
# pipeline architecture
pipe = Pipeline([
    ('reduce_dim', PCA()),
    ('classif', SVC(gamma=0.01))
])
# parameter values
nb_features = [2, 4]
Cs = [0.001, 0.01, 0.1]
kernels = ['linear', 'rbf']

# parameter grid
param_grid = [
    {
        'reduce_dim': [PCA()],
        'reduce_dim__n_components': nb_features,
        'classif__C': Cs,
        'classif__kernel':kernels
    },
    # {
    #     'reduce_dim': [SelectKBest(chi2)],
    #     'reduce_dim__k': nb_features,
    #     'classif__C': Cs,
    #     'classif__kernel':kernels
    # }
    # ,
    {
        'reduce_dim': [SelectKBest(mutual_info_classif)],
        'reduce_dim__k': nb_features,
        'classif__C': Cs,
        'classif__kernel':kernels
    }
]

# cross validation grid search instance
grid = GridSearchCV(pipe, cv=4, n_jobs=2, param_grid=param_grid, verbose=10)

# fit grid
grid.fit(X_train_sub, Y_train_sub)

# print best params
print(grid.best_params_)
