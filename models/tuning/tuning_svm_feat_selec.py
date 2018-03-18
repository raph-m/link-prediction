from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils import resample

from models.tools import load_data

# path
path_to_data = "data/"
path_to_plots = "models/tuning/plots/"

# used features

my_features_string = [
    "date_diff",
    # "overlap_title",
    "common_author",
    # # "score_1_2",
    # # "score_2_1",
    "cosine_distance",
    # "journal_similarity",
    # # "overlapping_words_abstract",
    # # "jaccard",
    # # "adar",
    "preferential_attachment",
    # # "resource_allocation_index",
    # "out_neighbors",
    "in_neighbors",
    "common_neighbors",
    # "shortest_path",
    # "popularity",
    # "common_successors",
    # "common_predecessors",
    # "paths_of_length_one",
    "authors_citation"
    # "coauthor_score"
    # # "katz"
    # # "katz_2"
]

# load data

(X_train,
 X_test,
 Y_train,
 my_features_index,
 my_features_dic) = load_data(my_features_string)

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
        'classif__kernel': kernels
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
        'classif__kernel': kernels
    }
]

# cross validation grid search instance
grid = GridSearchCV(pipe, cv=4, n_jobs=2, param_grid=param_grid, verbose=10)

# fit grid
grid.fit(X_train_sub, Y_train_sub)

# print best params
print(grid.best_params_)
