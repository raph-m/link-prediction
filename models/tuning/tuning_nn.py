import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from models.tools import load_data

# path
path_to_data = "data/"
path_to_submissions = "submissions/"
path_to_stacking = "stacking/"
path_to_plots = "models/plots/"

# load data
my_features_string = [
    "date_diff",
    "overlap_title",
    "common_author",
    "score_1_2",
    "score_2_1",
    "cosine_distance",
    "journal_similarity",
    # "overlapping_words_abstract",
    "jaccard",
    "adar",
    "preferential_attachment",
    "resource_allocation_index",
    "out_neighbors",
    "in_neighbors",
    "common_neighbors",
    # "shortest_path",
    "popularity",
    # "paths_of_length_one"
    # "katz"
    # "katz_2"
]

(X_train,
 X_test,
 Y_train,
 my_features_index,
 my_features_dic) = load_data(my_features_string)

# check for nans
data = pd.DataFrame(X_test)
print(data.info())
print(data.isna().sum(axis=0))
print(data.min(axis=0))
print(data.max(axis=0))
print(my_features_index)
print(my_features_dic)

# normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to create model, required for KerasClassifier
nb_input = len(my_features_string)


def create_model(neurons=1, dropout_rate=0.1, activation='relu'):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=nb_input, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, input_dim=nb_input, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# fixed parameters
epochs = 20
batch_size = 128

# create model
model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=1)

# define the grid search parameters
neurons = [15, 30, 45, 60, 75]
dropout_rate = [0.0, 0.1, 0.2, 0.3]
activation = ['relu', 'tanh', 'sigmoid']
param_grid = dict(neurons=neurons, dropout_rate=dropout_rate, activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, Y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
