from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import datetime
import pandas as pd
import numpy
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from models.tools import load_data
from models.tools import f1_score

# path
path_to_data = "data/"
path_to_submissions = "submissions/"
path_to_stacking = "stacking/"
path_to_plots = "plots/"

# tuned hyper-parameters

parameters = {
    "n_estimators": 150,
    "criterion": "entropy",  # default = gini
    "max_depth": 15,  # 9
    "min_samples_leaf": 4,  # 10
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

my_features_acronym = ["_".join(list(map(lambda x: x[0], string.split('_')))) for string in my_features_string]

(X_train,
 X_test,
 Y_train,
 my_features_index,
 my_features_dic) = load_data(my_features_string)

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
    model.add(Dense(2 * neurons, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, input_dim=nb_input, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# parameters
epochs = 30
batch_size = 128

# tuned parameters
dropout_rate = 0.2
neurons = 75

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# create model
nn = KerasClassifier(build_fn=create_model,
                        epochs=epochs,
                        batch_size=batch_size,
                        dropout_rate=dropout_rate,
                        neurons=neurons,
                        verbose=1
                        )

now = datetime.datetime.now()
print("date: " + str(now))
print("features: " + str(my_features_string))
print("model: Neural Network")
print("parameters:")
print(parameters)
print("cross validation:")

k = 5
kf = StratifiedKFold(k)
predictions = np.zeros((X_test.shape[0], k))
predictions_test = np.zeros((X_test.shape[0], k))
predictions_train = np.zeros(X_train.shape[0])
i = 0

test_score = 0.0
for train_index, test_index in kf.split(X_train, Y_train):
    nn.fit(X_train[train_index], Y_train[train_index])
    Y_pred = nn.predict(X_train[test_index])[:, 0]
    Y_pred_train = nn.predict(X_train[train_index])[:, 0]
    predictions[:, i] = nn.predict(X_test)[:, 0]
    predictions_test[:, i] = nn.predict_proba(X_test)[:, 1]
    predictions_train[test_index] = nn.predict_proba(X_train[test_index])[:, 1]
    # current_test_score = f1_score(Y_train[test_index], Y_pred)[:, 0]
    # test_score += current_test_score
    # print("train: " + str(f1_score(Y_train[train_index], Y_pred_train)))
    # print("test: " + str(current_test_score))
    i += 1
# print("CV test score: "+str(test_score/k))

# save submission file
Y_test = (np.sum(predictions, axis=1) > 2.5).astype(int)
submission = pd.DataFrame(Y_test)
submission.to_csv(
    path_or_buf=path_to_submissions + "-".join(my_features_acronym) + "nn_deep.csv",
    index=True,
    index_label="id",
    header=["category"]
)

# save probabilities for stacking
stacking_logits_test = np.sum(predictions_test, axis=1)
stacking_test = pd.DataFrame(stacking_logits_test)
stacking_test.to_csv(
    path_or_buf=path_to_stacking + "nn_deep_test" + ".csv",
    index=True,
    index_label="id",
    header=["category"]
)

stacking_train = pd.DataFrame(predictions_train)
stacking_train.to_csv(
    path_or_buf=path_to_stacking + "nn_deep_train" + ".csv",
    index=True,
    index_label="id",
    header=["category"]
)