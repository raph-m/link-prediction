import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from models.tools import f1_score, f1_score_lgbm

# path
path_to_data = "data/"
path_to_submissions = "submissions/"
path_to_stacking = "stacking/"
path_to_plots = "models/plots/"

# tuned hyperparameters
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    # 'metric': {},
    'num_leaves': 200,
    'learning_rate': 0.1,
    'feature_fraction': 0.4,
    'bagging_fraction': 0.6,
    'bagging_freq': 5,
    'verbose': 0,
    "min_data_in_leaf": 6,
    "max_depth": 150
}
# load data
training = pd.read_csv(path_to_data + "training_features.txt")
testing = pd.read_csv(path_to_data + "testing_features.txt")
del training["my_index"]
del testing["my_index"]

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
    "shortest_path",
    "popularity",
    "paths_of_length_one"
    "katz"
    "katz_2"
]

my_features_index = []
my_features_dic = {}
my_features_acronym = ["_".join(list(map(lambda x: x[0], string.split('_')))) for string in my_features_string]
print(my_features_acronym)
target = 0
for i in range(len(training.columns)):
    if training.columns[i] == "target":
        target = i
    elif training.columns[i] in my_features_string:
        my_features_index.append(i)
        my_features_dic.update({len(my_features_index): training.columns[i]})

# separating features and labels
training_val = training.values
testing_val = testing.values
X_train, Y_train = training_val[:, my_features_index].astype(float), training_val[:, target].astype(int)
X_test = testing_val[:, my_features_index]

now = datetime.datetime.now()
print("date: " + str(now))
print("features: " + str(my_features_string))
print("model: Random Forest")
print("parameters:")
print(params)
print("cross validation:")

k = 5
kf = KFold(k)
predictions = np.zeros((X_test.shape[0], k))
predictions_train = np.zeros(X_train.shape[0])
i = 0

results = []
print('Start training...')
for train_index, test_index in kf.split(X_train):
    lgb_train = lgb.Dataset(X_train[train_index], Y_train[train_index])
    lgb_eval = lgb.Dataset(X_train[test_index], Y_train[test_index], reference=lgb_train)
    gbm = lgb.train(params,
                    train_set=lgb_train,
                    num_boost_round=100,
                    valid_sets=lgb_eval,
                    verbose_eval=40,
                    feval=f1_score_lgbm
                    )
    res = gbm.predict(X_test)
    Y_pred = gbm.predict(X_train[test_index])
    Y_pred_train = gbm.predict(X_train[train_index])
    predictions[:, i] = res
    predictions_train[test_index] = Y_pred
    print("train: " + str(f1_score(Y_train[train_index], Y_pred_train.round())))
    print("test: " + str(f1_score(Y_train[test_index], Y_pred.round())))
    i += 1

# save submission file
Y_test = (np.sum(predictions, axis=1) > 2.5).astype(int)
submission = pd.DataFrame(Y_test)
submission.to_csv(
    path_or_buf=path_to_submissions + "-".join(my_features_acronym) + "lgbm" + ".csv",
    index=True,
    index_label="id",
    header=["category"]
)

# save probabilities for stacking
stacking_logits_test = np.sum(predictions, axis=1)
stacking_test = pd.DataFrame(stacking_logits_test)
stacking_test.to_csv(
    path_or_buf=path_to_stacking + "lgbm_test" + ".csv",
    index=True,
    index_label="id",
    header=["category"]
)

stacking_train = pd.DataFrame(predictions_train)
stacking_train.to_csv(
    path_or_buf=path_to_stacking + "lgbm_train" + ".csv",
    index=True,
    index_label="id",
    header=["category"]
)