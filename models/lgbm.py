import datetime
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import lightgbm as lgb
from tools import f1_score, binary_error, f1_score_lgbm

# path
path_to_data = "../../data/"
path_to_submissions = "../../submissions/"

# parameters
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
    "min_data_in_leaf": 2,
    "max_depth": 200
}
# load data
training = pd.read_csv(path_to_data + "training_features.txt")
testing = pd.read_csv(path_to_data + "testing_features.txt")
del training["my_index"]
del testing["my_index"]

my_features_string = [
    "overlap_title",
    "date_diff",
    "common_author",
    "journal_similarity",
    "overlapping_words_abstract",
    "cosine_distance",
    "shortest_path",
    # "author_min_shortest_path",
    # "author_max_shortest_path",
    # "author_sum_shortest_path",
    # "author_mean_shortest_path",
    # "author_out_degree_sum_source",
    "author_out_degree_mean_source",
    # "author_in_degree_sum_target",
    "author_in_degree_mean_target",
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
print("date: "+str(now))
print("features: "+str(my_features_string))
print("model: Random Forest")
print("parameters:")
print(params)
print("cross validation:")


k = 5
kf = KFold(k)
predictions = np.zeros((X_test.shape[0], k))
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
    predictions[:, i] = res

    Y_pred = gbm.predict(X_train[test_index]).round()
    Y_pred_train = gbm.predict(X_train[train_index]).round()
    predictions[:, i] = gbm.predict(X_test)
    print("train: "+str(f1_score(Y_train[train_index], Y_pred_train)))
    print("test: "+str(f1_score(Y_train[test_index], Y_pred)))

    i += 1

Y_test = (np.sum(predictions, axis=1) > 2.5).astype(int)

submission = pd.DataFrame(Y_test)
submission.to_csv(
    path_or_buf=path_to_submissions+"-".join(my_features_acronym)+".csv",
    index=True,
    index_label="id",
    header=["category"]
)
print("kaggle score: ")

