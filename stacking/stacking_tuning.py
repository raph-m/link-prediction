import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

# path
path_to_data = "data/"
path_to_submissions = "submissions/"
path_to_stacking = "stacking/"

# get labels
names = ["id1", "id2", "target"]
Y_train = pd.read_csv(path_to_data + "training_set.txt", names=names, delimiter=" ")
Y_test = pd.read_csv(path_to_data + "testing_set.txt", names=names, delimiter=" ")
Y_train = Y_train['target'].values
Y_test = Y_test['target'].values

# group model predictions as features
model_strings = ['lgbm', 'rf', 'svmlinear', 'nn', 'nn_deep']
X_train = pd.DataFrame(columns=model_strings)
X_test = pd.DataFrame(columns=model_strings)
for model in model_strings:
    X_train[model] = pd.read_csv(path_to_stacking + model + "_train.csv")['category']
    # take the mean of the test set probs of each cv fold
    if model == 'svm_linear':
        X_test[model] = 0.5 * pd.read_csv(path_to_stacking + model + "_test.csv")['category'].values
    else:
        X_test[model] = 0.2 * pd.read_csv(path_to_stacking + model + "_test.csv")['category'].values
print(X_train.head(), X_test.head())
X_train = X_train.values
X_test = X_test.values

# fit a grid searched logistic regression on top of the base models
# parameters grid
param_grid = {
    "C": [1, 0.1, 0.01, 0.001],
    "penalty": ["l2", "l1"]
}

# pipeline architecture
pipe = Pipeline([
    ('reduce_dim', SelectKBest(chi2)),
    ('classif', LogisticRegression())
])
# parameter values
nb_features = [2, 3, 4, 5]
C = [0.001, 0.01, 0.1]
kernels = ['linear', 'rbf']
n_estimators = [100, 200]
max_depth = [10, 20]
min_samples_leaf = [20]
penalty = ["l2", "l1"]

# parameter grid
param_grid = [
    {
        'reduce_dim__k': [5],
        'classif': [RandomForestClassifier(bootstrap=True, n_jobs=-1)],
        'classif__n_estimators': n_estimators,
        'classif__max_depth': max_depth,
        'classif__min_samples_leaf': min_samples_leaf,

    },
    {
        'reduce_dim__k': nb_features,
        'classif': [LogisticRegression(n_jobs=-1)],
        'classif__C': C,
        'classif__penalty': penalty
    }
]

# cross validation grid search instance
grid = GridSearchCV(pipe, cv=4, n_jobs=-1, param_grid=param_grid, verbose=10)

# fit grid
grid.fit(X_train, Y_train)

# print best params
print(grid.best_params_)

# get params
print(grid.best_params_)
parameters = grid.best_params_

# model instance for prediction
model = grid.best_estimator_

# cross validated predictions
k = 5
kf = StratifiedKFold(k)
predictions = np.zeros((X_test.shape[0], k))
i = 0

for train_index, test_index in kf.split(X_train, Y_train):
    model.fit(X_train[train_index], Y_train[train_index])
    Y_pred = model.predict(X_train[test_index])
    Y_pred_train = model.predict(X_train[train_index])
    predictions[:, i] = model.predict(X_test)
    print("train: " + str(f1_score(Y_train[train_index], Y_pred_train)))
    print("test: " + str(f1_score(Y_train[test_index], Y_pred)))
    i += 1

Y_test = (np.sum(predictions, axis=1) > 2.5).astype(int)
submission = pd.DataFrame(Y_test)
submission.to_csv(
    path_or_buf=path_to_submissions + "stack_sub2.csv",
    index=True,
    index_label="id",
    header=["category"]
)
