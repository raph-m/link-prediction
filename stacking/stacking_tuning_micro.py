import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier

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
    else :
        X_test[model] = 0.2 * pd.read_csv(path_to_stacking + model + "_test.csv")['category'].values
print(X_train.head(), X_test.head())
X_train = X_train.values
X_test = X_test.values

# GridSearchCV

# param grid

tuned_parameters = {
    "n_estimators": [100],
    "max_depth": [3, 7, 10],
    "min_samples_leaf": [6],
    "criterion" : ["entropy"]
}

# tuning
rf = RandomForestClassifier(
    bootstrap=True,
    n_jobs=-1
)

metrics = ["f1"]
grid_RF = GridSearchCV(rf,
                       param_grid=tuned_parameters,
                       scoring=metrics,
                       refit='f1',
                       cv=4,
                       n_jobs=-1,
                       verbose=10
                       )
grid_RF.fit(X_train, Y_train)
print("GridSearch best parameters", grid_RF.best_params_)