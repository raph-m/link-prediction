from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# path
path_to_data = "../data/"

# load data
training = pd.read_csv(path_to_data + "training_features.txt")
training.set_index("my_index", inplace=True)
testing = pd.read_csv(path_to_data + "testing_features.txt")
testing.set_index("my_index", inplace=True)

# separating features and labels
training_val = training.values
testing_val = testing.values
X_train, Y_train = training_val[:, 4:], training_val[:, 3]
X_test, Y_test = testing_val[:, 4:], testing_val[:, 3]

# # random forest
# RF = RandomForestClassifier()
# RF.fit(X_train, Y_train)
# Y_pred = RF.predict(X_test)


# if we can do KFold
RF = RandomForestClassifier()
kf = KFold(5)
for train_index, test_index in kf.split(X_train, Y_train):
    RF.fit(X_train[train_index], Y_train[train_index])
    Y_pred = RF.predict(X_train[test_index])
    print(accuracy_score(Y_train[test_index], Y_pred))
