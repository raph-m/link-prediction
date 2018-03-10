import pandas as pd

path_to_data = "data/"
training = pd.read_csv(path_to_data + "training_features.txt")
training.set_index("my_index", inplace=True)

testing = pd.read_csv(path_to_data + "testing_features.txt")
testing.set_index("my_index", inplace=True)

# create new csv file with only features so we can download faster
for col in training.columns:
    if col != ["katz", "katz_2"]:
        del training[col]

for col in testing.columns:
    if col != ["katz", "katz_2"]:
        del testing[col]

print("training head: ")
print(training.head())


print("testing head: ")
print(testing.head())

training.to_csv("training_katz.txt")
testing.to_csv("testing_katz.txt")


# merge new column in existing dataset:
# training_katz = pd.read_csv(path_to_data + "training_katz.txt")
# training_katz.set_index("my_index", inplace=True)
# testing_katz = pd.read_csv(path_to_data + "testing_katz.txt")
# testing_katz.set_index("my_index", inplace=True)
#
# training = pd.read_csv(path_to_data + "training_features.txt")
# training.set_index("my_index", inplace=True)
# testing = pd.read_csv(path_to_data + "testing_features.txt")
# testing.set_index("my_index", inplace=True)
#
# new_training = pd.merge(left=training, right=training_katz, left_index=True, right_index=True)
# new_testing = pd.merge(left=testing, right=testing_katz, left_index=True, right_index=True)
#
# new_training.to_csv("training_features.txt")
# new_testing.to_csv("training_features.txt")

