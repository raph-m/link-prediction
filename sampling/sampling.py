import pandas as pd

path_to_data = "~/Documents/polytechnique/3A/nlp/link-prediction/data/"

divide_by = 100
sample_size_string = str(divide_by)

nodes_header = ["id", "year", "title", "authors", "journal", "abstract"]
nodes = pd.read_csv(path_to_data+"node_information.csv", names=nodes_header)

names = ["id1", "id2", "target"]
training = pd.read_csv(path_to_data+"training_set.txt", names=names, delimiter=" ")

sample_1 = training.sample(frac=1.0/divide_by, replace=False)
sample_2 = sample_1.copy()
sample_2.columns = ["id2", "id1", "target"]

names = ["id1", "id2"]
testing = pd.read_csv(path_to_data+"testing_set.txt", names=names, delimiter=" ")

sample_1_testing = testing.sample(frac=1.0/divide_by, replace=False)

sample_2_testing = sample_1_testing.copy()
sample_2_testing.columns = ["id2", "id1"]

all_ids = pd.concat([sample_1, sample_2, sample_1_testing, sample_2_testing])
del all_ids["target"]
del all_ids["id2"]
all_ids.columns = ["id"]

all_ids_2 = all_ids.groupby(by="id").first().reset_index()

merged = all_ids_2.merge(right=nodes, how="inner")


merged.to_csv(path_to_data+"node_information"+sample_size_string+".csv", header=False)
sample_1.to_csv(path_to_data+"training_set"+sample_size_string+".txt", header=False, sep=" ")
sample_1_testing.to_csv(path_to_data+"testing_set"+sample_size_string+".txt", header=False, sep=" ")
