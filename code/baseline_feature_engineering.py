import ast
import pandas as pd
from tqdm import tqdm

from tools import lit_eval_nan_proof

# progress bar for pandas
tqdm.pandas(tqdm())

# path
path_to_data = "../data/"

# loading preprocessed data
converter_dict = {'authors': lit_eval_nan_proof, 'journal': lit_eval_nan_proof,
                  'title': lit_eval_nan_proof, 'abstract': lit_eval_nan_proof}
nodes = pd.read_csv(path_to_data + "nodes_preprocessed.csv", converters=converter_dict)
nodes.set_index("id", inplace=True)
training = pd.read_csv(path_to_data + "training_new_index.txt")
training.set_index("my_index", inplace=True)
testing = pd.read_csv(path_to_data + "testing_new_index.txt")
testing.set_index("my_index", inplace=True)

# adding baseline features in training dataframe

# features placeholders
overlap_title = []
date_diff = []
common_author = []

# IDs for training set
id1 = training['id1'].values
id2 = training['id2'].values

# computing features for training set
for i in tqdm(range(len(id1))):
    title1 = nodes.at[id1[i], 'title']
    title2 = nodes.at[id2[i], 'title']
    date1 = nodes.at[id1[i], 'year']
    date2 = nodes.at[id2[i], 'year']
    author1 = nodes.at[id1[i], 'authors']
    author2 = nodes.at[id2[i], 'authors']
    overlap_title.append(len(set(title1).intersection(set(title2))))
    date_diff.append(int(date1) - int(date2))
    if isinstance(author1, float) or isinstance(author2, float):
        common_author.append(0)
    else:
        common_author.append(len(set(author1).intersection(set(author2))))

# adding feature to dataframe
training["overlap_title"] = overlap_title
training["date_diff"] = date_diff
training["common_author"] = common_author


# repeat process for test set
overlap_title_test = []
date_diff_test = []
common_author_test = []
id1 = testing['id1'].values
id2 = testing['id2'].values
for i in tqdm(range(len(id1))):
    title1 = nodes.at[id1[i], 'title']
    title2 = nodes.at[id2[i], 'title']
    date1 = nodes.at[id1[i], 'year']
    date2 = nodes.at[id2[i], 'year']
    author1 = nodes.at[id1[i], 'authors']
    author2 = nodes.at[id2[i], 'authors']
    overlap_title_test.append(len(set(title1).intersection(set(title2))))
    date_diff_test.append(int(date1) - int(date2))
    if isinstance(author1, float) or isinstance(author2, float):
        common_author_test.append(0)
    else:
        common_author_test.append(len(set(author1).intersection(set(author2))))
testing["overlap_title"] = overlap_title_test
testing["date_diff"] = date_diff_test
testing["common_author"] = common_author_test

# save data sets
training.to_csv(path_to_data + "training_features.txt")
testing.to_csv(path_to_data + "testing_features.txt")
