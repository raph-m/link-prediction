import pandas as pd
from tqdm import tqdm

from tools import compare_journals, lit_eval_nan_proof

# progress bar for pandas
tqdm.pandas(tqdm())

# path
path_to_data = "../data/"

# loading data
converter_dict = {'authors': lit_eval_nan_proof, 'journal': lit_eval_nan_proof,
                  'title': lit_eval_nan_proof, 'abstract': lit_eval_nan_proof}
nodes = pd.read_csv(path_to_data + "nodes_preprocessed.csv",
                    converters=converter_dict)
nodes.set_index("id", inplace=True)
training = pd.read_csv(path_to_data + "training_features.txt")
training.set_index("my_index", inplace=True)
testing = pd.read_csv(path_to_data + "testing_features.txt")
testing.set_index("my_index", inplace=True)

# placeholder for second batch of features
journal_similarity = []
overlapping_words_abstract = []

# IDs for training set
id1 = training['id1'].values
id2 = training['id2'].values

# computing features for training set
for i in tqdm(range(len(id1))):
    journal1 = nodes.at[id1[i], 'journal']
    journal2 = nodes.at[id2[i], 'journal']
    abstract1 = nodes.at[id1[i], "abstract"]
    abstract2 = nodes.at[id2[i], "abstract"]
    if isinstance(journal1, float) or isinstance(journal2, float):
        journal_similarity.append(0)
    else:
        journal_similarity.append(compare_journals(journal1, journal2))
    overlapping_words_abstract.append(set(abstract1).intersection(set(abstract2)))

# adding feature to dataframe
training["journal_similarity"] = journal_similarity
training["overlapping_words_abstract"] = overlapping_words_abstract

# repeat process for test set
journal_similarity_test = []
overlapping_words_abstract_test = []
id1 = testing['id1'].values
id2 = testing['id2'].values
for i in tqdm(range(len(id1))):
    journal1 = nodes.at[id1[i], 'journal']
    journal2 = nodes.at[id2[i], 'journal']
    abstract1 = nodes.at[id1[i], "abstract"]
    abstract2 = nodes.at[id2[i], "abstract"]
    if isinstance(journal1, float) or isinstance(journal2, float):
        journal_similarity_test.append(0)
    else:
        journal_similarity_test.append(compare_journals(journal1, journal2))
    overlapping_words_abstract_test.append(set(abstract1).intersection(set(abstract2)))
testing["journal_similarity"] = journal_similarity_test
testing["overlapping_words_abstract"] = overlapping_words_abstract_test

# save data sets
training.to_csv(path_to_data + "training_features.txt")
testing.to_csv(path_to_data + "testing_features.txt")
