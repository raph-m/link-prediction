import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim import corpora, models

from tools import lit_eval_nan_proof

# progress bar for pandas
tqdm.pandas(tqdm())

# path
path_to_data = "../../data/"

# loading data
converter_dict = {'authors': lit_eval_nan_proof, 'journal': lit_eval_nan_proof,
                  'title': lit_eval_nan_proof, 'abstract': lit_eval_nan_proof}
nodes = pd.read_csv(path_to_data + "nodes_preprocessed.csv", converters=converter_dict)
nodes.set_index("id", inplace=True)
training = pd.read_csv(path_to_data + "training_features.txt")
training.set_index("my_index", inplace=True)
testing = pd.read_csv(path_to_data + "testing_features.txt")
testing.set_index("my_index", inplace=True)

# create dictionary for tfidf
abstracts = nodes['abstract'].values
dictionary = corpora.Dictionary(abstracts)

# instatiate tfidf model
tfidf = models.TfidfModel(dictionary=dictionary)


# handy functions to compute cosine distance
def get_tf_idf_encoding(index):
    abstract = nodes.at[index, "abstract"]
    abstract = abstract.split(" ")
    abstract = dictionary.doc2bow(abstract)
    ans = tfidf[[abstract]]
    return ans[0]


def my_norm(tfidf_abstract):
    ans = 0.0
    for (k, v) in tfidf_abstract:
        ans += v ** 2
    return np.sqrt(ans)


def cosine_distance(id1, id2):
    tfidf_abstract1 = get_tf_idf_encoding(id1)
    tfidf_abstract2 = get_tf_idf_encoding(id2)
    denom = my_norm(tfidf_abstract1) * my_norm(tfidf_abstract2)
    f1 = dict(tfidf_abstract1)
    f2 = dict(tfidf_abstract2)
    ans = 0.0
    for k, v in f1.items():
        if k in f2.keys():
            ans += v * f2[k]
    return ans / denom

# placeholder for feature
cosine_dist = []

# IDs for training set
id1 = training['id1'].values
id2 = training['id2'].values

# computing features for training set
for i in tqdm(range(len(id1))):
    cosine_dist.append(cosine_distance(id1[i], id2[i]))

# add feature to dataframe
training["cosine_distance"] = cosine_dist

# repeat process for test set
cosine_dist_test = []
id1 = testing['id1'].values
id2 = testing['id2'].values
for i in tqdm(range(len(id1))):
    cosine_dist_test.append(cosine_distance(id1[i], id2[i]))
testing["cosine_distance"] = cosine_dist_test

# save dataframe
training.to_csv(path_to_data + "training_features.txt")
testing.to_csv(path_to_data + "testing_features.txt")
