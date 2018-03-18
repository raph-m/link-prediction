import math

import numpy as np
import pandas as pd
from gensim import corpora, models
from tqdm import tqdm

from feature_engineering.tools import lit_eval_nan_proof

# this script adds the features score_1_2, score_2_1 and cosine_distance to the features csv files.
# this script takes approximately 10 minutes to run

# progress bar for pandas
tqdm.pandas(tqdm())

# path
path_to_data = "data/"

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
average_len = np.mean(np.array([len(a) for a in abstracts]))
dictionary = corpora.Dictionary(abstracts)


def my_tf(p):
    return math.log(1.0 + p)


# instantiate tf-idf model
tfidf = models.TfidfModel(dictionary=dictionary, wlocal=my_tf)


# handy functions to compute cosine distance
def get_tf_idf_encoding(index):
    abstract = nodes.at[index, "abstract"]
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
    f1 = dict(tfidf_abstract1)
    f2 = dict(tfidf_abstract2)
    ans = 0.0
    for k, v in f1.items():
        if k in f2.keys():
            ans += v * f2[k]
    return ans


def get_score(id1, id2, avglen, k1=1.2, b=0.75):
    abstract_1 = nodes.at[id1, "abstract"]
    len_1 = len(abstract_1)
    abstract_1 = dictionary.doc2bow(abstract_1)
    tf_1 = dict([
        (termid, tfidf.wlocal(tf))
        for termid, tf in abstract_1 if tfidf.idfs.get(termid, 0.0) != 0.0
    ])
    idf_1 = dict([
        (termid, tfidf.idfs.get(termid))
        for termid, tf in abstract_1 if tfidf.idfs.get(termid, 0.0) != 0.0
    ])

    abstract_2 = nodes.at[id2, "abstract"]
    abstract_2 = dictionary.doc2bow(abstract_2)
    tf_2 = dict([
        (termid, tfidf.wlocal(tf))
        for termid, tf in abstract_2 if tfidf.idfs.get(termid, 0.0) != 0.0
    ])

    ans = 0.0
    for k, v in tf_1.items():
        if k in tf_2.keys():
            ans += idf_1[k] * (v * (k1 + 1)) / (v + k1 * (1 - b + b * len_1 / avglen))
    return ans


# placeholder for feature
score_1_2 = []
score_2_1 = []
cosine_dist = []

# IDs for training set
id1 = training['id1'].values
id2 = training['id2'].values

# computing features for training set
for i in tqdm(range(len(id1))):
    score_1_2.append(get_score(id1[i], id2[i], average_len))
    score_2_1.append(get_score(id2[i], id1[i], average_len))
    cosine_dist.append(cosine_distance(id1[i], id2[i]))

# add feature to data-frame
training["score_1_2"] = score_1_2
training["score_2_1"] = score_2_1
training["cosine_distance"] = cosine_dist

score_1_2 = []
score_2_1 = []
cosine_dist = []

# IDs for training set
id1 = testing['id1'].values
id2 = testing['id2'].values

# computing features for training set
for i in tqdm(range(len(id1))):
    score_1_2.append(get_score(id1[i], id2[i], average_len))
    score_2_1.append(get_score(id2[i], id1[i], average_len))
    cosine_dist.append(cosine_distance(id1[i], id2[i]))

# add feature to data-frame
testing["score_1_2"] = score_1_2
testing["score_2_1"] = score_2_1
testing["cosine_distance"] = cosine_dist

# save data-frame
training.to_csv(path_to_data + "training_features.txt")
testing.to_csv(path_to_data + "testing_features.txt")
