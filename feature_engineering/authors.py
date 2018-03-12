import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

from feature_engineering.tools import lit_eval_nan_proof

# this script computes the features out_neighbors, in_neighbors and popularity by considering the directed
# graph of citations. Popularity is the sum of in degrees of predecessors.
# the script takes approximately 5 minutes to run

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

# loading data
converter_dict = {'authors': lit_eval_nan_proof, 'journal': lit_eval_nan_proof,
                  'title': lit_eval_nan_proof, 'abstract': lit_eval_nan_proof}
nodes = pd.read_csv(path_to_data + "nodes_preprocessed.csv",
                    converters=converter_dict)
nodes.set_index("id", inplace=True)

G = nx.DiGraph()
coauthors = nx.Graph()

print("building coauthor graph")
nodes_id = nodes.index.values
for i in range(len(nodes_id)):

    authors = nodes.loc[nodes_id[i]]["authors"]
    if authors is np.nan:
        authors = []

    authors = [a for a in authors if a != ""]

    for a in authors:
        G.add_node(a)
        coauthors.add_node(a)

    for a1 in authors:
        for a2 in authors:
            if a1 != a2:
                if coauthors.has_edge(a1, a2):
                    coauthors[a1][a2]["weight"] += 1
                else:
                    coauthors.add_edge(a1, a2, weight=1)

id1 = training["id1"].values
id2 = training["id2"].values

print("building citation graph")
for i in range(len(id1)):
    if i % 100000 == 0:
        print(i)

    current_authors_1 = nodes.loc[id1[i]]["authors"]
    current_authors_2 = nodes.loc[id2[i]]["authors"]

    if current_authors_1 is np.nan:
        current_authors_1 = []

    if current_authors_2 is np.nan:
        current_authors_2 = []

    current_authors_1 = [a for a in current_authors_1 if a != ""]
    current_authors_2 = [a for a in current_authors_2 if a != ""]

    if training.at[str(id1[i]) + "|" + str(id2[i]), "target"] == 1:
        for a1 in current_authors_1:
            for a2 in current_authors_2:
                if G.has_edge(a1, a2):
                    G[a1][a2]["weight"] += 1
                else:
                    G.add_edge(a1, a2, weight=1)


coauthor_score = np.zeros(len(id1))
normalized_coauthor_score = np.zeros(len(id1))
best_coauthor_score = np.zeros(len(id1))
authors_citation = np.zeros(len(id1))
normalized_authors_citation = np.zeros(len(id1))
best_authors_citation = np.zeros(len(id1))

print("building features for training")
for i in range(len(id1)):
    if i % 100000 == 0:
        print(i)
    current_authors_1 = nodes.loc[id1[i]]["authors"]
    current_authors_2 = nodes.loc[id2[i]]["authors"]

    if current_authors_1 is np.nan:
        current_authors_1 = []

    if current_authors_2 is np.nan:
        current_authors_2 = []

    current_authors_1 = [a for a in current_authors_1 if a != ""]
    current_authors_2 = [a for a in current_authors_2 if a != ""]

    if training.at[str(id1[i]) + "|" + str(id2[i]), "target"] == 1:
        for a1 in current_authors_1:
            for a2 in current_authors_2:
                G[a1][a2]["weight"] -= 1

    best = 0
    for a1 in current_authors_1:
        for a2 in current_authors_2:
            if G.has_edge(a1, a2):
                current = G[a1][a2]["weight"]
                authors_citation[i] += current
                if current > best:
                    best = current

    best_authors_citation[i] = best

    best = 0
    for a1 in current_authors_1:
        for a2 in current_authors_2:
            if coauthors.has_edge(a1, a2):
                current = coauthors[a1][a2]["weight"]
                coauthor_score[i] += current
                if current > best:
                    best = current

    best_coauthor_score[i] = best

    denom = len(current_authors_1) * len(current_authors_2)
    if denom > 0:
        normalized_authors_citation[i] = authors_citation[i] / denom
        normalized_coauthor_score[i] = coauthor_score[i] / denom

    if training.at[str(id1[i]) + "|" + str(id2[i]), "target"] == 1:
        for a1 in current_authors_1:
            for a2 in current_authors_2:
                G[a1][a2]["weight"] += 1

training["authors_citation"] = authors_citation
training["normalized_authors_citation"] = normalized_authors_citation
training["coauthor_score"] = coauthor_score
training["normalized_coauthor_score"] = normalized_coauthor_score
training["best_coauthor_score"] = best_coauthor_score
training["best_authors_citation"] = best_authors_citation

id1 = testing["id1"].values
id2 = testing["id2"].values

coauthor_score = np.zeros(len(id1))
normalized_coauthor_score = np.zeros(len(id1))
best_coauthor_score = np.zeros(len(id1))
authors_citation = np.zeros(len(id1))
normalized_authors_citation = np.zeros(len(id1))
best_authors_citation = np.zeros(len(id1))

print("building features for testing")
for i in range(len(id1)):
    if i % 100000 == 0:
        print(i)
    current_authors_1 = nodes.loc[id1[i]]["authors"]
    current_authors_2 = nodes.loc[id2[i]]["authors"]

    if current_authors_1 is np.nan:
        current_authors_1 = []

    if current_authors_2 is np.nan:
        current_authors_2 = []

    current_authors_1 = [a for a in current_authors_1 if a != ""]
    current_authors_2 = [a for a in current_authors_2 if a != ""]

    best = 0
    for a1 in current_authors_1:
        for a2 in current_authors_2:
            if G.has_edge(a1, a2):
                current = G[a1][a2]["weight"]
                authors_citation[i] += current
                if current > best:
                    best = current

    best_authors_citation[i] = best

    best = 0
    for a1 in current_authors_1:
        for a2 in current_authors_2:
            if coauthors.has_edge(a1, a2):
                current = coauthors[a1][a2]["weight"]
                coauthor_score[i] += current
                if current > best:
                    best = current

    best_coauthor_score[i] = best

    denom = len(current_authors_1) * len(current_authors_2)
    if denom > 0:
        normalized_authors_citation[i] = authors_citation[i] / denom
        normalized_coauthor_score[i] = coauthor_score[i] / denom

testing["authors_citation"] = authors_citation
testing["normalized_authors_citation"] = normalized_authors_citation
testing["coauthor_score"] = coauthor_score
testing["normalized_coauthor_score"] = normalized_coauthor_score
testing["best_coauthor_score"] = best_coauthor_score
testing["best_authors_citation"] = best_authors_citation

print("done, saving data")
# save data-frame
training.to_csv(path_to_data + "training_features.txt")
testing.to_csv(path_to_data + "testing_features.txt")

