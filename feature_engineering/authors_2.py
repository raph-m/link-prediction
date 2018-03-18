import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import time

from feature_engineering.tools import lit_eval_nan_proof

# this script computes the features authors_in_neighbors and authors_common_neighbors by considering
# the author's graph of citations.
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

    authors = np.unique([a for a in authors if a != ""])

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

    current_authors_1 = np.unique([a for a in current_authors_1 if a != ""])
    current_authors_2 = np.unique([a for a in current_authors_2 if a != ""])

    if training.at[str(id1[i]) + "|" + str(id2[i]), "target"] == 1:
        for a1 in current_authors_1:
            for a2 in current_authors_2:
                if G.has_edge(a1, a2):
                    G[a1][a2]["weight"] += 1
                else:
                    G.add_edge(a1, a2, weight=1)


authors_in_neighbors = np.zeros(len(id1))
normalized_authors_in_neighbors = np.zeros(len(id1))
best_authors_in_neighbors = np.zeros(len(id1))
authors_common_neighbors = np.zeros(len(id1))

print("building features for training")
for i in range(len(id1)):
    if i % 1000 == 0:
        print(i)
        print(time.time())
    current_authors_1 = nodes.loc[id1[i]]["authors"]
    current_authors_2 = nodes.loc[id2[i]]["authors"]

    if current_authors_1 is np.nan:
        current_authors_1 = []

    if current_authors_2 is np.nan:
        current_authors_2 = []

    current_authors_1 = np.unique([a for a in current_authors_1 if a != ""])
    current_authors_2 = np.unique([a for a in current_authors_2 if a != ""])

    if training.at[str(id1[i]) + "|" + str(id2[i]), "target"] == 1:
        for a1 in current_authors_1:
            for a2 in current_authors_2:
                G[a1][a2]["weight"] -= 1

    # this feature is commented because too long to compute
    # for a1 in current_authors_1:
    #     for p in G.successors(a1):
    #         for a2 in G.successors(p):
    #             if a2 in current_authors_2:
    #                 authors_common_neighbors[i] += min(G[a1][p]["weight"], G[p][a2]["weight"])

    best = 0
    for a1 in current_authors_2:
        current = len([g for g in G.predecessors(a1)])
        authors_in_neighbors[i] += current
        if current > best:
            best = current

    best_authors_in_neighbors[i] = best

    # normalize feature
    denom = len(current_authors_2)
    if denom > 0:
        normalized_authors_in_neighbors[i] = authors_in_neighbors[i] / denom

    if training.at[str(id1[i]) + "|" + str(id2[i]), "target"] == 1:
        for a1 in current_authors_1:
            for a2 in current_authors_2:
                G[a1][a2]["weight"] += 1

training["authors_in_neighbors"] = authors_in_neighbors
training["normalized_authors_in_neighbors"] = normalized_authors_in_neighbors
training["best_authors_in_neighbors"] = best_authors_in_neighbors
training["authors_common_neighbors"] = authors_common_neighbors

id1 = testing["id1"].values
id2 = testing["id2"].values

authors_in_neighbors = np.zeros(len(id1))
normalized_authors_in_neighbors = np.zeros(len(id1))
best_authors_in_neighbors = np.zeros(len(id1))
authors_common_neighbors = np.zeros(len(id1))

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

    current_authors_1 = np.unique([a for a in current_authors_1 if a != ""])
    current_authors_2 = np.unique([a for a in current_authors_2 if a != ""])

    # for a1 in current_authors_1:
    #     for p in G.successors(a1):
    #         for a2 in G.successors(p):
    #             if a2 in current_authors_2:
    #                 authors_common_neighbors[i] += min(G[a1][p]["weight"], G[p][a2]["weight"])

    best = 0
    for a1 in current_authors_2:
        current = len([g for g in G.predecessors(a1)])
        authors_in_neighbors[i] += current
        if current > best:
            best = current

    best_authors_in_neighbors[i] = best

    # normalize feature
    denom = len(current_authors_2)
    if denom > 0:
        normalized_authors_in_neighbors[i] = authors_in_neighbors[i] / denom

testing["authors_in_neighbors"] = authors_in_neighbors
testing["normalized_authors_in_neighbors"] = normalized_authors_in_neighbors
testing["best_authors_in_neighbors"] = best_authors_in_neighbors
testing["authors_common_neighbors"] = authors_common_neighbors

print("done, saving data")
# save data-frame
training.to_csv(path_to_data + "training_features.txt")
testing.to_csv(path_to_data + "testing_features.txt")

