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
                coauthors.add_edge(a1, a2)

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
                G.add_edge(a1, a2)


number_of_links = np.zeros(len(id1))
normalized_number_of_links = np.zeros(len(id1))
number_of_coauthors = np.zeros(len(id1))
normalized_number_of_coauthors = np.zeros(len(id1))

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
                try:
                    G.remove_edge(a1, a2)
                except:
                    pass

    for a1 in current_authors_1:
        current_successors = G.successors(a1)
        for a2 in current_authors_2:
            if a2 in current_successors:
                number_of_links[i] += 1

    for a1 in current_authors_1:
        current_neighbors = coauthors.neighbors(a1)
        for a2 in current_authors_2:
            if a2 in current_neighbors:
                number_of_coauthors[i] += 1

    denom = len(current_authors_1) * len(current_authors_2)
    if denom > 0:
        normalized_number_of_coauthors[i] = number_of_coauthors[i] / denom
        normalized_number_of_links[i] = number_of_links[i] / denom

    normalized_number_of_coauthors[i] = number_of_coauthors[i] / (len(current_authors_1) * len(current_authors_2))
    normalized_number_of_links[i] = number_of_links[i] / (len(current_authors_1) * len(current_authors_2))

    if training.at[str(id1[i]) + "|" + str(id2[i]), "target"] == 1:
        for a1 in current_authors_1:
            for a2 in current_authors_2:
                G.add_edge(a1, a2)

training["number_of_links"] = number_of_links
training["normalized_number_of_links"] = normalized_number_of_links
training["number_of_coauthors"] = number_of_coauthors
training["normalized_number_of_coauthors"] = normalized_number_of_coauthors

id1 = testing["id1"].values
id2 = testing["id2"].values

number_of_links = np.zeros(len(id1))
normalized_number_of_links = np.zeros(len(id1))
number_of_coauthors = np.zeros(len(id1))
normalized_number_of_coauthors = np.zeros(len(id1))

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

    for a1 in current_authors_1:
        current_successors = G.successors(a1)
        for a2 in current_authors_2:
            if a2 in current_successors:
                number_of_links[i] += 1

    for a1 in current_authors_1:
        current_neighbors = coauthors.neighbors(a1)
        for a2 in current_authors_2:
            if a2 in current_neighbors:
                number_of_coauthors[i] += 1

    denom = len(current_authors_1) * len(current_authors_2)
    if denom > 0:
        normalized_number_of_coauthors[i] = number_of_coauthors[i] / denom
        normalized_number_of_links[i] = number_of_links[i] / denom

testing["number_of_links"] = number_of_links
testing["normalized_number_of_links"] = normalized_number_of_links
testing["number_of_coauthors"] = number_of_coauthors
testing["normalized_number_of_coauthors"] = normalized_number_of_coauthors

print("done, saving data")
# save data-frame
training.to_csv(path_to_data + "training_features.txt")
testing.to_csv(path_to_data + "testing_features.txt")

