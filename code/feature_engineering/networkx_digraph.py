import math

import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim import corpora, models
import networkx as nx

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

G = nx.DiGraph()
G.add_nodes_from(nodes.index.values)
G.add_edges_from(zip(training[training["target"] == 1]["id1"], training[training["target"] == 1]["id2"]))

# IDs for training set
id1 = training['id1'].values
id2 = training['id2'].values

# placeholder for feature
n = len(id1)
out_neighbors = np.zeros(n)
in_neighbors = np.zeros(n)
popularity = np.zeros(n)

# computing features for training set
for i in tqdm(range(len(id1))):
    if training.at[str(id1[i]) + "|" + str(id2[i]), "target"] == 1:
        G.remove_edge(id1[i], id2[i])

    in_neighbors[i] = G.in_degree(id2[i])
    out_neighbors[i] = G.out_degree(id1[i])

    predecessors = G.predecessors(id2[i])
    pop = 0
    for p in predecessors:
        pop += G.in_degree(p)

    popularity[i] = pop

    if training.at[str(id1[i]) + "|" + str(id2[i]), "target"] == 1:
        G.add_edge(id1[i], id2[i])

# add feature to dataframe
training["out_neighbors"] = out_neighbors
training["in_neighbors"] = in_neighbors
training["popularity"] = popularity

# IDs for training set
id1 = testing['id1'].values
id2 = testing['id2'].values

# placeholder for feature
n = len(id1)
out_neighbors = np.zeros(n)
in_neighbors = np.zeros(n)
popularity = np.zeros(n)

# computing features for training set
for i in tqdm(range(len(id1))):
    if testing.at[str(id1[i]) + "|" + str(id2[i]), "target"] == 1:
        G.remove_edge(id1[i], id2[i])

    in_neighbors[i] = G.in_degree(id2[i])
    out_neighbors[i] = G.out_degree(id1[i])

    predecessors = G.predecessors(id2[i])
    pop = 0
    for p in predecessors:
        pop += G.in_degree(p)

    popularity[i] = pop

    if testing.at[str(id1[i]) + "|" + str(id2[i]), "target"] == 1:
        G.add_edge(id1[i], id2[i])

# add feature to dataframe
testing["out_neighbors"] = out_neighbors
testing["in_neighbors"] = in_neighbors
testing["popularity"] = popularity

# save dataframe
training.to_csv(path_to_data + "training_features.txt")
testing.to_csv(path_to_data + "testing_features.txt")
