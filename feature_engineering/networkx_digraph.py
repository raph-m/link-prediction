import networkx as nx
import numpy as np
import pandas as pd
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
common_predecessors = np.zeros(n)
common_successors = np.zeros(n)
paths_of_length_one = np.zeros(n)

# computing features for training set
for i in tqdm(range(len(id1))):
    if training.at[str(id1[i]) + "|" + str(id2[i]), "target"] == 1:
        G.remove_edge(id1[i], id2[i])

    in_neighbors[i] = G.in_degree(id2[i])
    out_neighbors[i] = G.out_degree(id1[i])

    current_common_successors = 0
    current_common_predecessors = 0
    current_paths_of_length_one = 0

    predecessors_2 = G.predecessors(id2[i])
    predecessors_1 = G.predecessors(id1[i])

    pop = 0
    for p in predecessors_2:
        pop += G.in_degree(p)
        if p in predecessors_1:
            current_common_predecessors += 1
    popularity[i] = pop

    successors_2 = G.successors(id2[i])
    successors_1 = G.successors(id1[i])

    for p in successors_1:
        if p in successors_2:
            current_common_successors += 1

    for p in successors_1:
        if p in predecessors_2:
            current_paths_of_length_one += 1

    common_successors[i] = current_common_successors
    common_predecessors[i] = current_common_predecessors
    paths_of_length_one[i] = current_paths_of_length_one

    if training.at[str(id1[i]) + "|" + str(id2[i]), "target"] == 1:
        G.add_edge(id1[i], id2[i])

# add feature to data-frame
training["out_neighbors"] = out_neighbors
training["in_neighbors"] = in_neighbors
training["popularity"] = popularity
training["common_successors"] = out_neighbors
training["common_predecessors"] = in_neighbors
training["paths_of_length_one"] = popularity

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

    in_neighbors[i] = G.in_degree(id2[i])
    out_neighbors[i] = G.out_degree(id1[i])

    current_common_successors = 0
    current_common_predecessors = 0
    current_paths_of_length_one = 0

    predecessors_2 = G.predecessors(id2[i])
    predecessors_1 = G.predecessors(id1[i])

    pop = 0
    for p in predecessors_2:
        pop += G.in_degree(p)
        if p in predecessors_1:
            current_common_predecessors += 1
    popularity[i] = pop

    successors_2 = G.successors(id2[i])
    successors_1 = G.successors(id1[i])

    for p in successors_1:
        if p in successors_2:
            current_common_successors += 1

    for p in successors_1:
        if p in predecessors_2:
            current_paths_of_length_one += 1

    common_successors[i] = current_common_successors
    common_predecessors[i] = current_common_predecessors
    paths_of_length_one[i] = current_paths_of_length_one

    popularity[i] = pop

# add feature to data-frame
testing["out_neighbors"] = out_neighbors
testing["in_neighbors"] = in_neighbors
testing["popularity"] = popularity
testing["common_successors"] = out_neighbors
testing["common_predecessors"] = in_neighbors
testing["paths_of_length_one"] = popularity

# save data-frame
training.to_csv(path_to_data + "training_features.txt")
testing.to_csv(path_to_data + "testing_features.txt")
