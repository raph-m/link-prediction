import igraph
import numpy as np
import pandas as pd
from tqdm import tqdm

from tools import lit_eval_nan_proof

# progress bar for pandas
tqdm.pandas(tqdm())

# path
path_to_data = "../data/"

# loading data
converter_dict = {'authors': lit_eval_nan_proof, 'journal': lit_eval_nan_proof,
                  'title': lit_eval_nan_proof, 'abstract': lit_eval_nan_proof}
nodes = pd.read_csv(path_to_data + "nodes_preprocessed.csv", converters=converter_dict)
nodes.set_index("id", inplace=True)
training = pd.read_csv(path_to_data + "training_features.txt")
training.set_index("my_index", inplace=True)
testing = pd.read_csv(path_to_data + "testing_features.txt")
testing.set_index("my_index", inplace=True)

# placeholders for graph features
shortest_path = []

# IDs for training set
id1 = training['id1'].values
id2 = training['id2'].values
target = training["target"].values

# creating graph of training set

# create empty directed graph
g = igraph.Graph(directed=True)

# some nodes may not be connected to any other node
# hence the need to create the nodes of the graph from node_info.csv,
# not just from the edge list
nodes = nodes.index.values
str_vec = np.vectorize(str)
nodes = str_vec(nodes)

# add vertices
g.add_vertices(nodes)

# add edges
edges = [(str(id1[i]), str(id2[i])) for i in range(len(id1)) if target[i] == 1]
print("here")
print(edges)
g.add_edges(edges)

for i in tqdm(range(len(id1))):
    if target[i] == 1:
        shortest_path.append(1)
    else:
        shortest_path.append(g.shortest_paths_dijkstra(source=str(id1[i]), target=str(id2[i]), mode="OUT"))
        
# adding feature to dataframe
training["shortest_path"] = shortest_path

# repeat process for test set
shortest_path_test = []
id1 = testing['id1'].values
id2 = testing['id2'].values
target = testing["target"].values
edges = [(id1[i], id2[i]) for i in range(len(id1)) if target[i] == 1]
g = igraph.Graph(directed=True)
g.add_vertices(nodes)
g.add_edges(edges)
for i in tqdm(range(len(id1))):
    if target[i] == 1:
        shortest_path_test.append(1)
    else:
        shortest_path.append(g.shortest_paths_dijkstra(source=id1[i], target=id2[i], mode="OUT"))
testing["shortest_path"] = shortest_path_test

# save data sets
training.to_csv(path_to_data + "training_features.txt")
testing.to_csv(path_to_data + "testing_features.txt")

