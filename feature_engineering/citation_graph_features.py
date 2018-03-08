import igraph
import numpy as np
import pandas as pd
from tqdm import tqdm

from feature_engineering.tools import lit_eval_nan_proof

# this script adds the feature shortest_path to the files training_features and testing_features
# this script takes approximately 1000 minutes to execute

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

# placeholders for graph features
shortest_path = []

# IDs for training set
id1 = training['id1'].values
id2 = training['id2'].values
target = training["target"].values

# creating graph of citations

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

# create and add edges
edges = [(str(id1[i]), str(id2[i])) for i in range(len(id1)) if target[i] == 1]
g.add_edges(edges)

for i in tqdm(range(len(id1))):
    if target[i] == 1:
        g.delete_edges([(str(id1[i]), str(id2[i]))])
    shortest_path.append(g.shortest_paths_dijkstra(source=str(id1[i]), target=str(id2[i]), mode="OUT")[0][0])
    if target[i] == 1:
        g.add_edge(str(id1[i]), str(id2[i]))
# adding feature to dataframe
training["shortest_path"] = shortest_path

# repeat process for test set
shortest_path_test = []
id1 = testing['id1'].values
id2 = testing['id2'].values
for i in tqdm(range(len(id1))):
    shortest_path_test.append(g.shortest_paths_dijkstra(source=str(id1[i]), target=str(id2[i]), mode="OUT")[0][0])
    if target[i] == 1:
        g.add_edge(str(id1[i]), str(id2[i]))
testing["shortest_path"] = shortest_path_test

# save data sets
training.to_csv(path_to_data + "training_features.txt")
testing.to_csv(path_to_data + "testing_features.txt")

