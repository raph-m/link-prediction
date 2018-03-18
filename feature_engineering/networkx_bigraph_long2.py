import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from feature_engineering.tools import lit_eval_nan_proof

# this script computes some features by considering the bidirectional graph of citations: jaccard, adar,
#  preferential_attachment, resource_allocation_index and common_neighbors
# approx 10 minutes to run it

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

G = nx.Graph()
G.add_nodes_from(nodes.index.values)
G.add_edges_from(zip(training[training["target"] == 1]["id1"], training[training["target"] == 1]["id2"]))

# IDs for training set
id1 = training['id1'].values
id2 = training['id2'].values

# placeholder for feature
n = len(id1)
katz = np.zeros(n)
katz_2 = np.zeros(n)
beta = 0.98
beta_2 = 0.90
breaking_point = 10

# computing features for training set
for i in tqdm(range(len(id1))):
    if training.at[str(id1[i]) + "|" + str(id2[i]), "target"] == 1:
        G.remove_edge(id1[i], id2[i])

    katz_acc = 0.0
    katz_2_acc = 0.0
    counter = 0
    try:
        iterator = nx.all_shortest_paths(G, source=id1[i], target=id2[i])
        for p in iterator:
            len_p = len(p)
            katz_acc += len_p * (beta ** len_p)
            katz_2_acc += len_p * (beta_2 ** len_p)
            counter += 1
            if counter >= breaking_point:
                break
        katz[i] = katz_acc
        katz[i] = katz_2_acc
    except:
        katz[i] = -1
        katz_2[i] = -1

    if training.at[str(id1[i]) + "|" + str(id2[i]), "target"] == 1:
        G.add_edge(id1[i], id2[i])

# add feature to data-frame
training["katz"] = katz
training["katz_2"] = katz_2

# IDs for training set
id1 = testing['id1'].values
id2 = testing['id2'].values

# placeholder for feature
n = len(id1)
katz = np.zeros(n)
katz_2 = np.zeros(n)

# computing features for training set
for i in tqdm(range(len(id1))):
    katz_acc = 0.0
    katz_2_acc = 0.0
    counter = 0
    try:
        iterator = nx.all_shortest_paths(G, source=id1[i], target=id2[i])
        for p in iterator:
            len_p = len(p)
            katz_acc += len_p * (beta ** len_p)
            katz_2_acc += len_p * (beta_2 ** len_p)
            counter += 1
            if counter >= breaking_point:
                break
        katz[i] = katz_acc
        katz[i] = katz_2_acc
    except:
        katz[i] = -1
        katz_2[i] = -1

# add feature to data-frame
testing["katz"] = katz
testing["katz_2"] = katz_2

# save data-frame
training.to_csv(path_to_data + "training_features.txt")
testing.to_csv(path_to_data + "testing_features.txt")
