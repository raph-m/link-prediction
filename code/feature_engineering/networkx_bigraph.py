import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx

from code.feature_engineering.tools import lit_eval_nan_proof

# this script computes some features by considering the bidirectional graph of citations: jaccard, adar,
#  preferential_attachment, resource_allocation_index and common_neighbors
# approx 10 minutes to run it
# NB: the katz feature is not computed since it requires way too much computer power.

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
jaccard = np.zeros(n)
adar = np.zeros(n)
preferential_attachment = np.zeros(n)
resource_allocation_index = np.zeros(n)
common_neighbors = np.zeros(n)

# computing features for training set
for i in tqdm(range(len(id1))):
    if training.at[str(id1[i]) + "|" + str(id2[i]), "target"] == 1:
        G.remove_edge(id1[i], id2[i])

    pred = nx.jaccard_coefficient(G, [(id1[i], id2[i])])
    pred = [(u, v, p) for (u, v, p) in pred]
    jaccard[i] = pred[0][2]

    pred = nx.adamic_adar_index(G, [(id1[i], id2[i])])
    pred = [(u, v, p) for (u, v, p) in pred]
    adar[i] = pred[0][2]

    pred = nx.preferential_attachment(G, [(id1[i], id2[i])])
    pred = [(u, v, p) for (u, v, p) in pred]
    preferential_attachment[i] = pred[0][2]

    pred = nx.resource_allocation_index(G, [(id1[i], id2[i])])
    pred = [(u, v, p) for (u, v, p) in pred]
    resource_allocation_index[i] = pred[0][2]

    pred = nx.common_neighbors(G, id1[i], id2[i])
    pred = len([u for u in pred])
    common_neighbors[i] = pred

    if training.at[str(id1[i]) + "|" + str(id2[i]), "target"] == 1:
        G.add_edge(id1[i], id2[i])

# add feature to data-frame
training["jaccard"] = jaccard
training["adar"] = adar
training["preferential_attachment"] = preferential_attachment
training["resource_allocation_index"] = resource_allocation_index
training["common_neighbors"] = resource_allocation_index


# IDs for training set
id1 = testing['id1'].values
id2 = testing['id2'].values

# placeholder for feature
n = len(id1)
jaccard = np.zeros(n)
adar = np.zeros(n)
preferential_attachment = np.zeros(n)
resource_allocation_index = np.zeros(n)
common_neighbors = np.zeros(n)

# computing features for training set
for i in tqdm(range(len(id1))):
    pred = nx.jaccard_coefficient(G, [(id1[i], id2[i])])
    pred = [(u, v, p) for (u, v, p) in pred]
    jaccard[i] = pred[0][2]

    pred = nx.adamic_adar_index(G, [(id1[i], id2[i])])
    pred = [(u, v, p) for (u, v, p) in pred]
    adar[i] = pred[0][2]

    pred = nx.preferential_attachment(G, [(id1[i], id2[i])])
    pred = [(u, v, p) for (u, v, p) in pred]
    preferential_attachment[i] = pred[0][2]

    pred = nx.resource_allocation_index(G, [(id1[i], id2[i])])
    pred = [(u, v, p) for (u, v, p) in pred]
    resource_allocation_index[i] = pred[0][2]

    pred = nx.common_neighbors(G, id1[i], id2[i])
    pred = len([u for u in pred])
    common_neighbors[i] = pred

# add feature to data-frame
testing["jaccard"] = jaccard
testing["adar"] = adar
testing["preferential_attachment"] = preferential_attachment
testing["resource_allocation_index"] = resource_allocation_index
testing["common_neighbors"] = resource_allocation_index


# save data-frame
training.to_csv(path_to_data + "training_features.txt")
testing.to_csv(path_to_data + "testing_features.txt")


# bout de code pour katz:
# katz = 0.0
# counter = 0
# try:
#     iterator = nx.all_shortest_paths(G, source=id1[i], target=id2[i])
#     for p in iterator:
#         len_p = len(p)
#         katz += len_p * beta ** (len_p)
#         counter += 1
#         if counter >= 1:
#             break
# except:
#     pass

