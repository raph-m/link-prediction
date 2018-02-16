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

    if training.at[str(id1[i]) + "|" + str(id2[i]), "target"] == 1:
        G.add_edge(id1[i], id2[i])

# add feature to dataframe
training["jaccard"] = jaccard
training["adar"] = adar
training["preferential_attachment"] = preferential_attachment
training["resource_allocation_index"] = resource_allocation_index


# IDs for training set
id1 = testing['id1'].values
id2 = testing['id2'].values

# placeholder for feature
n = len(id1)
jaccard = np.zeros(n)
adar = np.zeros(n)
preferential_attachment = np.zeros(n)
resource_allocation_index = np.zeros(n)

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

# add feature to dataframe
testing["jaccard"] = jaccard
testing["adar"] = adar
testing["preferential_attachment"] = preferential_attachment
testing["resource_allocation_index"] = resource_allocation_index


# save dataframe
training.to_csv(path_to_data + "training_features.txt")
testing.to_csv(path_to_data + "testing_features.txt")
