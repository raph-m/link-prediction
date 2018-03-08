import numpy as np
import pandas as pd
import networkx as nx
from multiprocessing import Pool
from tqdm import tqdm

from code.feature_engineering.tools import lit_eval_nan_proof

# this script computes some features by considering the bidirectional graph of citations: katz
# approx 100 hours to run it

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

# IDs for training set
id1 = training['id1'].values
id2 = training['id2'].values

# placeholder for feature
n = len(id1)
katz = np.zeros(n)
katz_2 = np.zeros(n)
breaking_point = 10
beta = 0.98
beta_2 = 0.9


def work(i=None, id1=None, id2=None, df=None, training=True):

    G = nx.Graph()
    G.add_nodes_from(nodes.index.values)
    G.add_edges_from(zip(df[df["target"] == 1]["id1"], df[df["target"] == 1]["id2"]))

    if training:
        if df.at[str(id1[i]) + "|" + str(id2[i]), "target"] == 1:
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
    except:
        katz[i] = -1
        katz_2[i] = -1

    return katz_acc, katz_2_acc

# computing features for training set
pool = Pool()
tasks = []
for i in range(len(id1)):
    kwds = {
        "i": i,
        "id1": id1,
        "id2": id2,
        "df": training,
        "training": True
    }
    tasks.append(pool.apply_async(work, kwds=kwds))
pool.close()
pool.join()
for i in range(len(katz)):
    katz[i], katz_2[i] = tasks[i].get()

# add feature to data-frame
training["katz"] = katz
training["katz_2"] = katz_2


# IDs for testing set
id1 = testing['id1'].values
id2 = testing['id2'].values

# placeholder for feature
n = len(id1)

katz = np.zeros(n)
katz_2 = np.zeros(n)

# computing features for testing set
pool = Pool()
tasks = []
for i in range(len(id1)):
    kwds = {
        "i": i,
        "id1": id1,
        "id2": id2,
        "df": training,
        "training": True
    }
    tasks.append(pool.apply_async(work, kwds=kwds))
pool.close()
pool.join()
for i in range(len(katz)):
    katz[i], katz_2[i] = tasks[i].get()

# add feature to data-frame
testing["katz"] = katz
testing["katz_2"] = katz_2


# save data-frame
# training.to_csv(path_to_data + "training_features.txt")
# testing.to_csv(path_to_data + "testing_features.txt")
