import numpy as np
import pandas as pd
import networkx as nx
from multiprocessing import Pool
from tqdm import tqdm

from feature_engineering.tools import lit_eval_nan_proof

# this script computes some features by considering the bidirectional graph of citations: katz

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
print("start computing for training: ")
print("size of data to process: " + str(n))
katz = np.zeros(n)
katz_2 = np.zeros(n)
breaking_point = 10
beta = 0.98
beta_2 = 0.9


def work(i0=None, n=None, is_training=True):

    print(i0)
    G = nx.DiGraph()
    G.add_nodes_from(nodes.index.values)
    G.add_edges_from(zip(training[training["target"] == 1]["id1"], training[training["target"] == 1]["id2"]))

    ans = np.zeros(n)
    ans_2 = np.zeros(n)

    for i in range(n):
        if is_training:
            if training.at[str(id1[i0+i]) + "|" + str(id2[i0+i]), "target"] == 1:
                G.remove_edge(id1[i0+i], id2[i0+i])

        katz_acc = 0.0
        katz_2_acc = 0.0
        counter = 0
        try:
            iterator = nx.all_shortest_paths(G, source=id1[i0+i], target=id2[i0+i])
            for p in iterator:
                len_p = len(p)
                katz_acc += len_p * (beta ** len_p)
                katz_2_acc += len_p * (beta_2 ** len_p)
                counter += 1
                if counter >= breaking_point:
                    break
        except:
            ans[i] = -1
            ans_2[i] = -1

        if is_training:
            if training.at[str(id1[i0+i]) + "|" + str(id2[i0+i]), "target"] == 1:
                G.add_edge(id1[i0+i], id2[i0+i])

        ans[i] = katz_acc
        ans_2[i] = katz_2_acc

    print(i0)

    return ans, ans_2, i0


def callback(r):
    ans, ans_2, i0 = r

# computing features for training set

pool = Pool()
print("starting pool...")
import time
start = time.time()
n_tasks = 2000
tasks = []
step = int(n / n_tasks)
print(step)
for i0 in range(n_tasks):
    kwds = {
        "i0": i0 * step,
        "n": step,
        "is_training": True
    }
    tasks.append(pool.apply_async(work, kwds=kwds, callback=callback))
pool.close()
pool.join()
for i in range(n_tasks):
    katz[i * step: (i + 1) * step],\
        katz_2[i * step: (i+1) * step], _ = tasks[i].get()

end = time.time()
print(end-start)
# add feature to data-frame
training["katz"] = katz
training["katz_2"] = katz_2


# IDs for testing set
print("start computing for training: ")
id1 = testing['id1'].values
id2 = testing['id2'].values

# placeholder for feature
n = len(id1)
print("size of data to process: " + str(n))

katz = np.zeros(n)
katz_2 = np.zeros(n)

pool = Pool()
print("starting pool...")
n_tasks = 2000
tasks = []
step = int(n / n_tasks)
for i0 in range(n_tasks):
    kwds = {
        "i0": i0 * step,
        "n": step,
        "is_training": False
    }
    tasks.append(pool.apply_async(work, kwds=kwds, callback=callback))
pool.close()
pool.join()
for i in range(n_tasks):
    katz[i * step: (i + 1) * step],\
        katz_2[i * step: (i+1) * step], _ = tasks[i].get()

# add feature to data-frame
testing["katz"] = katz
testing["katz_2"] = katz_2

print("done, saving data")
# save data-frame
training.to_csv(path_to_data + "training_features.txt")
testing.to_csv(path_to_data + "testing_features.txt")
