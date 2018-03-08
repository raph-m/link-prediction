import time
from itertools import permutations

import igraph
import numpy as np
import pandas as pd
from code.feature_engineering.tools import lit_eval_nan_proof
from tqdm import tqdm

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

# create author graph
# vertices are authors
# edge of weight 1 if they cowrote a paper, 2 if they only cite each other

# create empty directed graph
g = igraph.Graph(directed=True)

# add vertices
authors = nodes['authors']
authors_set = list(set(authors.dropna().sum()))
g.add_vertices(authors_set)

# first, add citation edges
citation_edges = []
index = []
id1 = training['id1'].values
id2 = training['id2'].values
index_train = training.index
target = training["target"].values
for i in tqdm(range(len(id1))):
    if target[i] == 1:
        authors1 = nodes.at[id1[i], 'authors']
        authors2 = nodes.at[id2[i], 'authors']
        if isinstance(authors1, float) or isinstance(authors2, float):
            continue
        for author1 in authors1:
            for author2 in authors2:
                citation_edges.append((author1, author2))
                index.append(index_train[i])
weights = [2] * len(citation_edges)
g.add_edges(citation_edges)

# then, add coauthor edges
authors_array = authors.values
coauthor_edges = []
index_nodes = nodes.index.values
for i in tqdm(range(len(authors_array))):
    if isinstance(authors_array[i], float):
        continue
    coauthors = permutations(authors_array[i], 2)
    for pair in coauthors:
        coauthor_edges.append((pair[0], pair[1]))
        index.append(index_nodes[i])
weights += [1] * len(coauthor_edges)
g.add_edges(coauthor_edges)

# add weights and index
g.es['weight'] = weights
g.es['index'] = index

# compute features such as shortest path

# features placeholders
min_shortest_path = []
max_shortest_path = []
sum_shortest_path = []
mean_shortest_path = []
author_in_degree_mean_target = []
author_in_degree_sum_target = []
author_out_degree_mean_source = []
author_out_degree_sum_source = []

# get training ids
id1 = training['id1'].values
id2 = training['id2'].values
index_train = training.index

# compute features
for i in tqdm(range(len(id1))):
    authors1 = nodes.at[id1[i], 'authors']
    authors2 = nodes.at[id2[i], 'authors']
    if isinstance(authors1, float) or isinstance(authors2, float):
        print("NAN")
        min_shortest_path.append(np.nan)
        max_shortest_path.append(np.nan)
        sum_shortest_path.append(np.nan)
        mean_shortest_path.append(np.nan)
        if isinstance(authors1, float):
            author_out_degree_sum_source.append(np.nan)
            author_out_degree_mean_source.append(np.nan)
        else:
            t0 = time.time()
            edgseq_to_remove = g.es.select(index=id1[i])
            edge_tuple = []
            for e in edgseq_to_remove:
                edge_tuple.append(e.tuple)
            edge_weights = [1] * len(edge_tuple)
            edge_index = [id1[i]] * len(edge_tuple)
            g.delete_edges(index=id1[i])
            t1 = time.time()
            print(t1 - t0)
            sum_out = 0
            n_source = len(authors1)
            for author1 in authors1:
                sum_out += g.strength(author1, mode='OUT', weights="weight")
            t2 = time.time()
            print(t2 - t1)
            mean_out = sum_out / n_source
            author_out_degree_sum_source.append(sum_out)
            author_out_degree_mean_source.append(mean_out)
            g.add_edges(edge_tuple)
            g.es[len(g.es) - len(edge_tuple):]["weight"] = edge_weights
            g.es[len(g.es) - len(edge_tuple):]["index"] = edge_index
            t3 = time.time()
            print(t3 - t2)
        if isinstance(authors2, float):
            author_in_degree_sum_target.append(np.nan)
            author_in_degree_mean_target.append(np.nan)
        else:
            edgseq_to_remove = g.es.select(index=id2[i])
            edge_tuple = []
            for e in edgseq_to_remove:
                edge_tuple.append(e.tuple)
            edge_weights = [1] * len(edge_tuple)
            edge_index = [id2[i]] * len(edge_tuple)
            g.delete_edges(index=id2[i])
            sum_in = 0
            n_target = len(authors2)
            for author2 in authors2:
                sum_in += g.strength(author2, mode='IN', weights="weight")
            mean_in = sum_in / n_target
            author_in_degree_sum_target.append(sum_in)
            author_in_degree_mean_target.append(mean_in)
            g.add_edges(edge_tuple)
            g.es[len(g.es) - len(edge_tuple):]["weight"] = edge_weights
            g.es[len(g.es) - len(edge_tuple):]["index"] = edge_index
        continue
    print("NO NAN")
    t0 = time.time()
    edgseq_to_remove = g.es.select(index_in=[index_train[i], id1[i], id2[i]])
    edge_tuple = []
    edge_weights = []
    edge_index = []
    for e in edgseq_to_remove:
        edge_tuple.append(e.tuple)
        edge_weights.append(e["weight"])
        edge_index.append(e["index"])
    g.delete_edges(index=index_train[i])
    t1 = time.time()
    print(t1 - t0)
    min_value = float('inf')
    max_value = - float('inf')
    sum_value = 0
    n = len(authors1) * len(authors2)
    for author1 in authors1:
        for author2 in authors2:
            current = g.shortest_paths_dijkstra(source=author1, target=author2,
                                                mode="OUT", weights="weight")[0][0]
            min_value = current if current < min_value else min_value
            max_value = current if current > max_value else max_value
            sum_value += current
    mean_value = sum_value / n
    t2 = time.time()
    print(t2 - t1)
    sum_out = 0
    sum_in = 0
    n_source = len(authors1)
    n_target = len(authors2)
    for author1 in authors1:
        sum_out += g.strength(author1, mode='OUT', weights="weight")
    for author2 in authors2:
        sum_in += g.strength(author2, mode='IN', weights="weight")
    mean_out = sum_out / n_source
    mean_in = sum_in / n_target
    t3 = time.time()
    print(t3 - t2)
    min_shortest_path.append(min_value)
    max_shortest_path.append(max_value)
    sum_shortest_path.append(sum_value)
    mean_shortest_path.append(mean_value)
    author_out_degree_sum_source.append(sum_out)
    author_out_degree_mean_source.append(mean_out)
    author_in_degree_sum_target.append(sum_in)
    author_in_degree_mean_target.append(mean_in)
    g.add_edges(edge_tuple)
    g.es[len(g.es) - len(edge_tuple):]["weight"] = edge_weights
    g.es[len(g.es) - len(edge_tuple):]["index"] = edge_index
    t4 = time.time()
    print(t4 - t3)

# add feature to dataframe
training["author_min_shortest_path"] = min_shortest_path
training["author_max_shortest_path"] = max_shortest_path
training["author_sum_shortest_path"] = sum_shortest_path
training["author_mean_shortest_path"] = mean_shortest_path
training['author_out_degree_sum_source'] = author_out_degree_sum_source
training['author_out_degree_mean_source'] = author_out_degree_mean_source
training['author_in_degree_sum_target'] = author_in_degree_sum_target
training['author_in_degree_mean_target'] = author_in_degree_mean_target

# repeat process for test set
min_shortest_path_test = []
max_shortest_path_test = []
sum_shortest_path_test = []
mean_shortest_path_test = []
author_in_degree_mean_target_test = []
author_in_degree_sum_target_test = []
author_out_degree_mean_source_test = []
author_out_degree_sum_source_test = []
id1 = testing['id1'].values
id2 = testing['id2'].values
for i in tqdm(range(len(id1))):
    authors1 = nodes.at[id1[i], 'authors']
    authors2 = nodes.at[id2[i], 'authors']
    if isinstance(authors1, float) or isinstance(authors2, float):
        min_shortest_path_test.append(np.nan)
        max_shortest_path_test.append(np.nan)
        sum_shortest_path_test.append(np.nan)
        mean_shortest_path_test.append(np.nan)
        if isinstance(authors1, float):
            author_out_degree_sum_source_test.append(np.nan)
            author_out_degree_mean_source_test.append(np.nan)
        else:
            sum_out = 0
            n_source = len(authors1)
            for author1 in authors1:
                sum_out += g.strength(author1, mode='OUT', weights="weight")
            mean_out = sum_out / n_source
            author_out_degree_sum_source_test.append(sum_out)
            author_out_degree_mean_source_test.append(mean_out)
        if isinstance(authors2, float):
            author_in_degree_sum_target_test.append(np.nan)
            author_in_degree_mean_target_test.append(np.nan)
        else:
            sum_in = 0
            n_target = len(authors2)
            for author2 in authors2:
                sum_in += g.strength(author2, mode='IN', weights="weight")
            mean_in = sum_in / n_target
            author_in_degree_sum_target_test.append(sum_in)
            author_in_degree_mean_target_test.append(mean_in)
        continue
    min_value = float('inf')
    max_value = - float('inf')
    sum_value = 0
    n = len(authors1) * len(authors2)
    for author1 in authors1:
        for author2 in authors2:
            current = g.shortest_paths_dijkstra(source=author1, target=author2,
                                                mode="OUT", weights=g.es["weight"])[0][0]
            min_value = current if current < min_value else min_value
            max_value = current if current > max_value else max_value
            sum_value += current
    mean_value = sum_value / n
    sum_out = 0
    sum_in = 0
    n_source = len(authors1)
    n_target = len(authors2)
    for author1 in authors1:
        sum_out += g.strength(author1, mode='OUT', weights="weight")
    for author2 in authors2:
        sum_in += g.strength(author2, mode='IN', weights="weight")
    mean_out = sum_out / n_source
    mean_in = sum_in / n_target
    min_shortest_path_test.append(min_value)
    max_shortest_path_test.append(max_value)
    sum_shortest_path_test.append(sum_value)
    mean_shortest_path_test.append(mean_value)
    author_out_degree_sum_source_test.append(sum_out)
    author_out_degree_mean_source_test.append(mean_out)
    author_in_degree_sum_target_test.append(sum_in)
    author_in_degree_mean_target_test.append(mean_in)

# add feature to dataframe
testing["author_min_shortest_path"] = min_shortest_path_test
testing["author_max_shortest_path"] = max_shortest_path_test
testing["author_sum_shortest_path"] = sum_shortest_path_test
testing["author_mean_shortest_path"] = mean_shortest_path_test
testing['author_out_degree_sum_source'] = author_out_degree_sum_source_test
testing['author_out_degree_mean_source'] = author_out_degree_mean_source_test
testing['author_in_degree_sum_target'] = author_in_degree_sum_target_test
testing['author_in_degree_mean_target'] = author_in_degree_mean_target_test

# save data sets
training.to_csv(path_to_data + "training_features.txt")
testing.to_csv(path_to_data + "testing_features.txt")
