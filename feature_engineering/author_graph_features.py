import time
from itertools import permutations, product

import igraph
import numpy as np
import pandas as pd
from feature_engineering.tools import lit_eval_nan_proof
from tqdm import tqdm

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
edges = {}
# store edge ids for each edge
ids = {}
# store weights
weights = {}
id1 = training['id1'].values
id2 = training['id2'].values
index_train = training.index
target = training["target"].values
# edge id
id = 0
# store all the edges related to each citation
eid = {}
for i in tqdm(range(len(id1))):
    # if there is a 
    if target[i] == 1:
        authors1 = nodes.at[id1[i], 'authors']
        authors2 = nodes.at[id2[i], 'authors']
        # check that author information is not missing
        if isinstance(authors1, float) or isinstance(authors2, float):
            continue
        # if authors available then add edges
        pairs = list(product(authors1, authors2))
        # for each pair of authors
        for pair in pairs:
            # if edge already exists
            if pair in edges:
                # increment weight
                weights[pair] += 1
                # add id to edges related to this citation
                if index_train[i] in eid:
                    eid[index_train[i]] += [id]
                else:
                    eid[index_train[i]] = [id]
            # if doesn't exist
            else:
                # create edge
                edges[pair] = 1
                # keep track of edge id
                ids[pair] = id
                # add id to edges related to this citation
                if index_train[i] in eid:
                    eid[index_train[i]] += [id]
                else:
                    eid[index_train[i]] = [id]
                # store weight
                weights[pair] = 1
                # increment id
                id += 1

# then, add coauthor edges
authors_array = authors.values
index_nodes = nodes.index.values
# for each document
for i in tqdm(range(len(authors_array))):
    # if missing author info, skip
    if isinstance(authors_array[i], float):
        continue
    # if not for each pair of coauthors
    coauthors = permutations(authors_array[i], 2)
    for pair in coauthors:
        # if edge already exists
        if pair in edges:
            # increment weight
            weights[pair] += 2
        # if doesn't exist
        else:
            # create edge
            edges[pair] = 1
            # store weight
            weights[pair] = 2
        
# add edges to graph
g.add_edges(list(edges))

# add weights
weights = list(edges.values())
max_weight = max(weights)
weights = max_weight - np.array(weights) + 1
g.es['weight'] = list(weights)

# compute features such as shortest path

# features placeholders
min_shortest_path = []
max_shortest_path = []
mean_shortest_path = []
author_in_degree_mean_target = []
author_in_degree_max_target = []
author_out_degree_mean_source = []
author_out_degree_max_source = []
author_common_neigbors_mean = []
author_common_neigbors_max = []
author_jaccard_mean = []
author_jaccard_max = []

# get training ids
id1 = training['id1'].values
id2 = training['id2'].values
target = training["target"].values
index_train = training.index

# compute features for all samples
for i in tqdm(range(len(id1))):
    authors1 = nodes.at[id1[i], 'authors']
    authors2 = nodes.at[id2[i], 'authors']
    # if one of the articles has missing author info
    if isinstance(authors1, float) or isinstance(authors2, float):
        # print("NAN")
        # no shortest path can be computed
        min_shortest_path.append(np.nan)
        max_shortest_path.append(np.nan)
        mean_shortest_path.append(np.nan)
        # if author info is missing for first doc
        if isinstance(authors1, float):
            # no degree can be computed
            author_out_degree_max_source.append(np.nan)
            author_out_degree_mean_source.append(np.nan)
        # if not missing
        else:
            # compute degrees
            out = g.strength(authors1, weights="weight")
            mean_out = np.mean(out)
            max_out = np.max(out)
            author_out_degree_max_source.append(max_out)
            author_out_degree_mean_source.append(mean_out)
        # if it is missing for the second document
        if isinstance(authors2, float):
            # no degree can be computed
            author_in_degree_max_target.append(np.nan)
            author_in_degree_mean_target.append(np.nan)
        # if not
        else:
            # compute degrees for other document
            in_ = g.strength(authors2, weights="weight")
            mean_in = np.mean(in_)
            max_in = np.max(in_)
            author_in_degree_max_target.append(max_in)
            author_in_degree_mean_target.append(mean_in)
        continue
    # print("NO NAN")
    # if there's no missing author information
    # set weights of unwanted edges to zero
    if target[i] == 1:
        # print('target is 1')
        t0 = time.time()
        # print('fetching edge ids')
        eids_to_unweigh = eid[index_train[i]]
        t1 = time.time()
        for id in eids_to_unweigh:
            g.es['weight'][id] += 1
        t1_bis = time.time()
        print('bottleneck', t1 - t0, t1_bis - t1)
    # compute shortest paths
    # print("computing shortest path")
    t1 = time.time()
    # paths = g.shortest_paths_dijkstra(source=authors1, target=authors2,
    #                                             mode="OUT", weights="weight")[0][0]
    # min_value = np.min(paths)
    # max_value = np.max(paths)
    # mean_value = np.mean(paths)
    t2 = time.time()
    print('shortest_path', t2 - t1)
    # compute degrees
    out = g.strength(authors1, weights="weight")
    in_ = g.strength(authors2, weights="weight")
    mean_out = np.mean(out)
    max_out = np.max(out)
    in_ = g.strength(authors2, weights="weight")
    mean_in = np.mean(in_)
    max_in = np.max(in_)
    t3 = time.time()
    print('weighted degree', t3 - t2)
    # create set of pairs as vertex ids as well as index values
    pairs = list(product(authors1, authors2))
    pairs_index = list(product(range(len(authors1)), range(len(authors2))))
    # compute jaccard similarity
    # jaccards = g.similarity_jaccard(pairs=pairs)
    # max_jacc = np.max(jaccards)
    # mean_jacc = np.mean(jaccards)
    t4 = time.time()
    # print('jacc', t4 - t3)
    # compute common neighbours
    hoods1 = g.neighborhood(vertices=authors1)
    hoods2 = g.neighborhood(vertices=authors2)
    common_hoods = [set(hoods1[i]).intersection(set(hoods2[j])) for (i, j) in pairs_index]
    common_hoods_size = list(map(len, common_hoods))
    max_hood = np.max(common_hoods_size)
    mean_hood = np.mean(common_hoods_size)
    t5 = time.time()
    print('common hoods', t5 - t4)
    # append features to corresponding set
    # min_shortest_path.append(min_value)
    # max_shortest_path.append(max_value)
    # mean_shortest_path.append(mean_value)
    author_out_degree_max_source.append(max_out)
    author_out_degree_mean_source.append(mean_out)
    author_in_degree_max_target.append(max_in)
    author_in_degree_mean_target.append(mean_in)
    author_common_neigbors_mean.append(mean_hood)
    author_common_neigbors_max.append(max_hood)
    # author_jaccard_mean.append(max_jacc)
    # author_jaccard_max.append(mean_jacc)
    if target[i] == 1:
        for id in eids_to_unweigh:
            g.es['weight'][id] = 0
    t6 = time.time()
    # print("append features", t6 - t5)

# add feature to dataframe
# training["author_min_shortest_path"] = min_shortest_path
# training["author_max_shortest_path"] = max_shortest_path
# training["author_sum_shortest_path"] = sum_shortest_path
# training["author_mean_shortest_path"] = mean_shortest_path
training['author_out_degree_max_source'] = author_out_degree_max_source
training['author_out_degree_mean_source'] = author_out_degree_mean_source
training['author_in_degree_max_target'] = author_in_degree_max_target
training['author_in_degree_mean_target'] = author_in_degree_mean_target
training['author_common_neigbors_mean'] = author_common_neigbors_mean
training['author_common_neigbors_max'] = author_common_neigbors_max
# training['author_jaccard_mean'] = author_jaccard_mean
# training['author_jaccard_max'] = author_jaccard_max

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
