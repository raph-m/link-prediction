import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import permutations
import igraph

from code.feature_engineering.tools import lit_eval_nan_proof

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
# edge of weight 1 if they co-wrote a paper, 2 if they only cite each other

# create empty directed graph
g = igraph.Graph(directed=True)

# add vertices
authors = nodes['authors']
authors_set = list(set(authors.dropna().sum()))
g.add_vertices(authors_set)

# first, add citation edges
citation_edges = []
id1 = training['id1'].values
id2 = training['id2'].values
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
weights = [2] * len(citation_edges)
g.add_edges(citation_edges)
g.es['weights'] = weights

# then, add coauthor edges
coauthor_edges = []
for author_array in tqdm(authors):
    if isinstance(author_array, float):
        continue
    coauthors = permutations(author_array, 2)
    for pair in coauthors:
        eid = g.get_eid(pair[0], pair[1], error=False)
        if eid >= 0:
            g.es[eid]["weight"] = 1
        else:
            coauthor_edges.append((pair[0], pair[1]))
g.add_edges(coauthor_edges)

# compute features such as shortest path

# features placeholdes
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

# compute features
for i in tqdm(range(len(id1))):
    authors1 = nodes.at[id1[i], 'authors']
    authors2 = nodes.at[id2[i], 'authors']
    if isinstance(authors1, float) or isinstance(authors2, float):
        min_shortest_path.append(np.nan)
        max_shortest_path.append(np.nan)
        sum_shortest_path.append(np.nan)
        mean_shortest_path.append(np.nan)
        if isinstance(authors1, float):
            author_out_degree_sum_source.append(np.nan)
            author_out_degree_mean_source.append(np.nan)
        else:
            sum_out = 0
            n_source = len(authors1)
            for author1 in authors1:
                sum_out += g.degree(author1, mode='OUT')
            mean_out = sum_out / n_source
            author_out_degree_sum_source.append(sum_out)
            author_out_degree_mean_source.append(mean_out)
        if isinstance(authors2, float):
            author_in_degree_sum_target.append(np.nan)
            author_in_degree_mean_target.append(np.nan)
        else:
            sum_in = 0
            n_target = len(authors2)
            for author2 in authors2:
                sum_in += g.degree(author2, mode='IN')
            mean_in = sum_in / n_target
            author_in_degree_sum_target.append(sum_in)
            author_in_degree_mean_target.append(mean_in)
        continue
    min_value = float('inf')
    max_value = - float('inf')
    sum_value = 0
    n = len(authors1) * len(authors2)
    for author1 in authors1:
        for author2 in authors2:
            current = g.shortest_paths_dijkstra(source=author1, target=author2, mode="OUT")[0][0]
            min_value = current if current < min_value else min_value
            max_value = current if current > max_value else max_value
            sum_value += current
    mean_value = sum_value / n
    sum_out = 0
    sum_in = 0
    n_source = len(authors1)
    n_target = len(authors2)
    for author1 in authors1:
        sum_out += g.degree(author1, mode='OUT')
    for author2 in authors2:
        sum_in += g.degree(author2, mode='IN')
    mean_out = sum_out / n_source
    mean_in = sum_in / n_target
    min_shortest_path.append(min_value)
    max_shortest_path.append(max_value)
    sum_shortest_path.append(sum_value)
    mean_shortest_path.append(mean_value)
    author_out_degree_sum_source.append(sum_out)
    author_out_degree_mean_source.append(mean_out)
    author_in_degree_sum_target.append(sum_in)
    author_in_degree_mean_target.append(mean_in)


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
                sum_out += g.degree(author1, mode='OUT')
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
                sum_in += g.degree(author2, mode='IN')
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
            current = g.shortest_paths_dijkstra(source=author1, target=author2, mode="OUT")[0][0]
            min_value = current if current < min_value else min_value
            max_value = current if current > max_value else max_value
            sum_value += current
    mean_value = sum_value / n
    sum_out = 0
    sum_in = 0
    n_source = len(authors1)
    n_target = len(authors2)
    for author1 in authors1:
        sum_out += g.degree(author1, mode='OUT')
    for author2 in authors2:
        sum_in += g.degree(author2, mode='IN')
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

