import igraph
import pandas as pd
from tqdm import tqdm
from itertools import permutations

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
                g.add_edge(source=author1, target=author2, weight=2)

# then, add coauthor edges
for author_array in tqdm(authors[:4]):
    coauthors = permutations(author_array, 2)
    for pair in coauthors:
        eid = g.get_eid(pair[0], pair[1], error=False)
        if eid >= 0:
            g.es[eid]["weight"] = 1
        else:
            g.add_edge(pair[0], pair[1], weight=1)

# compute features such as shortest path

# features placeholdes
min_shortest_path = []
max_shortest_path = []

# get training ids
id1 = testing['id1'].values
id2 = testing['id2'].values

# compute features
for i in tqdm(range(len(id1))):
    authors1 = nodes.at[id1[i], 'authors']
    authors2 = nodes.at[id2[i], 'authors']
    if isinstance(authors1, float) or isinstance(authors2, float):
        continue
    min_value = float('inf')
    max_value = - float('inf')
    for author1 in authors1:
        for author2 in authors2:
            current = g.shortest_paths_dijkstra(source=author1, target=author2, mode="OUT")[0][0]
            min_value = current if current < min_value else min_value
            max_value = current if current > max_value else max_value
    min_shortest_path.append(min_value)
    max_shortest_path.append(max_value)

# add feature to dataframe
training["author_min_shortest_path"] = min_shortest_path
training["author_max_shortest_path"] = max_shortest_path


# repeat process for test set
min_shortest_path_test = []
max_shortest_path_test = []
id1 = testing['id1'].values
id2 = testing['id2'].values
for i in tqdm(range(len(id1))):
    authors1 = nodes.at[id1[i], 'authors']
    authors2 = nodes.at[id2[i], 'authors']
    if isinstance(authors1, float) or isinstance(authors2, float):
        continue
    min_value = float('inf')
    max_value = - float('inf')
    for author1 in authors1:
        for author2 in authors2:
            current = g.shortest_paths_dijkstra(source=author1, target=author2, mode="OUT")[0][0]
            min_value = current if current < min_value else min_value
            max_value = current if current > max_value else max_value
    min_shortest_path_test.append(min_value)
    max_shortest_path_test.append(max_value)

# add feature to dataframe
testing["author_min_shortest_path"] = min_shortest_path_test
testing["author_max_shortest_path"] = max_shortest_path_test
