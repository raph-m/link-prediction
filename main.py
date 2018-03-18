import time
start = time.time()
print("preprocessing:")
import feature_engineering.preprocessing
end = time.time()
print("done in: "+str(end-start))


start = time.time()
print("baseline_feature_engineering:")
import feature_engineering.baseline_feature_engineering
end = time.time()
print("done in: "+str(end-start))


start = time.time()
print("basic_features:")
import feature_engineering.basic_features
end = time.time()
print("done in: "+str(end-start))


start = time.time()
print("cosine_distance:")
import feature_engineering.cosine_distance
end = time.time()
print("done in: "+str(end-start))


start = time.time()
print("citation_graph_features:")
import feature_engineering.citation_graph_features
end = time.time()
print("done in: "+str(end-start))


start = time.time()
print("networkx_bigraph:")
import feature_engineering.networkx_bigraph
end = time.time()
print("done in: "+str(end-start))


start = time.time()
print("networkx_digraph:")
import feature_engineering.networkx_digraph
end = time.time()
print("done in: "+str(end-start))


start = time.time()
print("author's features:")
import feature_engineering.authors
end = time.time()
print("done in: "+str(end-start))


start = time.time()
print("author's features:")
import feature_engineering.authors_2
end = time.time()
print("done in: "+str(end-start))

# TODO: ML and stacking
