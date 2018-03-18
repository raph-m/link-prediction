# feature engineering

import time

start = time.time()
print("preprocessing:")

end = time.time()
print("done in: " + str(end - start))

start = time.time()
print("baseline_feature_engineering:")

end = time.time()
print("done in: " + str(end - start))

start = time.time()
print("basic_features:")

end = time.time()
print("done in: " + str(end - start))

start = time.time()
print("cosine_distance:")

end = time.time()
print("done in: " + str(end - start))

start = time.time()
print("networkx_bigraph:")

end = time.time()
print("done in: " + str(end - start))

start = time.time()
print("networkx_digraph:")

end = time.time()
print("done in: " + str(end - start))

start = time.time()
print("author's features:")

end = time.time()
print("done in: " + str(end - start))

start = time.time()
print("author's features:")

end = time.time()
print("done in: " + str(end - start))

# models : train them and store the output probits for stacking purposes

start = time.time()
print("SVM:")

end = time.time()
print("done in: " + str(end - start))

start = time.time()
print("Random Forest:")

end = time.time()
print("done in: " + str(end - start))

start = time.time()
print("LightGBM:")

end = time.time()
print("done in: " + str(end - start))

start = time.time()
print("shallow NN:")

end = time.time()
print("done in: " + str(end - start))

start = time.time()
print("deep NN:")

end = time.time()
print("done in: " + str(end - start))

# train the model stack and generate final submission "stack_sub_rf.csv"

start = time.time()
print("stack :")

end = time.time()
print("done in: " + str(end - start))
