import time

start = time.time()
print("networkx_bigraph_long:")
import feature_engineering.networkx_bigraph_long
end = time.time()
print("done in: "+str(end-start))