from util.utils import read_fbin, read_bin, get_total_nvecs_fbin, pytorch_cos_sim
from numpy import linalg
from statistics import median
import numpy as np

from torch import stack as torch_stack
from sklearn.cluster import KMeans

import datetime

import os
import nmslib

#Random seed for reproducibility of the kmeans clustering (set to None for non-determinism)
RANDOM_SEED = 505

#Total number of queries to search for (sequential)
MAXIMUM_QUERIES = 10000

#Gets an ISO string timestamp, helps with seeing how long things took to run
def ts():
    return str(datetime.datetime.now());

#Renders the filename for a shard
def shard_filename(path,name):
    return f'{path}shard{name}.hnsw'

def centroids_filename(path):
    return f'{path}centroids.hnsw'


"""
Creates a new shard graph for a centroid shard
The shard is an HNSW graph with neighborhoods of the parent centroid.
The shard is persisted to disk for each addition.
The shard is loaded from disk and searched when a query is in its centroid neighborhood.
"""
def query_shard(shard_name,query):
    shard = nmslib.init(method='hnsw', space='l2')
    shard.loadIndex(shard_name,load_data=True)
    results, distances = shard.knnQuery(query,k=10)
    return results, distances

"""
Creates the centroid index as an HNSW graph, which will be held in RAM.
Each node in the graph is a centroid that has a disk-persisted shard.
"""
def load_index(path):
    index = nmslib.init(method='hnsw', space='l2')
    index.createIndex(print_progress=True)
    index.loadIndex(centroids_filename(path))
    return index

"""
Creates the index and shard graphs for an entire dataset
"""
def query_index(path,query_file,dtype,k=10):
    
    #Get the centroid index
    print(f'Load Centroid Index: {ts()}')
    index = load_index(path)
    start_time = datetime.datetime.now().timestamp()
    print(f'Search Centroid Index for {MAXIMUM_QUERIES} queries: {ts()}')

    points = read_bin(query_file, dtype, start_idx=0, chunk_size=MAXIMUM_QUERIES)

    qnum = 0
    for query in points:
        
        #get the centroids for the query
        centroids, centroid_distances = index.knnQuery(query,k=3)
        
        #search the shard
        shard_name = shard_filename(path,centroids[0])
        results, result_distances = query_shard(shard_name,query)

        #log results
        #print(f'Found {qnum} in shard {centroids[0]}: {results[0]} {result_distances[0]} at {ts()}')
        #for i in range(len(results)):
        #    print(f'{qnum} result {i} :: {result_distances[i]} {results[i]}')
        #qnum += 1

    print(f"Done! {ts()}")
    end_time = datetime.datetime.now().timestamp()
    seconds = end_time - start_time
    print(f"Queries Per Second: {MAXIMUM_QUERIES/seconds}")

query_index("data/shards/","data/bigann/query.public.10K.u8bin",np.uint8)

"""
These settings took 7 minutes on my macbook pro with other stuff running to fit KMeans:
RANDOM_SEED = 505
SAMPLE_SIZE = 100000
M = 1000
MAX_ITER = 50
BATCH_SIZE = 1000000
"""

"""
The idea is to go *very* wide with the clustering, to increase the number of shards
For 10k centroids there are 10k shards (each with 100k vectors)
For 100k centroids there are 100k shards (each with 10k vectors)
For 1m centroids there are 1m shards (each with 1k vectors)
"""