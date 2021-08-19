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

#Size of the sample of points examined for during clustering
SAMPLE_SIZE = 10000

#Number of centroids to find, and also the number of shards
M = 100

#Maximum iterations of the kmeans clustering centroid fitter
MAX_ITER = 25

#Batch size for reading points from the input file during the sharding algorithm
BATCH_SIZE = 10000

#Maximum data points to index (set to None to index everything in the dataset file)
MAX_POINTS = 200000

#Gets an ISO string timestamp, helps with seeing how long things took to run
def ts():
    return str(datetime.datetime.now());

#Renders the filename for a shard
def shard_filename(path,name):
    return f'{path}shard{name}.hnsw'

def centroids_filename(path):
    return f'{path}centroids.hnsw'

#Show the extremes of the similarity scores between all the centroids
def show_distance_stats(points):
    similarities = pytorch_cos_sim(points,points)
    scores = []
    for a in range(len(similarities)-1):
        for b in range(a+1, len(similarities)):
            scores.append(float(similarities[a][b]))
    scores = sorted(scores)
    print(f'Farthest:{scores[0]}    Median:{median(scores)}     Closest:{scores[len(scores)-1]}')

"""
This will take a sample of the dataset to fit centroids that will be used as shard entry points
"""
def find_centroids(data_file, dtype, sample_size: int = SAMPLE_SIZE, n_clusters: int = M, max_iter: int = MAX_ITER):
    print(f'Loading Samples: {ts()}')
    points = read_bin(data_file, dtype, start_idx=0, chunk_size=sample_size)
    print(f'Clustering dataset shape: {points.shape}')
    print(f'Starting KMeans: {ts()}')
    if RANDOM_SEED:
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, max_iter=max_iter).fit(points)
    else:
        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter).fit(points)
    
    return kmeans.cluster_centers_

"""
Adds a batch of points to a specific shard
"""
def add_points(path,name,ids,points):
    shardpath = shard_filename(path,name)
    shard = nmslib.init(method='hnsw', space='l2')
    shard.loadIndex(shardpath,load_data=True)
    shard.addDataPointBatch(points,ids)
    shard.createIndex(print_progress=False)
    shard.saveIndex(shardpath,save_data=True)

"""
Creates a new shard graph for a centroid shard
The shard is an HNSW graph with neighborhoods of the parent centroid.
The shard is persisted to disk for each addition.
The shard is loaded from disk and searched when a query is in its centroid neighborhood.
"""
def add_shard(path,name):
    shard = nmslib.init(method='hnsw', space='l2')
    shard.createIndex(print_progress=False)
    shard.saveIndex(shard_filename(path,name),save_data=True)

"""
Creates the centroid index as an HNSW graph, which will be held in RAM.
Each node in the graph is a centroid that has a disk-persisted shard.
"""
def add_index(path,centroids):
    index = nmslib.init(method='hnsw', space='l2')
    index.addDataPointBatch(centroids)
    index.createIndex(print_progress=True)
    index.saveIndex(centroids_filename(path),save_data=True)
    return index

"""
Creates the index and shard graphs for an entire dataset
"""
def index_dataset(
        path,
        data_file, 
        dtype, 
        batch_size: int = BATCH_SIZE, 
        sample_size: int = SAMPLE_SIZE, 
        n_clusters: int = M, 
        max_iter: int = MAX_ITER,
        max_points: int = MAX_POINTS
    ):
    
    #Get the centroid index and initialize the shards
    centroids = find_centroids(data_file,dtype,sample_size=sample_size,n_clusters=n_clusters,max_iter=max_iter)
    print(f'Done Fitting KMeans: {ts()}')
    show_distance_stats(centroids)
    index = add_index(path,centroids)
    print(f'Created Centroid Index: {ts()}')
    for i in range(len(centroids)):
        add_shard(path,i)


    #Prepare for batch indexing
    total_num_elements = get_total_nvecs_fbin(data_file)
    if max_points and max_points<total_num_elements:
        range_upper = max_points
    else:
        range_upper = total_num_elements

    print(f"Total number of points in dataset: {total_num_elements}")
    print(f"Maximum number of points to index: {range_upper}")

    print(f"Reading data from {data_file} in {BATCH_SIZE} chunks")
    #Load and index the datafile in batches
    for batch in range(0, range_upper, BATCH_SIZE):

        print(f"Processing index {batch}: {ts()}")
        points = read_bin(data_file, dtype, start_idx=batch, chunk_size=BATCH_SIZE)

        #get the centroids for all the points in the batch
        results = index.knnQueryBatch(points,k=1,num_threads=2)

        #group the points by centroid
        group_ids = {}
        group_points = {}
        for i in range(len(points)):
            point_id = batch+i
            point = points[i]
            key = results[i][0][0] #first id of the first tuple of the result
            if key not in group_ids:
                group_ids[key] = []
                group_points[key] = []
            group_ids[key].append(point_id)
            group_points[key].append(point)

        #add the points to the appropriate shards
        for key in group_ids.keys():
            add_points(path,key,group_ids[key],group_points[key])

        #assert len(list(group_ids.keys())) == len(points)

    print(f"Done! {ts()}")

index_dataset("data/shards/","data/bigann/learn.100M.u8bin",np.uint8)

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