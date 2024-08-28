from util.utils import read_fbin, read_bin, get_total_nvecs_fbin
from numpy import linalg
from statistics import median
import numpy as np

# desired number of shards
M = 1000

# target maximum distance between points to fall inside a shard
DIST_MAX = 100

# number of times to sample the input dataset to approximate the dist_max
# SAMPLE_TIMES = 10
# size of the sample of points examined linearly during max dist computation
SAMPLE_SIZE = 1000

# batch size for reading points from the input file during the sharding algorithm
BATCH_SIZE = 1000000


# expected: 1 280 000 008
# file size: 1 280 0 000 008
def compute_max_dist(data_file: str, sample_size: int = SAMPLE_SIZE)->float:
    points = read_fbin(data_file, start_idx=0, chunk_size=sample_size)
    #points = read_bin(filename=data_file, dtype=np.float32, start_idx=0, chunk_size=sample_size)
    print(points.shape)
    num_rows, num_cols = points.shape
    print(num_rows)

    dists = np.sqrt(np.sum((points[None, :] - points[:, None])**2, -1))
    # dists = []
    # for i in range(0,num_rows):
    #    for j in range(1,num_rows-1):
    #        dist = linalg.norm(points[i]-points[j])
    #        dists.append(dist)

    print(dists)

    return median(dists.flatten())


def shard_by_dist(data_file: str, dist: float, shards_m: int = M):
    # set of integer order ids of each point that was already placed into a shard => processed
    processed_point_ids = set()
    complete_shards = 0

    total_num_elements = get_total_nvecs_fbin(data_file)
    print(f"Total number of points to process: {total_num_elements}")
    print(f"Reading data from {data_file} in {BATCH_SIZE} chunks")

    range_upper = total_num_elements
    print(f"range_upper={range_upper}")

    # map from shard id to number of elements
    shards = {}

    while complete_shards < M:
        complete_shards = len(shards.keys())
        for i in range(0, range_upper, BATCH_SIZE):
            print(f"Processing index={i}")
            points = read_fbin(data_file, start_idx=i, chunk_size=BATCH_SIZE)
            print(points.shape)
            print(points.shape[0])
            if i not in processed_point_ids:
                # fix the starting point
                first_point = points[0]
                # mark it visited
                processed_point_ids.add(i)
                # drop it from the input points
                points = np.delete(points, first_point, axis=0)
                print("going inside inner loop over j")
                print(points.shape[0])
                for j in range(0, points.shape[0]):
                    # id of the candidate is a combination of the running i-th batch and offset j within it
                    candidate_point_id = i + j
                    if candidate_point_id not in processed_point_ids:
                        dist_j = linalg.norm(first_point-points[j])
                        if dist_j <= dist:
                            processed_point_ids.add(candidate_point_id)

    print(len(processed_point_ids))


points_file = "data/big_ann/yandex/text2image-1b/query.learn.50M.fbin"

computed_dist_max = compute_max_dist(points_file)
print(f"computed {computed_dist_max}")

shard_by_dist(points_file, computed_dist_max)
