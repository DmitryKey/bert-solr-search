import numpy as np
from sklearn.cluster import KMeans


def find_closest_args(centroids, embeddings):
    """helper function to find the closest embeddings to the
    cluster centroids. Not required in the context of reindexing,
    since model.labels_ of kmeans gives us all the assignments
    """''
    centroid_min = 1e10
    cur_arg = -1
    args = {}
    used_idx = []
    for j, centroid in enumerate(centroids):
        for i, feature in enumerate(embeddings):
            value = np.linalg.norm(feature - centroid)
            if value < centroid_min and i not in used_idx:
                cur_arg = i
                centroid_min = value
        used_idx.append(cur_arg)
        args[j] = cur_arg
        centroid_min = 1e10
        cur_arg = -1
    return args


def kmeans_cluster(data_vectors, k: int = 5):
    """Performs kmeans clustering and returns the cluster
    allocation of the data vectors to the centroids

    Args:
        data_vectors ([type]): embedding vectors
        k (int, optional): Number of cluster centroids. Defaults to 5.

    Returns:
        [type]: [description]
    """
    model = KMeans(k).fit(data_vectors)
    data_size = len(data_vectors)
    labels_ = model.labels_
    label_dict = {i: labels_[i] for i in range(data_size)}
    sorted_dict = {k: v for k, v in sorted(label_dict.items(), key=lambda item: item[1])}
    return sorted_dict
