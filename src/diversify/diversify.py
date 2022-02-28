import random

import pandas as pd
import numpy as np

from abc import ABC, abstractmethod

from scipy.spatial.distance import cdist
from sklearn.preprocessing import MultiLabelBinarizer

from diversify.dpp import dpp
from diversify.kmeans_indices import kmeans_cluster

# Static random seed to ensure predictable randomization results
RANDOM_SEED = 57895447


def build_similarity_matrix(data, method="vec"):
    if method == "vec":
        dist = cdist(data, data, metric='cosine')
        dist = 1 - dist

    return dist


def diversify(data: list, method: str, categorical=None) -> list:
    """Diversify results using the given method


    Args:
        data (list): [description]
        method (str): [description]

    Returns:
        list: [description]
    """

    # Figure out vectors for categorical data
    if categorical:
        cat_list = [x.get(categorical) for x in data]
        mlb = MultiLabelBinarizer()
        mlb.fit(cat_list)
        vectors = mlb.transform(cat_list)
        for i in range(len(data)):
            data[i]["vector"] = vectors[i]

    data = pd.DataFrame(data)
    data['old_index'] = data.index


    if method == "random":
        div = RandomDiversify()
    elif method == "dpp":
        div = DppDiversify()
    elif method == "kmeans":
        div = KmeansDiversify()
    else:
        return data.to_dict('records')

    return div.diversify(data).to_dict('records')


class BaseDiversify(ABC):
    method = "vec"

    @abstractmethod
    def diversify(self, data):
        """Diversify the given input
        """
        pass

    @staticmethod
    def reorder_list(data: pd.DataFrame, indices: list):
        pass


class RandomDiversify(BaseDiversify):

    def diversify(self, data: pd.DataFrame):

        np.random.seed(seed=57895447)
        random.seed(57895447)

        df = data.sample(frac=1)

        return df


class DppDiversify(BaseDiversify):

    def diversify(self, data: pd.DataFrame):

        size = len(data)

        similarity_matrix = build_similarity_matrix(data["vector"].to_list())

        scores = data["_score"].to_numpy()
        kernel_matrix = scores.reshape((size, 1)) * similarity_matrix * scores.reshape((1, size))

        res = dpp(kernel_matrix, size)
        data = data.reindex(res)

        return data


class KmeansDiversify(BaseDiversify):

    def diversify(self, data: pd.DataFrame):

        size = len(data)
        num_clusters = size // 5
        data_vectors = data["vector"].to_list()
        res_dict = kmeans_cluster(data_vectors, num_clusters)
        res_indices = list(res_dict.keys())

        data = data.reindex(res_indices)

        return data
