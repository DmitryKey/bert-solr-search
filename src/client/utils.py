import numpy as np
from sklearn.preprocessing import normalize


def get_solr_vector_search(vectors):
    """
    Takes user keyword query, computes BERT embedding and returns a
    comma-separated string with vector points
    :param query: user keyword query
    :return: CSV string suitable for querying in Solr
    """
    return ','.join(str(elem) for elem in vectors.flat)


def to_solr_vector(vectors):
    """
    Takes BERT vectors array and converts into indexed representation: like so:
    1|point_1 2|point_2 ... n|point_n
    :param vectors: BERT vector points
    :return: Solr-friendly indexed representation targeted for indexing
    """
    solr_vector = []
    for vector in vectors:
        for i, point in enumerate(vector):
            solr_vector.append(str(i) + '|' + str(point))
    solr_vector = " ".join(solr_vector)
    return solr_vector


def myfmt(r):
   return "%.10f" % (r,)


def get_elasticsearch_vector(query_vector):
    query_vector = normalize(query_vector, norm='l2', axis=1)
    query_vector = np.round(query_vector.astype(np.float64), 10)
    return query_vector.flatten().tolist()

