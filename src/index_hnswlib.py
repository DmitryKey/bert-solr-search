import hnswlib
import numpy as np
from enum import Enum


class MetricSpace(Enum):
    l2 = 1,
    cosine = 2,
    ip = 3


def build_hnswlib_index(dim,
                        num_elements,
                        data,
                        data_labels,
                        space: MetricSpace,
                        p: hnswlib.Index,
                        filename_prefix='index',
                        output_dir='../data/hnswlib_index'):
    if p is None:
        # Declaring index
        print(f"Declaring index for metric {space.name} and {dim} dimensions")
        p = hnswlib.Index(space=str(space.name), dim=dim) # possible options are l2, cosine or ip

        # Initializing index - the maximum number of elements should be known beforehand
        #
        # M - the number of bi-directional links created for every new element during construction.
        # Reasonable range for M is 2-100.
        # Higher M work better on datasets with high intrinsic dimensionality and/or high recall, while
        # low M work better for datasets with low intrinsic dimensionality and/or low recalls.
        # The parameter also determines the algorithm's memory consumption, which is
        # roughly M * 8-10 bytes per stored element. This is 7.5G for 50M vectors.
        # As an example for dim=4 random vectors optimal M for search is somewhere around 6,
        # while for high dimensional datasets (word embeddings, good face descriptors),
        # higher M are required (e.g. M=48-64) for optimal performance at high recall.
        # The range M=12-48 is ok for the most of the use cases. When M is changed one has to update the other parameters.
        # Nonetheless, ef and ef_construction parameters can be roughly estimated by
        # assuming that M*ef_{construction} is a constant.
        #
        # ef_construction - the parameter has the same meaning as ef, but controls the index_time/index_accuracy.
        # Bigger ef_construction leads to longer construction, but better index quality.
        # At some point, increasing ef_construction does not improve the quality of the index.
        # One way to check if the selection of ef_construction was ok is to measure a recall for M nearest neighbor search
        # when ef =ef_construction: if the recall is lower than 0.9, than there is room for improvement.
        #
        # num_elements - defines the maximum number of elements in the index.
        # The index can be extened by saving/loading (load_index function has a parameter which
        # defines the new maximum number of elements).
        ef_construction = 50
        M = 16
        print(f"Initializing index for {num_elements} with ef_construction={ef_construction} M={M}")

        print("Index init")
        p.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)

    # Index parameters are exposed as class properties:
    print(f"Parameters passed to constructor:  space={p.space}, dim={p.dim}")
    print(f"Index construction: M={p.M}, ef_construction={p.ef_construction}")
    print(f"Index size is {p.element_count} and index capacity is {p.max_elements}")

    print(f"Got p={p}")
    print("---------------")
    # Element insertion (can be called several times):
    print("Indexing")
    p.add_items(data, data_labels)

    # Serializing and deleting the index:
    index_path = output_dir + '/' + filename_prefix + '_' + str(num_elements) + '_elements.bin'
    print("Saving index to '%s'" % index_path)
    p.save_index(index_path)
    del p
    return index_path


def load_index(dim: int, fname: str, space: MetricSpace, max_elements: int):
    p = hnswlib.Index(space=str(space.name), dim=dim)
    p.load_index(fname, max_elements)
    return p

def search(p, topK, data):
    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    print(f"Searching {topK} elements")
    # Controlling the recall by setting ef:
    ef = 50
    if ef <= topK:
        raise Exception(f"ef should always be > topK, but got: ef={ef}, topK={topK}")

    p.set_ef(ef) # ef should always be > topK
    print(f"Search speed/quality trade-off parameter: ef={p.ef}")
    labels, distances = p.knn_query(data, k=topK)
    print("Found labels:")
    print(labels[0])
    print("Their distances")
    print(distances[0])


if __name__ == '__main__':
    dim = 128
    num_elements = 1000000
    topK = 10

    # Generating sample data
    print(f"Generating sample data for {num_elements} vectors of {dim} dimensions")
    data = np.float32(np.random.random((num_elements, dim)))
    data_labels = np.arange(num_elements)

    build_hnswlib_index(dim=dim, num_elements=num_elements, data=data, data_labels=data_labels)
