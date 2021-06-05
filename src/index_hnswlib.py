import hnswlib
import numpy as np

dim = 128
num_elements = 1000000
topK = 10

# Generating sample data
print(f"Generating sample data for {num_elements} vectors of {dim} dimensions")
data = np.float32(np.random.random((num_elements, dim)))
data_labels = np.arange(num_elements)

# Declaring index
p = hnswlib.Index(space='l2', dim=dim) # possible options are l2, cosine or ip

# Initializing index - the maximum number of elements should be known beforehand
p.init_index(max_elements=num_elements, ef_construction=200, M=16)

# Element insertion (can be called several times):
print("Indexing")
p.add_items(data, data_labels)

# Controlling the recall by setting ef:
p.set_ef(50) # ef should always be > k

# Query dataset, k - number of closest elements (returns 2 numpy arrays)
print(f"Searching {topK} elements")
labels, distances = p.knn_query(data, k=topK)
print("Found labels:")
print(labels[0])
print("Their distances")
print(distances[0])

# Index parameters are exposed as class properties:
print(f"Parameters passed to constructor:  space={p.space}, dim={p.dim}")
print(f"Index construction: M={p.M}, ef_construction={p.ef_construction}")
print(f"Index size is {p.element_count} and index capacity is {p.max_elements}")
print(f"Search speed/quality trade-off parameter: ef={p.ef}")

# Serializing and deleting the index:
index_path = '../data/hnswlib_index/index_' + str(num_elements) + '_elements.bin'
print("Saving index to '%s'" % index_path)
p.save_index(index_path)
del p