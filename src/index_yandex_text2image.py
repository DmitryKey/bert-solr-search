from util.utils import read_fbin
from index_hnswlib import build_hnswlib_index, load_index, MetricSpace
import time

data_file = "data/big_ann/yandex/text2image-1b/query.learn.50M.fbin"
BATCH_SIZE = 1000000
dim = 200
TOTAL_NUM_ELEMENTS = 50000000
DATASET = "yandex_text2image-1b"

print("Start of processing:")
print(f"TOTAL_NUM_ELEMENTS={TOTAL_NUM_ELEMENTS}")
print(f"BATCH_SIZE={BATCH_SIZE}")

start_time = time.time()

for i in range(0, TOTAL_NUM_ELEMENTS, BATCH_SIZE):
    print(f"Processing index={i}")
    print(f"Reading data from {data_file} in {BATCH_SIZE} chunks")
    vectors = read_fbin(data_file, start_idx=i, chunk_size=BATCH_SIZE)
    print("Done")
    print(vectors)

    if i == 0:
        print("Building HNSW index for the first batch")
        index_fname = build_hnswlib_index(dim=dim,
                            num_elements=TOTAL_NUM_ELEMENTS,
                            data=vectors,
                            data_labels=None,
                            space=MetricSpace.ip,
                            p=None,
                            filename_prefix=DATASET,
                            output_dir='data/hnswlib_index')
    else:
        print("Incremental update of HNSW index with data from next batch")
        # first load the index
        print(f"Loading index from {index_fname}")
        p = load_index(dim=dim, fname=index_fname, space=MetricSpace.ip, max_elements=TOTAL_NUM_ELEMENTS)
        # then update it
        print("Updating index")
        build_hnswlib_index(dim=dim,
                            num_elements=TOTAL_NUM_ELEMENTS,
                            data=vectors,
                            data_labels=None,
                            space=MetricSpace.ip,
                            p=p,
                            filename_prefix=DATASET,
                            output_dir='data/hnswlib_index')
        print(f"Iteration {i} complete")

end_time = time.time()
print("All done. Took: {} seconds".format(end_time-start_time))
