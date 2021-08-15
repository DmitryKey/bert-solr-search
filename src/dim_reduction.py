import numpy as np

def read_bin(filename, dtype, start_idx=0, chunk_size=None):
    with open(filename, "rb") as f:
        # The header is two np.int32 values
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = 100000000
        print(f"number of vectors: {nvecs}")
        print(f"dimensions: {dim}")
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size

        if dtype == np.uint8:
            type_multiplier = 1
        elif dtype == np.float32:
            type_multiplier = 4

        arr = np.fromfile(f, count=nvecs * dim, dtype=dtype, offset=start_idx * dim * type_multiplier)
    return arr.reshape(-1, dim)


points = read_bin("/Users/dmitry/Desktop/BigANN/dimensionality_reduction/learn.100M_compressed_32.fbin", np.float32, 0, 10)
print(points)
