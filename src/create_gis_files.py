import bz2
from bert_serving.client import BertClient
from client.elastic_client import ElasticClient
import time
from data_utils import parse_dbpedia_data, SearchEngine, vectors_to_gis_files

print("Initializing BERT client")
bc = BertClient()

# Read the input compressed file as is, without decompressing.
# Though disks are cheap, isn't it great to save them?
input_file = 'data/dbpedia/long_abstracts_en.ttl.bz2'
print("Reading BZ2 file")
source_file = bz2.BZ2File(input_file, "r")

# Change this constant to vary the number of indexed abstracts
# set to -1 to index all
MAX_DOCS = 2
output_numpy_file = "data/gsi_apu/" + str(MAX_DOCS) + "_bert_vectors.npy"
output_pickle_file = "data/gsi_apu/" + str(MAX_DOCS) + "_bert_vectors_docids.pkl"

if __name__ == '__main__':
    print("parsing abstracts and computing BERT embeddings...")
    start_time = time.time()
    vectors_to_gis_files(
        source_file,
        bc,
        SearchEngine.ELASTICSEARCH,
        MAX_DOCS,
        output_numpy_file,
        output_pickle_file
    )
    end_time = time.time()
    print("All done. Took: {} seconds".format(end_time-start_time))
