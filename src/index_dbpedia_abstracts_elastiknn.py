import bz2
from bert_serving.client import BertClient
from client.elastic_client import ElasticClient
import time
from data_utils import parse_dbpedia_data, SearchEngine, EmbeddingModel

print("Initializing BERT and Elastic clients")
bc = BertClient()
ec = ElasticClient(configs_dir='es_conf')

# Read the input compressed file as is, without decompressing.
# Though disks are cheap, isn't it great to save them?
input_file = 'data/dbpedia/long_abstracts_en.ttl.bz2'
print("Reading BZ2 file")
source_file = bz2.BZ2File(input_file, "r")

# Change this constant to vary the number of indexed abstracts
# set to -1 to index all
MAX_DOCS = 10000

if __name__ == '__main__':
    print("parsing and indexing abstracts with BERT based vectors...")
    start_time = time.time()
    index = "elastiknn"
    ec.delete_index(index)
    ec.create_index(index)
    ec.index_documents(index, parse_dbpedia_data(
        source_file,
        bc,
        EmbeddingModel.BERT_UNCASED_768,
        SearchEngine.ELASTICSEARCH,
        MAX_DOCS))
    end_time = time.time()
    print("All done. Took: {} seconds".format(end_time-start_time))
