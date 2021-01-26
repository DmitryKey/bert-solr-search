import bz2
from bert_serving.client import BertClient
from client.elastic_client import ElasticClient
import time
from data_utils import parse_and_index_data, SearchEngine

print("Initializing BERT and Elastic clients")
bc = BertClient()
ec = ElasticClient(configs_dir='es_conf')

# Read the input compressed file as is, without decompressing.
# Though disks are cheap, isn't it great to save them?
input_file = 'data/dbpedia/long_abstracts_en.ttl.bz2'
print("Reading BZ2 file")
source_file = bz2.BZ2File(input_file, "r")

if __name__ == '__main__':
    print("parsing and indexing abstracts with BERT based vectors...")
    start_time = time.time()
    index = "vector"
    ec.delete_index(index)
    ec.create_index(index)
    ec.index_documents(index, parse_and_index_data(source_file, bc, SearchEngine.ELASTICSEARCH))
    end_time = time.time()
    print("All done. Took: {} seconds".format(end_time-start_time))
