
BERT models with Solr and Elasticsearch
===

https://user-images.githubusercontent.com/123553/118112866-3f44e480-b3ee-11eb-92ca-0250bc65bd37.mp4

https://user-images.githubusercontent.com/123553/118320272-a94fad80-b504-11eb-96cf-1810813e2284.mp4


This code is described in the following Medium stories, taking one step at a time: 

[Neural Search with BERT and Solr](https://medium.com/@dmitry.kan/neural-search-with-bert-and-solr-ea5ead060b28) (August 18,2020)

[Fun with Apache Lucene and BERT Embeddings](https://medium.com/swlh/fun-with-apache-lucene-and-bert-embeddings-c2c496baa559) (November 15, 2020)

[Speeding up BERT Search in Elasticsearch](https://dmitry-kan.medium.com/speeding-up-bert-search-in-elasticsearch-750f1f34f455) (March 15, 2021)

[Ask Me Anything about Vector Search](https://dmitry-kan.medium.com/ask-me-anything-about-vector-search-4252a01f3889) (June 20, 2021) This blog post gives the answers to the 3 most interesting questions asked during the AMA session at Berlin Buzzwords 2021. The video recording is available here: https://www.youtube.com/watch?v=blFe2yOD1WA

[5 ways to increase result diversity at web-scale (Video)](https://dmitry-kan.medium.com/5-ways-to-increase-result-diversity-at-web-scale-4b4f79a898a6) Implementation of result diversity algorithms on top of neural search using code in this repository. [Streamlit demo](https://github.com/DmitryKey/bert-solr-search/blob/master/src/search_demo_elasticsearch.py)

Also, if you are interested in Vector Databases and Neural Search Frameworks, these two blog posts provide more information:

[Not All Vector Databases Are Made Equal
](https://towardsdatascience.com/milvus-pinecone-vespa-weaviate-vald-gsi-what-unites-these-buzz-words-and-what-makes-each-9c65a3bd0696) A detailed comparison of Milvus, Pinecone, Vespa, Weaviate, Vald, GSI and Qdrant

[Neural Search Frameworks: A Head-to-Head Comparison](https://dmitry-kan.medium.com/neural-search-frameworks-a-head-to-head-comparison-976aa6662d20) This blog post introduces my Vector Search Pyramid concept, through which I explain vector search space layer by layer. This blog post focuses on neural search framework layer.

![Bert in Solr hat](img/bert_solr.png)
![Bert with_es burger](img/bert_es.png)

---

Tech stack:
- Hugging Face
- Solr / Elasticsearch / ODFE (OpenSearch)
- Solr / Elasticsearch / OpenSearch
- streamlit
- Python 3.8 (upgraded recently)

Code for dealing with Solr and Elasticsearch has been copied from the great (and highly recommended) https://github.com/o19s/hello-ltr project.
OpenSearch client is implemented on top of this code and https://github.com/DmitryKey/search_with_machine_learning_course/blob/main/week1/opensearch.py

# How to install
If you encounter issues with the above installation, consider installing full list of packages:

`pip install -r requirements_freeze.txt`

# Let's install bert-as-service components

`pip install bert-serving-server`

`pip install bert-serving-client`    

# Download a pre-trained BERT model 
into the `bert-model/` directory in this project. I have chosen [uncased_L-12_H-768_A-12.zip](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)
for this experiment. Unzip it.

# Now let's start the BERT service

`bash start_bert_server.sh`

# Run a sample bert client
    python src/bert_client.py
 to compute vectors for 3 sample sentences:

        Bert vectors for sentences ['First do it', 'then do it right', 'then do it better'] : [[ 0.13186474  0.32404128 -0.82704437 ... -0.3711958  -0.39250174
          -0.31721866]
         [ 0.24873531 -0.12334424 -0.38933852 ... -0.44756213 -0.5591355
          -0.11345179]
         [ 0.28627345 -0.18580122 -0.30906814 ... -0.2959366  -0.39310536
           0.07640187]]

The steps so far set up the stage for our further experiment with indexing in the preferred search engine.

# Dataset
This is by far the key ingredient of every experiment. You want to find an interesting
collection of texts, that are suitable for semantic level search. Well, maybe all texts are. I have chosen a collection of abstracts from DBPedia,
which I downloaded from here: https://wiki.dbpedia.org/dbpedia-version-2016-04 and placed into `data/dbpedia` directory in bz2 format.
You don't need to extract this file onto disk: the provided code will read directly from the compressed file.

# Data preprocessing and Indexing in Solr
Before running preprocessing / indexing, you need to configure the vector plugin, which allows indexing and querying the vector data.
You can find the plugin for Solr 8.x here: https://github.com/DmitryKey/solr-vector-scoring/releases

After the plugin's jar has been added, configure it in the solrconfig.xml like so:

    <queryParser name="vp" class="com.github.saaay71.solr.VectorQParserPlugin" />

Schema also requires an addition: field of type `VectorField` is required in order to index vector data:


    <field name="vector" type="VectorField" indexed="true" termOffsets="true" stored="true" termPositions="true" termVectors="true" multiValued="true"/>

Find ready-made schema and solrconfig here: https://github.com/DmitryKey/bert-solr-search/tree/master/solr_conf

Let's preprocess the downloaded abstracts, and index them in Solr. First, execute the following command to start Solr:

    bin/solr start -m 2g
    
If during processing you will notice:

    <...>/bert-solr-search/venv/lib/python3.7/site-packages/bert_serving/client/__init__.py:299: UserWarning: some of your sentences have more tokens than "max_seq_len=500" set on the server, as consequence you may get less-accurate or truncated embeddings.
    here is what you can do:
    - disable the length-check by create a new "BertClient(check_length=False)" when you do not want to display this warning
    - or, start a new server with a larger "max_seq_len"
      '- or, start a new server with a larger "max_seq_len"' % self.length_limit)


The `index_dbpedia_abstracts_solr.py` script will output statistics:


    Maximum tokens observed per abstract: 697
    Flushing 100 docs
    Committing changes
    All done. Took: 82.46466588973999 seconds
    
We know how many abstracts there are:    
    
    bzcat data/dbpedia/long_abstracts_en.ttl.bz2 | wc -l
    5045733
    
# Data preprocessing and Indexing in Elasticsearch
This project implements several ways to index vector data:
* `src/index_dbpedia_abstracts_elastic.py` vanilla Elasticsearch: using `dense_vector` data type
* `src/index_dbpedia_abstracts_elastiknn.py` elastiknn plugin: implements own data type. I used `elastiknn_dense_float_vector`

Each indexer relies on ready-made Elasticsearch mapping file, that can be found in `es_conf/` directory:
* es_conf/vector_settings.json is used for vanilla vector search
* es_conf/elastiknn_settings.json is used for KNN vector search implemented with [elastiknn plugin](https://github.com/alexklibisz/elastiknn).

To configure elastiknn, please refer to its excellent documentation.

# Data preprocessing and Indexing in ODFE (Open Distro for Elasticsearch)

* `src/index_dbpedia_abstracts_opendistro.py` ODFE: uses nmslib to build Hierarchical Navigable Small World (HNSW) graphs during indexing

It is important to understand, that unlike vanilla or elastiknn implementation (Java), nmslib implements HNSW graphs in C++ and therefore
ODFE will use off-heap memory to build this data structure. In order to achieve optimal indexing and search performance, you need to consider
the following hyper-parameters:

* number of shards and number of replicas: https://opendistro.github.io/for-elasticsearch-docs/docs/elasticsearch/#primary-and-replica-shards
* KNN space type: `cosinesimil`, `hammingbit`, `l1`, `l2`
* refresh interval
* number of segments in the Lucene index
* circuit_breaker_limit -- cluster level setting, controlling the portion of RAM used for off-heap graphs

The recommended formula for computing RAM used for storing the graphs:

    RAM(vector_dimension) = 1.1 * (4 * vector_dimension + 8 * M) bytes / vector

In this project we compute vectors with 768 dimensions. For 1M vectors and M=16 we will need:

   RAM(768) = 1.1 * (4 * 768 + 8 * 16) * 1,000,000 ~= 3.28 GB

Replicas will double the amount of RAM needed for your cluster.

# Data preprocessing and Indexing in GSI APU
In order to use GSI APU solution, a user needs to produce two files:
numpy 2D array with vectors of desired dimension (768 in my case)
a pickle file with document ids matching the document ids of the said vectors in Elasticsearch.

After these data files get uploaded to the GSI server, the same data gets indexed in Elasticsearch. The APU powered search is performed on up to 3 Leda-G PCIe APU boards.
Since I’ve run into indexing performance with bert-as-service solution, 
I decided to take SBERT approach from Hugging Face to prepare the numpy and pickle array files. 
This allowed me to index into Elasticsearch freely at any time, without waiting for days.
You can use this script to do this on DBPedia data, which allows choosing between:

    EmbeddingModel.HUGGING_FACE_SENTENCE (SBERT)
    EmbeddingModel.BERT_UNCASED_768 (bert-as-service)

To generate the numpy and pickle files, use the following script: `scr/create_gsi_files.py`.
This script produces two files:

    data/1000000_EmbeddingModel.HUGGING_FACE_SENTENCE_vectors.npy
    data/1000000_EmbeddingModel.HUGGING_FACE_SENTENCE_vectors_docids.pkl

Both files are perfectly suitable for indexing with Solr and Elasticsearch.

To test the GSI plugin, you will need to upload these files to GSI server for loading them both to Elasticsearch and APU.

Running the BERT search demo
===
There are two streamlit demos for running BERT search
for Solr and Elasticsearch. Each demo compares to BM25 based search.
The following assumes that you have bert-as-service up and running (if not, launch it with `bash start_bert_server.sh`)
and either Elasticsearch or Solr running with the index containing field with embeddings.

To run a demo, execute the following on the command line from the project root:

    # for experiments with Elasticsearch
    streamlit run src/search_demo_elasticsearch.py

    # for experiments with Solr
    streamlit run src/search_demo_solr.py
