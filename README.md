Bert with Solr and Elasticsearch
===

This code is described in two Medium stories: 

[Neural Search with BERT and Solr](https://medium.com/@dmitry.kan/neural-search-with-bert-and-solr-ea5ead060b28)

[Fun with Apache Lucene and BERT Embeddings](https://medium.com/swlh/fun-with-apache-lucene-and-bert-embeddings-c2c496baa559)

![Bert in Solr hat](img/bert.png)

Tech stack: 
- bert-as-service
- solr / elasticsearch
- streamlit
- Python 3

Code for dealing with Solr has been copied from the great (and highly recommended) https://github.com/o19s/hello-ltr project.

# Install tensorflow

`pip install tensorflow==1.15.3`

If you try to install tensorflow 2.3, bert service will fail to start, there is an existing issue about it.

# Let's install bert-as-service components

`pip install bert-serving-server`

`pip install bert-serving-client`    

# Download a pre-trained BERT model 
into the bert-model/ directory in this project. I have chosen uncased_L-12_H-768_A-12.zip
for this experiment. Unzip it.

# Now let's start the BERT service

`bash start_bert_server.sh`

# Run a sample bert client
 to compute vectors for 3 sentences:

        Bert vectors for sentences ['First do it', 'then do it right', 'then do it better'] : [[ 0.13186474  0.32404128 -0.82704437 ... -0.3711958  -0.39250174
          -0.31721866]
         [ 0.24873531 -0.12334424 -0.38933852 ... -0.44756213 -0.5591355
          -0.11345179]
         [ 0.28627345 -0.18580122 -0.30906814 ... -0.2959366  -0.39310536
           0.07640187]]

This sets up the stage for our further experiment with Solr.

# Dataset
 This is by far the key ingredient of every experiment. You want to find an interesting
collection of texts, that are suitable for semantic level search. Well, maybe all texts are. I have chosen a collection of abstracts from DBPedia,
that I downloaded from here: https://wiki.dbpedia.org/dbpedia-version-2016-04 and placed into data/ directory in bz2 format.
You don't need to extract this file onto disk: our code will read directly from the compressed file.

# Preprocessing and Indexing
Let's preprocess the downloaded abstracts, and index them in Solr. First, execute the following command to start Solr:

    bin/solr start -m 2g
    
If during processing you will notice:

    <...>/bert-solr-search/venv/lib/python3.7/site-packages/bert_serving/client/__init__.py:299: UserWarning: some of your sentences have more tokens than "max_seq_len=500" set on the server, as consequence you may get less-accurate or truncated embeddings.
    here is what you can do:
    - disable the length-check by create a new "BertClient(check_length=False)" when you do not want to display this warning
    - or, start a new server with a larger "max_seq_len"
      '- or, start a new server with a larger "max_seq_len"' % self.length_limit)


The same index_dbpedia_abstracts.py code will output statistics:


    Maximum tokens observed per abstract: 697
    Flushing 100 docs
    Committing changes
    All done. Took: 82.46466588973999 seconds
    
We know how many abstracts there are:    
    
    bzcat data/dbpedia/long_abstracts_en.ttl.bz2 | wc -l
    5045733
    
Running the BERT search demo
===
There are two streamlit demos for running BERT search
for Solr and Elasticsearch. Each demo compares to BM25 based search.

To run a demo, execute the following on the command line from the project root:

    # for experiments with Elasticsearch
    streamlit run src/search_demo_elasticsearch.py

    # for experiments with Solr
    streamlit run src/search_demo_solr.py