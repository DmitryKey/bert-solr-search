import pickle
import re
import nltk
from enum import Enum

import numpy
from bert_serving.client import BertClient
from client.utils import to_solr_vector
from sklearn.preprocessing import normalize

VERBOSE = False


class SearchEngine(Enum):
    SOLR = 1
    ELASTICSEARCH = 2


def compute_vectors(text, bc):
    """
    Compute BERT embeddings for the input string
    :param text: single string with input text to compute BERT embedding for
    :param bc: BERT service object
    :return: encoded sentence/token-level embeddings, rows correspond to sentences
    :rtype: numpy.ndarray or list[list[float]]
    """
    return bc.encode([text])


def parse_dbpedia_data(source_file, bc: BertClient, search_engine: SearchEngine, max_docs: int):
    """
    Parses the input file of abstracts, computes BERT embeddings for each abstract,
    and indexes into the chosen search engine
    :param max_docs: maximum number of input documents to process; -1 for no limit
    :param source_file: input file
    :param bc: BERT service client
    :param search_engine: the desired search engine see {@code:SearchEngine} class
    :return: yields document by document to the consumer
    """
    global VERBOSE
    count = 0
    max_tokens = 0

    if -1 < max_docs < 50:
        VERBOSE = True

    ten_percent = 10 * max_docs / 100
    if ten_percent <= 0:
        ten_percent = 1000

    for line in source_file:
        line = line.decode("utf-8")

        # skip commented out lines
        comment_regex = '^#'
        if re.search(comment_regex, line):
            continue

        token_size = len(line.split())
        if token_size > max_tokens:
            max_tokens = token_size

        # skip lines with 20 tokens or less, because they tend to contain noise
        # (this may vary in your dataset)
        if token_size <= 20:
            continue

        first_url_regex = '^<([^\>]+)>\s*'

        x = re.search(first_url_regex, line)
        if x:
            url = x.group(1)
            # also remove the url from the string
            line = re.sub(first_url_regex, '', line)
        else:
            url = ''

        # remove the second url from the string: we don't need to capture it, because it is repetitive across
        # all abstracts
        second_url_regex = '^<[^\>]+>\s*'
        line = re.sub(second_url_regex, '', line)

        # remove some strange line ending, that occurs in many abstracts
        language_at_ending_regex = '@en \.\n$'
        line = re.sub(language_at_ending_regex, '', line)

        # compute vectors for a concatenated string of sentences
        vector = compute_vectors(' ||| '.join(nltk.sent_tokenize(line, "english")), bc)

        if search_engine == SearchEngine.SOLR:
            # convert BERT vector into Solr vector format
            vector = to_solr_vector(vector)
        elif search_engine == SearchEngine.ELASTICSEARCH:
            vector = vector.flatten()
        else:
            raise Exception("Unknown search engine: {}".format(search_engine))

        # form the input object for this abstract
        doc = {
            "vector": vector,
            "_text_": line,
            "url": url,
            "id": count+1
        }

        yield doc
        count += 1

        if count % ten_percent == 0:
            print("Processed and indexed {} documents".format(count))

        if count == max_docs:
            break

    source_file.close()
    print("Maximum tokens observed per abstract: {}".format(max_tokens))


def vectors_to_gis_files(source_file,
                         bc: BertClient,
                         search_engine: SearchEngine,
                         max_docs: int,
                         output_numpy_file,
                         output_pickle_file):
    """
    Reads the DBPedia data, computes embeddings and produces two files:
    one with normalized vectors (numpy array), the other with doc ids (pickle)
    :param max_docs:
    :param search_engine:
    :param bc:
    :param source_file:
    :param output_numpy_file: normalized vectors file
    :param output_pickle_file: doc ids pickle file
    """
    docs_iter = parse_dbpedia_data(source_file, bc, search_engine, max_docs)
    big_vector_arr = []
    big_docid_arr = []
    for doc in docs_iter:
        # flattened numpy array
        vector = doc['vector']
        big_vector_arr.append(vector)
        big_docid_arr.append(doc['id'])

    print(numpy.shape(big_vector_arr))
    # print(big_arr)
    # normalize and serialize to numpy binary format
    normalized_arr = normalize(big_vector_arr, norm='l2', axis=1)
    normalized_arr = normalized_arr.astype(numpy.float32)
    numpy.save(file=output_numpy_file, arr=normalized_arr)
    # serialize to pickle
    with open(output_pickle_file, 'wb') as pickleFile:
        string_list = [str(i) for i in big_docid_arr]
        pickle.dump(string_list, pickleFile)
    # docid_ndarray = numpy.array(big_docid_arr)
    # docid_ndarray.dump(output_pickle_file)
