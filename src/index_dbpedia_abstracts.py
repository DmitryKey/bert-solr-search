import bz2
import re
import nltk
from bert_serving.client import BertClient
from client.solr_client import SolrClient
import time


bc = BertClient()
sc = SolrClient()

source_file = bz2.BZ2File("data/dbpedia/long_abstracts_en.ttl.bz2", "r")

VERBOSE = False


def compute_vectors(text):
    return bc.encode(text)


def to_solr_vector(vectors):
    solr_vector = []
    for vector in vectors:
        for i, point in enumerate(vector):
            solr_vector.append(str(i) + '|' + str(point))
    solr_vector = " ".join(solr_vector)
    return solr_vector


def parse_and_index_data():

    global VERBOSE

    count = 0
    # change this constant to vary the number of indexed abstracts
    # set to -1 to index all
    max_docs = 1000

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

        # skip lines with 20 tokens or less
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
        vectors = compute_vectors([' ||| '.join(nltk.sent_tokenize(line, "english"))])

        # convert BERT vector into Solr vector format
        solr_vector = to_solr_vector(vectors)

        # form the Solr input object for this abstract
        solr_doc = {
            "vector": solr_vector,
            "_text_": line,
            "url": url
        }

        if VERBOSE:
            print(solr_doc)

        yield solr_doc
        count += 1

        if count % ten_percent == 0:
            print("Processed and indexed {} documents".format(count))

        if count == max_docs:
            break

    source_file.close()
    print("Maximum tokens observed per abstract: {}".format(max_tokens))


if __name__ == '__main__':
    print("parsing and indexing abstracts with solr BERT based vectors...")
    start_time = time.time()
    sc.index_documents("vector-search", parse_and_index_data())
    end_time = time.time()
    print("All done. Took: {} seconds".format(end_time-start_time))
