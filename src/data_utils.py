import re
import nltk
from enum import Enum
from client.utils import to_solr_vector

VERBOSE = False


class SearchEngine(Enum):
    SOLR = 1
    ELASTICSEARCH = 2


def compute_vectors(text, bc):
    return bc.encode([text])


def parse_data(source_file, bc, search_engine: SearchEngine, max_docs):
    """
    Parses the input file of abstracts, computes BERT embeddings for each abstract,
    and indexes into the chosen search engine
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
