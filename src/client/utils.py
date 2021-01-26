def get_solr_vector_search(bert_client, query):
    """
    Takes user keyword query, computes BERT embedding and returns a
    comma-separated string with vector points
    :param bert_client: used for computing BERT embedding
    :param query: user keyword query
    :return: CSV string suitable for querying in Solr
    """
    return ','.join(str(elem) for elem in bert_client.encode([query]).flat)


def to_solr_vector(vectors):
    """
    Takes BERT vectors array and converts into indexed representation: like so:
    1|point_1 2|point_2 ... n|point_n
    :param vectors: BERT vector points
    :return: Solr-friendly indexed representation targeted for indexing
    """
    solr_vector = []
    for vector in vectors:
        for i, point in enumerate(vector):
            solr_vector.append(str(i) + '|' + str(point))
    solr_vector = " ".join(solr_vector)
    return solr_vector


def get_elasticsearch_vector(bert_client, query):
    """
    Compute the BERT embedding of the given query and return an array of vector values
    :param bert_client: BERT client to the bert-as-service
    :param query: user query
    :return: BERT embedding array
    """
    return bert_client.encode([query]).flatten('F')
