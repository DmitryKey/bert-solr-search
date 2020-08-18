def get_solr_vector(bert_client, query):
    return ','.join(str(elem) for elem in bert_client.encode([query]).flat)
