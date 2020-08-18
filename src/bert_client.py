from bert_serving.client import BertClient

bc = BertClient()

sentences = ['First do it', 'then do it right', 'then do it better']
vectors = bc.encode(sentences)

print("Bert vectors for sentences {} : {}".format(sentences, vectors))


query = ['mathematics']
query_vec_str = ','.join(str(elem) for elem in bc.encode(query).flat)
print("query: '{}' bert vectors: {}".format(query, query_vec_str))

