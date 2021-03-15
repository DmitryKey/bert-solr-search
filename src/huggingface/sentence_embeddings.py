from sentence_transformers import SentenceTransformer
import numpy

# expensive: downloads the model
model = SentenceTransformer('bert-base-nli-mean-tokens')
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.']
sentence_embeddings = model.encode(sentences)

print(numpy.shape(sentence_embeddings))

print("Sentence embeddings:")
print(sentence_embeddings)

