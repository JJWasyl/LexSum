import os.path
import numpy as np


def build_glove_embeddings(vocab_dict, embedding_dim, max_words=15000, local_dir=False):
    glove_dir = "/content/gdrive/My Drive/licencjat"
    if local_dir:
        glove_dir = "C:/Users/Father/Desktop/licencjat"

    # Embeddings dict
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    # word_index = {w: i for i, w in enumerate(embeddings_index.keys(), 1)}

    # Embedding matrix
    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in vocab_dict.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        if i == max_words - 1:
            break

    return embeddings_index, embedding_matrix

