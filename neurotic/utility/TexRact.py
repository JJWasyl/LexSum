import numpy as np
from keras.preprocessing.text import Tokenizer



def load_arrays(filename):
    file = open(filename, "rb")
    cases = np.load(file, allow_pickle=True)
    summaries = np.load(file, allow_pickle=True)
    file.close()
    return cases, summaries


def min_max_seq(sequences):
    max_seq_len = 0
    min_seq_len = 10e6

    for elem in sequences:
        if len(elem) > max_seq_len:
            max_seq_len = len(elem)

        if len(elem) < min_seq_len:
            min_seq_len = len(elem)

    return min_seq_len, max_seq_len


def tokenize(input_arrays, target_arrays, vocab_size=15000):
    # input_length = len(input_arrays)
    # target_length = len(target_arrays)
    # tokenized_inputs = np.empty(input_length, dtype=object)
    # tokenized_outputs = np.empty(target_length, dtype=object)

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(input_arrays + target_arrays)

    tokenized_inputs = tokenizer.texts_to_sequences(input_arrays)
    tokenized_outputs = tokenizer.texts_to_sequences(target_arrays)

    vocab = tokenizer.word_index
    x = {"PAD": 0}
    vocab.update(x)
    reverse_vocab = dict(zip(vocab.values(), vocab.keys()))
    print('Found %s unique tokens.' % len(vocab))

    return tokenized_inputs, tokenized_outputs, vocab, reverse_vocab


def reverse_tokenize(tokenized_array, reverse_vocab):
    result = []
    for item in tokenized_array:
        try:
            result.append(reverse_vocab[item])
        except KeyError:
            result.append("UNK")

    return result
