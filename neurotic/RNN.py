import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed, RepeatVector, Activation
from tensorflow.python.keras.layers import concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from pickle import dump, load
from neurotic.utility import TexRact
import rouge


class KerasBatchGenerator(object):

    def __init__(self, x_data, y_data, doc_len, sum_len, batch_size, vocabulary_len):
        self.x_data = x_data
        self.y_data = y_data
        self.doc_len = doc_len
        self.sum_len = sum_len
        self.batch_size = batch_size
        self.vocabulary_len = vocabulary_len
        self.current_idx = 0

    def generate_one_shot(self):
        x = np.zeros((self.batch_size, self.doc_len))
        y = np.zeros((self.batch_size, self.sum_len, self.vocabulary_len))
        num_bacthes = len(self.x_data) // self.batch_size

        while True:
            for i in range(num_bacthes):
                start = i * self.batch_size
                stop = (i + 1) * self.batch_size
                if self.batch_size * self.current_idx + i >= len(self.x_data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[:, :] = self.x_data[start:stop]
                temp_y = self.y_data[start:stop]
                for line_idx, sequence in enumerate(temp_y):
                    y[line_idx, :, :] = to_categorical(sequence, num_classes=self.vocabulary_len, dtype=int)

                yield x, y

    def generate_rec_a(self):
        encoder_in = []
        decoder_in = []
        decoder_ta = []
        line_idx = 0
        while True:
            for case_idx in range(len(self.x_data)):
                target_words = self.y_data[case_idx]
                x = self.x_data[case_idx]
                decoder_input_line = []

                for idx in range(len(target_words) - 1):
                    decoder_input_line = decoder_input_line + [target_words[idx]]
                    decoder_target_label = np.zeros(self.vocabulary_len)
                    next_word = target_words[idx + 1]
                    if next_word != 0:
                        decoder_target_label[next_word] = 1
                    decoder_in.append(decoder_input_line)
                    encoder_in.append(x)
                    decoder_ta.append(decoder_target_label)
                    line_idx += 1
                    if line_idx >= self.batch_size:
                        yield [np.array(encoder_in), pad_sequences(decoder_in, self.sum_len)], np.array(decoder_ta)
                        line_idx = 0
                        encoder_in = []
                        decoder_in = []
                        decoder_ta = []


def build_one_shot_model(vocab_size, src_txt_length, sum_txt_length,
                         embedding_matrix, latent_dim=100,
                         mask_zero=True, use_glove=True):
    # vocab_size += 1
    # encoder
    encoder_inputs = Input(shape=(src_txt_length,), name="Encoder-Input")
    if use_glove:
        encoder1 = Embedding(vocab_size, latent_dim,
                             name="Body-Word-Embedding", weights=[embedding_matrix],
                             mask_zero=mask_zero, trainable=False
                             )(encoder_inputs)
    else:
        encoder1 = Embedding(vocab_size, latent_dim, name="Body-Word-Embedding",
                             mask_zero=mask_zero
                             )(encoder_inputs)

    encoder2 = LSTM(latent_dim, name="Encoder-LSTM")(encoder1)
    encoder3 = RepeatVector(sum_txt_length, name="Encoder-output")(encoder2)

    # decoder
    decoder1 = LSTM(latent_dim, return_sequences=True, name="Decoder-LSTM")(encoder3)
    decoder_outputs = TimeDistributed(Dense(vocab_size, name="Final-Output-Dense"))(decoder1)
    model_output = Activation('softmax')(decoder_outputs)

    model = tf.keras.Model(inputs=encoder_inputs, outputs=model_output)
    model.compile(loss='categorical_crossentropy', optimizer=tf.train.AdamOptimizer(1e-4),
                  metrics=['categorical_accuracy'])
    return model


def build_rec_model_a(vocab_size, src_txt_length, sum_txt_length,
                      embedding_matrix, latent_dim=100,
                      mask_zero=True, use_glove=True):

    # Source text input
    source_inputs = Input(shape=(src_txt_length,), name="Source-Input")
    if use_glove:
        source_layer_1 = Embedding(vocab_size, latent_dim,
                                   name="Source-Embedding", weights=[embedding_matrix],
                                   mask_zero=mask_zero, trainable=False
                                   )(source_inputs)
    else:
        source_layer_1 = Embedding(vocab_size, latent_dim,
                                   name="Source-Embedding", mask_zero=mask_zero
                                   )(source_inputs)

    source_layer_2 = LSTM(latent_dim, name="Source-LSTM")(source_layer_1)

    # Summary text input
    summary_inputs = Input(shape=(sum_txt_length,), name="Summary-Input")

    if use_glove:
        summary_layer_1 = Embedding(vocab_size, latent_dim,
                                    name="Summary-Embedding", weights=[embedding_matrix],
                                    mask_zero=mask_zero, trainable=False
                                    )(summary_inputs)
    else:
        summary_layer_1 = Embedding(vocab_size, latent_dim,
                                    name="Summary-Embedding", mask_zero=mask_zero
                                    )(summary_inputs)

    summary_layer_2 = LSTM(latent_dim, name="Summary-LSTM")(summary_layer_1)

    # Decoder output
    decoder_1 = concatenate([source_layer_2, summary_layer_2])
    decoder_outputs = Dense(vocab_size, activation='softmax')(decoder_1)

    model = Model(inputs=[source_inputs, summary_inputs], outputs=decoder_outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model


def save_model(model, history, filename):
    # serialize model to JSON
    model_json = model.to_json()
    with open(filename + ".json", "w") as json_file:
        json_file.write(model_json)

    # save the history
    with open(filename + "_history.pkl", "wb") as file_history:
        dump(history.history, file_history)

    # serialize weights to HDF5
    model.save_weights(filename + "_weights.h5")
    print("Saved model to disk")


def load_from_checkpoint(filename, google=True):
    if google:
        new_model = load_model("/content/gdrive/My Drive/LeXSum/histories/"+filename+".hdf5")
        return new_model
    else:
        new_model = load_model("models/" + filename + ".hdf5")
        return new_model


def load_model_arch(model_name, weights_name, history_name, load_history=False, google=True):
    if google:
        json_file = open("/content/gdrive/My Drive/LeXSum/histories/" + model_name + ".json", 'r')
    else:
        json_file = open("models/" + model_name + ".json", 'r')

    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    if google:
        loaded_model.load_weights("/content/gdrive/My Drive/LeXSum/histories/" + weights_name + ".h5")
    else:
        loaded_model.load_weights("models/" + weights_name + ".h5")

    if load_history:
        if google:
            with open("/content/gdrive/My Drive/LeXSum/histories/" + history_name + ".pkl", "rb") as file_history:
                loaded_history = load(file_history)
        else:
            with open("models/" + history_name + ".pkl", "rb") as file_history:
                loaded_history = load(file_history)

        print("Loaded model and history from disk")
        return loaded_model, loaded_history

    print("Loaded model from disk")
    return loaded_model


def summarize_rec_a(model, input_seq, target_text_len, reverse_vocab, verbose=True):
    wid_list = [0]
    sum_input_seq = pad_sequences([wid_list], target_text_len)
    terminated = False

    target_text = ''
    i = 0
    while not terminated:
        output_tokens = model.predict([input_seq, sum_input_seq])
        sample_token_idx = np.argmax(output_tokens[0, :])
        sample_word = reverse_vocab[sample_token_idx]
        wid_list = wid_list + [sample_token_idx]

        target_text += ' ' + sample_word

        if i >= target_text_len:
            terminated = True
        else:
            sum_input_seq = pad_sequences([wid_list], target_text_len)

        if verbose:
            print("\r", "Progress: {0} out of".format(i), target_text_len, end='')
        i += 1

    return target_text.strip()


def prepare_results(metric, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1',
                                                                 100.0 * f)


def rouge_test_internal(originals, targets):
    for aggregator in ['Avg', 'Best']:
        print('Evaluation with {}'.format(aggregator))
        apply_avg = aggregator == 'Avg'
        apply_best = aggregator == 'Best'

        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                                max_n=1,
                                limit_length=True,
                                length_limit=100,
                                length_limit_type='words',
                                apply_avg=apply_avg,
                                apply_best=apply_best,
                                alpha=0.5,  # Default F1_score
                                weight_factor=1.2,
                                stemming=True)

        scores = evaluator.get_scores(originals, targets)

        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
                print(prepare_results(metric, results['p'], results['r'], results['f']))

        print()


def execute_rouge_test(X_test, y_test, model, reverse_vocab):
    generated_summaries = []
    original_summaries = []

    for elem in X_test:
        summary = summarize_rec_a(model, np.reshape(elem, (1, len(elem))), 100, reverse_vocab, verbose=False)
        generated_summaries.append(" ".join(TexRact.reverse_tokenize(summary, reverse_vocab)))

    for elem in y_test:
        original_summaries.append(" ".join(TexRact.reverse_tokenize(elem, reverse_vocab)))

    rouge_test_internal(original_summaries, generated_summaries)