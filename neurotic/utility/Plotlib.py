import matplotlib.pyplot as plt
import numpy as np


def plot_word_chart(vocab):
    plt.subplot(1, 1, 1)
    plt.title("Word frequency")
    plt.ylabel("Total number of occurrences")
    plt.xlabel("Word rank")
    plt.plot(list(range(len(vocab))), sorted(list(vocab.values()), reverse=True), '-', color='blue', lw=2)
    plt.xscale('log')
    plt.ylim(0, len(vocab)*1.1)
    plt.show()


def create_bins(input_arr, bin_count, maxlen):
    multiple = maxlen / bin_count
    binlist = np.empty(bin_count, dtype=int)
    for elem in input_arr:
        it = 0
        temp = len(elem) - multiple
        while temp > 0:
            temp = round(temp - multiple)
            it += 1

        binlist[it] += 1
    return binlist


def plot_length_chart(input_arr, title, bin_count=100, maxlen=260000):
    plt.subplot(1, 1, 1)
    plt.title(title)
    plt.ylabel("cases")
    plt.xlabel("length")
    array_lengths = [len(elem) for elem in input_arr]
    #case_count = create_bins(input_arr, bin_count, maxlen)
    array_lengths = sorted(array_lengths)
    plt.hist(array_lengths, bins=bin_count)
    plt.show()


def plot_history(history, name):
    loss_values = history["loss"]
    val_loss_values = history["val_loss"]
    acc_values = history["categorical_accuracy"]
    val_acc_values = history["val_categorical_accuracy"]
    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, 'bo', label="Training loss")
    plt.plot(epochs, val_loss_values, 'b', label=" Validation loss")
    plt.title("Training loss for " + name)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.clf()
    plt.plot(epochs, acc_values, 'bo', label="Training accuracy")
    plt.plot(epochs, val_acc_values, 'b', label="Validation accuracy")
    plt.title("Training accuracy for " + name)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_accuracy(accuracy, val_accuracy, name):
    epochs = range(1, len(accuracy) + 1)
    plt.clf()
    plt.plot(epochs, accuracy, 'bo', label="Training accuracy")
    plt.plot(epochs, val_accuracy, 'b', label=" Validation accuracy")
    plt.title("Training accuracy for " + name)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def plot_loss(loss, val_loss, name):
    epochs = range(1, len(loss) + 1)
    plt.clf()
    plt.plot(epochs, loss, 'bo', label="Training loss")
    plt.plot(epochs, val_loss, 'b', label=" Validation loss")
    plt.title("Training loss for " + name)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()