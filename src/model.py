from clean import *
from config import *

import tensorflow as tf
import numpy as np

sequence_size, BATCH_SIZE, num_epochs, MIN_WORD_FREQUENCY = get_configs()


def create_sequences(char_array):
    step = 3
    sequences = []
    next_char = []

    for i in range(0, len(char_array) - sequence_size, step):
        sequences.append(char_array[i: i + sequence_size])
        next_char.append(char_array[i + sequence_size])

    return sequences, next_char


def vectorization(sequences_array, next_char_array, batch_size, char_array, char_indicies):
    index = 0
    while True:
        x = np.zeros((batch_size, sequence_size, len(char_array)), dtype=np.bool)
        y = np.zeros((batch_size, len(char_array)), dtype=np.bool)
        for i in range(batch_size):
            for t, c in enumerate(sequences_array[index]):
                x[i, t, char_indicies[c]] = 1
            y[i, char_indicies[next_char_array[index]]] = 1

            index = index + 1
            if index == len(sequences_array):
                index = 0
        yield x, y


def shuffle_and_split_training_set(sequences_array, next_char_array, percentage_test=3):
    tmp_sequences = []
    tmp_next_char = []
    for i in np.random.permutation(len(sequences_array)):
        tmp_sequences.append(sequences_array[i])
        tmp_next_char.append(next_char_array[i])
    cut_index = int(len(sequences_array) * (1. - (percentage_test / 100.)))
    x_train, x_test = tmp_sequences[:cut_index], tmp_sequences[cut_index:]
    y_train, y_test = tmp_next_char[:cut_index], tmp_next_char[cut_index:]

    print("Training set = %d\nTest set = %d" % (len(x_train), len(y_test)))
    return x_train, y_train, x_test, y_test


# Double check it's char_array and not something else
def build_model(char_array):
    model = tf.keras.Sequential()
    model.add(tf.keras.layer.LSTM(128, input_shape=(sequence_size, len(char_array)), activation='relu', return_sequences=True))
    model.add(tf.keras.layer.LSTM(128, input_shape=(sequence_size, len(char_array)), activation='relu'))
    model.add(tf.keras.layer.Dense(len(char_array), activation='softmax'))
    return model


