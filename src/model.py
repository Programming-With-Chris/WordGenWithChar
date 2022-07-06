from gc import callbacks
from clean import *
from config import *

import tensorflow as tf
import numpy as np

SEQUENCE_SIZE, BATCH_SIZE, NUM_EPOCHS, MIN_WORD_FREQUENCY = get_configs()


def create_sequences(char_array):
    step = 3
    sequences = []
    next_char = []

    for i in range(0, len(char_array) - SEQUENCE_SIZE, step):
        sequences.append(char_array[i: i + SEQUENCE_SIZE])
        next_char.append(char_array[i + SEQUENCE_SIZE])

    return sequences, next_char


def vectorization(sequences_array, next_char_array, batch_size, char_vocab, char_indicies):
    index = 0
    while True:
        x = np.zeros((batch_size, SEQUENCE_SIZE, len(char_vocab)), dtype=np.bool)
        y = np.zeros((batch_size, len(char_vocab)), dtype=np.bool)
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


# TODO Double check it's char_array and not something else
def build_model(char_array):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, input_shape=(SEQUENCE_SIZE, len(char_array)), activation='relu', return_sequences=True))
    model.add(tf.keras.layers.LSTM(128, input_shape=(SEQUENCE_SIZE, len(char_array)), activation='relu'))
    model.add(tf.keras.layers.Dense(len(char_array), activation='softmax'))
    return model


def train_model(story_array, model_save_location): 
    char_array = stories_to_char_array(story_array)
    char_indices, indices_char, char_vocab = build_char_dictionaries(char_array)

    sequences, next_chars = create_sequences(char_array)

    sequences_train, next_chars_train, sequences_test, next_char_test = shuffle_and_split_training_set(sequences, next_chars)

    model = build_model(char_vocab) 
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
    callbacks_list = [early_stopping]

    model.fit(vectorization(sequences, next_chars, BATCH_SIZE, char_vocab, char_indices), 
              steps_per_epoch=int(len(sequences) / NUM_EPOCHS) + 1,
              epochs=NUM_EPOCHS,
              callbacks=callbacks_list,
              validation_data=vectorization(sequences_test, next_char_test, SEQUENCE_SIZE, char_vocab, char_indices), 
              validation_steps=int(len(sequences_test) / NUM_EPOCHS) + 1) 
    model.save(model_save_location) 
    return model


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def load_model(checkpoint_location):
    model = tf.keras.models.load_model(checkpoint_location)
    print(model.summary())
    return model


def generate_text(model, char_indicies, indicies_char, seed, sequence_length, diversity, quantity, vocabulary): 

    split_seed = seed.split()
    seed_chars = []
    for word in split_seed:
        for x in word:
            seed_chars.append(x)

    for i in range(quantity):
        x_pred = np.zeros((1, sequence_length, len(vocabulary)))
        for t, char in enumerate(seed_chars):
            if t < sequence_length:
                x_pred[0, t, char_indicies[char]] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indicies_char[next_index]

        seed_chars = seed_chars[1:]
        seed_chars.append(next_char)
 
        print(next_char)
        if i % 100 == 0:
            print("\n")
    print("\n")
