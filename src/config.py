MIN_WORD_FREQUENCY = 5
maxlen = 25
BATCH_SIZE = 20
num_epochs = 10


def get_configs():
    return maxlen, BATCH_SIZE, num_epochs, MIN_WORD_FREQUENCY