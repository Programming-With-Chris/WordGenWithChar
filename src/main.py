from file import *
from clean import *
from model import *

import random as rand


if __name__ == '__main__':
    text_array = FileWork().read_by_author("Edgar Allan Poe")
    char_array = stories_to_char_array(text_array)

    sequences = create_sequences(char_array)

    char_indicies, indicies_char, char_vocab = build_char_dictionaries(char_array)

    seed_starting_pos = rand.randint(0, len(char_array))
    seed = char_array[seed_starting_pos: seed_starting_pos + 10000]
    
    model = train_model(text_array, "test_model_location.h5")
    generate_text(model, char_indicies, indicies_char, seed, SEQUENCE_SIZE, 1.0, 250, char_vocab)
    
    
