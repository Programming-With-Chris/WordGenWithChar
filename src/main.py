from file import *
from clean import *
from model import *


if __name__ == '__main__':
    text_array = FileWork().read_by_author("Edgar Allan Poe")
    char_array = stories_to_char_array(text_array)

    sequences = create_sequences(char_array)
