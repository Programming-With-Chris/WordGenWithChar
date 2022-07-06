

def stories_to_char_array(story_array):
    char_array = []
    for story in story_array:
        ## any character replaces needed?
        split_story = story.split()
        for word in split_story:
            char_list = list(word)
            for x in word:
                char_array.append(x)
    return char_array


def build_char_dictionaries(char_array):

    char_vocab = sorted(set(char_array))

    char_indices = dict((c, i) for i, c in enumerate(char_vocab))
    indices_char = dict((i, c) for i, c in enumerate(char_vocab))

    return char_indices, indices_char, char_vocab
