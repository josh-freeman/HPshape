""" a few utilitary functions for preprocessing"""
import re
import string
from collections import Counter
import spacy

import numpy as np

from util.constants import F


def all_in_one_line(path: str):
    """
    replace doc with a single line.
    :param path:
    :return:
    """
    with open(path, "rw", encoding="utf8") as doc:
        txt = doc.read()
        doc.write(txt.replace("\n", " "))
        doc.close()


def remove_page_lines_hp(path: str):
    """
    specific to harry potter books
    :param path: the path to a given book
    :return:
    """
    with open(path, "r+", encoding="utf8") as doc:
        txt = doc.read()
        doc.seek(0)
        doc.write(re.sub("Page \|.*Rowling", repl="", string=txt))
        doc.truncate()
        doc.close()


def remove_consecutive_blank_lines(path: str):
    """

    :param path: the path to a given book
    :return:
    """
    with open(path, "r+", encoding="utf8") as doc:
        txt = doc.read()
        doc.seek(0)
        doc.write(re.sub("^\s+$", repl="", string=txt))
        doc.truncate()
        doc.close()


def remove_punctuation(text: str):
    return text.translate(str.maketrans('', '', string.punctuation + '’'))


def lemmatize(text: str):
    """

    :param text: a text *without punctuation and \n*.
    :return: list of lemmas
    """
    assert (not any(p in text for p in string.punctuation))

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    nlp.max_length = 2000000  # allocate 2 MB of RAM for the lemmatizer
    doc = nlp(text)
    return [token.lemma_ for token in doc]


def context_around_index(i, tokenized_word_list, c):
    return tokenized_word_list[i - c:i] + tokenized_word_list[
                                          i + 1:i + c + 1]  # In list[first:last], last is not included.


def x_and_ys_list_from(tokenized_word_list: list, c: int):
    """
    from a tokenized word list, return a list of contexts of size 2c+1, starting with the word at index c, ending at index -c.
    :param tokenized_word_list:
    :param c:
    :return:A list of contexts, represented as (word (string), list of surrounding words (string))
    """
    ret = []  # a list of tuples
    for i in range(c, len(tokenized_word_list) - 1 - c):
        word = tokenized_word_list[i]
        context = context_around_index(i, tokenized_word_list, c)
        ret.append((word, context))
    return ret


def vocab_from(x_and_ys_list):
    """
    Establish a list of all words at the center of c contexts
    :param x_and_ys_list: list[(array(shape=(v,1),
    array(shape=(v,1)))]
    :return: an ordered SET (list with no doubles)
    """
    x, _ = zip(*x_and_ys_list)
    cnt = Counter(x)
    from util.constants import MIN_WORD_THRESHOLD
    return [k for k, v in cnt.items() if v >= MIN_WORD_THRESHOLD]


def pre_proc(path: str, c: int, vocab: list = None, training=True) -> (list[str], list[tuple[np.ndarray, np.ndarray]]):
    """
    :param training: whether to use the lower 90 % of the string (true) or upper 10 % (false)
    :param vocab:
    :param c: the window size
    :param path: the absolute path to the text
    :return: a tuple of
    - first, a vocab of size v:=size(vocab). It is represented by a list of all words of interest found in the text. If
      vocab is not None, the argument vocab will simply be returned as such.

    - second, corresponding samples, as a one-hot word, and its context (as a sum of the one-hot vectors of the
    words it comprises).
    """
    if path is None:
        return vocab, []  # check for empty path.

    remove_page_lines_hp(path)
    remove_consecutive_blank_lines(path)
    with open(path, encoding='utf8') as data:
        text = data.read()
        assert len(text) != 0
        text = text[:-len(text) // F] if training else text[-len(text) // F:]  # slice text as needed for train/eval
        text = remove_punctuation(text.strip().lower())  # remove punctuation and lower.
        text = re.sub('\s+|[^a-zA-Z]', ' ',
                      text)  # remove whitespace and anything remaining that is not an English letter
        tokenized_word_list = lemmatize(text)  # list of lemmas.

        x_and_ys_list = x_and_ys_list_from(tokenized_word_list, c)  # make a first list of tuples from the tokens.
        vocab = list(set(vocab_from(x_and_ys_list))) if vocab is None else vocab  # Establish a list of all words at the
        # center of c contexts *if no value is provided*

        vocab = np.array(vocab)  # as np array for optimization!

        x_and_ys_list = [(x, ys) for (x, ys) in x_and_ys_list if
                         x in vocab and all(y in vocab for y in ys)]  # only keep tuples containing words in vocab

        def __one_hot(word) -> np.array:
            """
            return one hot version of a word according to the vocab variable
            :param word: the word to be represented as one-hot.
            :return:
            """

            return np.where(vocab == word, 1, 0)

        x_and_ys_list = list(map(lambda x_ys: (__one_hot(x_ys[0]), sum(list(map(__one_hot, x_ys[1])))),
                                 x_and_ys_list))  # to tuples (__one_hot,sum_of_one_hots)
    return vocab, x_and_ys_list
