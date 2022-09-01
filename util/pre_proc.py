""" a few utilitary functions for preprocessing"""
import re
import string
from collections import Counter
import spacy

import numpy as np


def allinoneline(path: str):
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


def remove_consecutive_blanklines(path: str):
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
    return text.translate(str.maketrans('', '', string.punctuation))


def lemmatize(text: str):
    """

    :param text: a text *without punctuation*.
    :return: list of lemmas
    """
    assert (not any(p in text for p in string.punctuation))
    load_model = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    doc = load_model(text)
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
        ret += (word, context)
    return ret


def vocab_from(x_and_ys_list, c):
    """

    :param x_and_ys_list:
    :param c:
    :return: an ordered SET (list with no doubles)
    """
    x, _ = zip(*x_and_ys_list)
    cnt = Counter(x)
    return [k for k, v in cnt.items() if v > c]


def preproc(text: str, c: int) -> (list, list):
    """
    :param c: the window size
    :param text:
    :return: a tuple of - first, a vocab of size V:=size(vocab). It is
    represented by a list of all words of interest found in the text. - second, a list[(ndarray(shape=(V,1),
    ndarray(shape=(V,1)))]. Tuples represents a one-hot word, and its context (as a sum of the one-hot vectors of the
    words it comprises).
    """

    text = remove_punctuation(text)  # remove punctuation
    tokenized_word_list = lemmatize(text)  # list of lemmas

    x_and_ys_list = x_and_ys_list_from(tokenized_word_list, c)  # make a first list of tuples from the tokens.
    vocab = np.ndarray(
        vocab_from(x_and_ys_list, c))  # Establish a list of all words at the center of c contexts (:=vocab)

    x_and_ys_list = [(x, ys) for (x, ys) in x_and_ys_list if
                     x in vocab and all(y in vocab for y in ys)]  # only keep tuples containing words in vocab

    def __one_hot(word) -> np.ndarray:
        """
        return one hot version of a word according to the vocab variable
        :param word: the word to be represented as one-hot.
        :return:
        """
        return np.where(vocab == word, 1, 0)

    x_and_ys_list = map(lambda x_ys: (__one_hot(x_ys[0]), sum(map(__one_hot, x_ys[1]))),
                        x_and_ys_list)  # to tuples (__one_hot,sum_of_one_hots)

    return vocab, x_and_ys_list