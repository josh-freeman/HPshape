""" a few utilitary functions for preprocessing"""
import re
import string
from collections import Counter


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
    return ...


def x_and_ys_list_from(tokenized_word_list, c):
    return ...


def vocab_from(x_and_ys_list, c):
    x, _ = zip(*x_and_ys_list)
    cnt = Counter(x)
    return [k for k, v in cnt.items() if v > c]


def preproc(text: str, c: int) -> (list, list):
    """
    :param c: the window size
    :param text:
    :return: a tuple of
     - first, a vocab of size V:=size(vocab). It is represented by a list of all words of interest found in the text.
     - second, a list[(ndarray(shape=(V,1),ndarray(shape=(V,1)))]. Tuples represents a one-hot word, and its context (as a sum of the one-hot vectors of the words it comprises).
    """

    text = remove_punctuation(text)  # remove punctuation
    text = lemmatize(text)
    tokenized_word_list = text.split()  # tokenize with text.split()

    x_and_ys_list = x_and_ys_list_from(tokenized_word_list, c)
    vocab = vocab_from(x_and_ys_list, c)  # Establish a list of all words at the center of c contexts (:=vocab)

    return ...
