""" a few utilitary functions for preprocessing"""
import re


def allinoneline(path: str):
    with open(path, "rw", encoding="utf8") as doc:
        txt = doc.read()
        doc.write(txt.replace("\n", " "))
        doc.close()


def removepagelineshp(path: str):
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


def removeconsecutiveblanklines(path: str):
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


def preproc(text: str, c: int) -> (list, list):
    """
    :param c: the window size
    :param text:
    :return: a tuple of
     - first, a vocab of size V:=size(vocab). It is represented by a list of all words of interest found in the text.
     - second, a list[(ndarray(shape=(V,1),ndarray(shape=(V,1)))]. Tuples represents a one-hot word, and its context (as a sum of the one-hot vectors of the words it comprises).
    """
    return ...
