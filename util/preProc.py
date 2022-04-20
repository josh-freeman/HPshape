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
    with open(path, "rw", encoding="utf8") as doc:
        txt = doc.read()
        doc.write(re.sub("^Page \|.*$", repl="", string=txt))
        doc.close()

def removeconsecutiveblanklines(path: str):
    """

    :param path: the path to a given book
    :return:
    """
    with open(path, "rw", encoding="utf8") as doc:
        txt = doc.read()
        doc.write(re.sub("^\s+$", repl="", string=txt))
        doc.close()
