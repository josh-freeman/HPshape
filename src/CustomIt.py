from gensim import utils

from util.constants import RESOURCES_DIRNAME, BOOK_NAMES, CURR_BOOK_NR
from util.util import absolute_path
class CustomIt:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        for i in [CURR_BOOK_NR]: ##for the moment, only run this i
            corpus_path = absolute_path(f"/{RESOURCES_DIRNAME}/{BOOK_NAMES[i]}")
            with open(corpus_path, encoding="utf8") as line:
                # assume there's one document per corpus_path, tokens separated by whitespace
                yield utils.simple_preprocess(line.read())

