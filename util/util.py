import os
from unidecode import unidecode as decode

def absolute_path(relative_path):
    """
    :param relative_path: The relative path from the dir of __main__.py
    :return: absolute path
    """
    return os.path.dirname(__file__) + "/../" + relative_path

def normalize(s):
    """
    normalize a single string
    :param s:the string to normalize
    :return:normalized string (lowered and unidecode.decoded)
    """
    return decode(s).lower().strip()

def distinct(lst):
    """
    remove duplicates from list
    :param lst: list to remove duplicates from
    :return: list free of duplicates
    """
    return list(dict.fromkeys(lst))