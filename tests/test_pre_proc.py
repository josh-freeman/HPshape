from unittest import TestCase

from util.pre_proc import lemmatize


class Test(TestCase):

    def test_lemmatize(self):
        assert (lemmatize("swag is cool my bro I like to be cool") == ['swag', 'be', 'cool', 'my', 'bro', 'I', 'like',
                                                                       'to', 'be', 'cool'])


