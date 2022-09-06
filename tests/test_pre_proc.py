from unittest import TestCase

from util.pre_proc import lemmatize, vocab_from_list_of_words, x_and_ys_list_from


class Test(TestCase):

    def test_lemmatize(self):
        assert (lemmatize("swag is cool my bro I like to be cool") == ['swag', 'be', 'cool', 'my', 'bro', 'I', 'like',
                                                                       'to', 'be', 'cool'])

    def test_vocab_from(self):
        x_and_ys_list = [('cool', ['swag', 'be', 'my', 'bro']), ('my', ['be', 'cool', 'bro', 'I']),
                         ('bro', ['cool', 'my', 'I', 'like']), ('I', ['my', 'bro', 'like', 'to']),
                         ('like', ['bro', 'I', 'to', 'be'])]
        assert(['cool', 'my', 'bro', 'I', 'like'] == vocab_from_list_of_words(list(zip(*x_and_ys_list))[0]))

    def test_x_and_ys_list_from(self):
        assert ([('cool', ['swag', 'be', 'my', 'bro']), ('my', ['be', 'cool', 'bro', 'I']),
                 ('bro', ['cool', 'my', 'I', 'like']), ('I', ['my', 'bro', 'like', 'to']),
                 ('like', ['bro', 'I', 'to', 'be'])] == x_and_ys_list_from(
            ['swag', 'be', 'cool', 'my', 'bro', 'I', 'like', 'to', 'be', 'cool'], 2))
