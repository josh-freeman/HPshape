from unittest import TestCase

from util.util import plot_losses


class Test(TestCase):
    def test_plot_losses(self):
        plot_losses([1, 2, 3, 4], [1, 3, 6, 2], description="test: $swag$")
