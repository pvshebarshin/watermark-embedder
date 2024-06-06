import unittest

from python.algorithms.stenography.stego import calculate_epsilon


class StegoTest(unittest.TestCase):

    def test_calculate_epsilon(self):
        epsilon = calculate_epsilon(64, 16)
        self.assertEquals(0.018699956271367817, epsilon)
