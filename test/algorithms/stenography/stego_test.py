import unittest

from algorithms.stenography.stego import getBitPosition, extract_bits_from_word, calculate_epsilon


class StegoTest(unittest.TestCase):

    def test_getPositions(self):
        self.assertEquals([3, 1], getBitPosition(0))
        self.assertEquals([4, 1], getBitPosition(1))
        self.assertEquals([2, 2], getBitPosition(2))
        self.assertEquals([3, 2], getBitPosition(3))
        self.assertEquals([4, 2], getBitPosition(4))
        self.assertEquals([1, 3], getBitPosition(5))
        self.assertEquals([2, 3], getBitPosition(6))
        self.assertEquals([3, 3], getBitPosition(7))
        self.assertEquals([4, 3], getBitPosition(8))
        self.assertEquals([1, 4], getBitPosition(9))
        self.assertEquals([2, 4], getBitPosition(10))
        self.assertEquals([3, 4], getBitPosition(11))
        self.assertEquals([1, 5], getBitPosition(12))
        self.assertEquals([2, 5], getBitPosition(13))
        self.assertEquals([3, 5], getBitPosition(14))
        self.assertEquals([1, 6], getBitPosition(15))
        self.assertEquals([2, 6], getBitPosition(16))
        self.assertEquals([3, 6], getBitPosition(17))
        self.assertEquals([1, 7], getBitPosition(18))
        self.assertEquals([2, 7], getBitPosition(19))
        self.assertEquals([3, 7], getBitPosition(20))

    def test_extract_bits_from_word(self):
        message = '1234567'
        array = extract_bits_from_word(message)

        self.assertEquals(21, len(array[0]))
        self.assertEquals(21, len(array[1]))
        self.assertEquals(12, len(array[2]))

        self.assertEquals('110001001100100011001', array[0])
        self.assertEquals('100110100001101010011', array[1])
        self.assertEquals('011000110111', array[2])

    def test_calculate_epsilon(self):
        message = '69'
        array = extract_bits_from_word(message)
        epsilon = calculate_epsilon(64, len(array))

        self.assertEquals(0.0011687472669604886, epsilon)
