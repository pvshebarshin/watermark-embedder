import unittest

from algorithms.stenography.stego import getBitPosition, extract_bits_from_word, calculate_epsilon, \
    getBitInversePosition


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

    def test_getPositions_for_array(self):
        matrix = [[1, 2, 3, 4, 5, 6, 7, 8],
                  [9, 10, 11, 12, 13, 14, 15, 16],
                  [17, 18, 19, 20, 21, 22, 23, 24],
                  [25, 26, 27, 28, 29, 30, 31, 32],
                  [33, 34, 35, 36, 37, 38, 39, 40],
                  [41, 42, 43, 44, 45, 46, 47, 48],
                  [49, 50, 51, 52, 53, 54, 55, 56],
                  [57, 58, 59, 60, 61, 62, 63, 64]]

        pos = getBitPosition(0)
        matrix[pos[0]][pos[1]] = 10000
        self.assertEquals(10000, matrix[3][1])

    def test_inverse_positions(self):
        matrix1 = [[1, 2, 3, 4, 5, 6, 7, 8],
                   [9, 10, 11, 12, 13, 14, 15, 16],
                   [17, 18, 19, 20, 21, 22, 23, 24],
                   [25, 26, 27, 28, 29, 30, 31, 32],
                   [33, 34, 35, 36, 37, 38, 39, 40],
                   [41, 42, 43, 44, 45, 46, 47, 48],
                   [49, 50, 51, 52, 53, 54, 55, 56],
                   [57, 58, 59, 60, 61, 62, 63, 64]]

        matrix2 = [[1, 2, 3, 4, 5, 6, 7, 8],
                   [9, 10, 11, 12, 13, 14, 15, 16],
                   [17, 18, 19, 20, 21, 22, 23, 24],
                   [25, 26, 27, 28, 29, 30, 31, 32],
                   [33, 34, 35, 36, 37, 38, 39, 40],
                   [41, 42, 43, 44, 45, 46, 47, 48],
                   [49, 50, 51, 52, 53, 54, 55, 56],
                   [57, 58, 59, 60, 61, 62, 63, 64]]

        for i in range(len(matrix1)):
            for j in range(len(matrix1[0])):
                if i == 0 or j == 0 or (i == 4 and j == 4):
                    continue
                matrix1[8 - j][8 - i] = -matrix1[j][i]

        for i in range(21):
            pos = getBitPosition(i)
            inv_pos = getBitInversePosition(i)
            matrix2[inv_pos[1]][inv_pos[0]] = -matrix2[pos[1]][pos[0]]
        matrix2[7][7] = -matrix2[1][1]
        matrix2[6][7] = -matrix2[2][1]
        matrix2[7][6] = -matrix2[1][2]

        for i in range(len(matrix1)):
            for j in range(len(matrix1[0])):
                self.assertEquals(matrix1[i][j], matrix2[i][j])

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
