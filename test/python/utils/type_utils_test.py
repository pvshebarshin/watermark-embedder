import unittest

from python.utils.type_utils import getBitPosition, getBitInversePosition, string_to_bitstring, merge_matrices


class TypeUtilsTest(unittest.TestCase):

    def test_merge_matrices(self):
        matrix1 = [[0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0]]
        matrix2 = [[3, 0, 0, 0, 0, 0, 0, 0],
                   [3, 0, 0, 0, 0, 0, 0, 0],
                   [3, 0, 0, 0, 0, 0, 0, 0],
                   [3, 0, 0, 0, 0, 0, 0, 0],
                   [3, 0, 0, 0, 0, 0, 0, 0],
                   [3, 0, 0, 0, 0, 0, 0, 0],
                   [3, 0, 0, 0, 0, 0, 0, 0],
                   [3, 0, 0, 0, 0, 0, 0, 0]]
        matrix3 = [[1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1]]
        matrix4 = [[4, 1, 1, 1, 1, 1, 1, 1],
                   [4, 1, 1, 1, 1, 1, 1, 1],
                   [4, 1, 1, 1, 1, 1, 1, 1],
                   [4, 1, 1, 1, 1, 1, 1, 1],
                   [4, 1, 1, 1, 1, 1, 1, 1],
                   [4, 1, 1, 1, 1, 1, 1, 1],
                   [4, 1, 1, 1, 1, 1, 1, 1],
                   [4, 1, 1, 1, 1, 1, 1, 1]]

        array_of_matrices = [matrix1, matrix2, matrix3, matrix4]
        result_matrix = merge_matrices(array_of_matrices)

        self.assertEquals(0, result_matrix[0][0])
        self.assertEquals(1, result_matrix[15][15])
        self.assertEquals(3, result_matrix[7][8])
        self.assertEquals(4, result_matrix[8][8])

        array_of_matrices = [matrix1, matrix1, matrix1, matrix3, matrix3, matrix3, matrix4, matrix4, matrix4]
        result_matrix = merge_matrices(array_of_matrices)

        self.assertEquals(0, result_matrix[0][0])
        self.assertEquals(1, result_matrix[23][23])
        self.assertEquals(4, result_matrix[16][16])
        self.assertEquals(4, result_matrix[23][8])

    def test_string_to_bitstring(self):
        bits = string_to_bitstring('HHH')
        self.assertEquals(24, len(bits))

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
