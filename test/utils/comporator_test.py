import unittest
import numpy as np

from utils.comparator import count_different_cells, mse, getPSNR_f, getCapacity_f


class ComparatorTest(unittest.TestCase):

    def test_count_different_cells(self):
        matrix1 = [[1.0, 2, 3, 44.44],
                   [5, 6, 4, 8],
                   [9, 10, 11, 12.12]]
        matrix2 = [[1.0, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12.12]]

        self.assertEquals(2, count_different_cells(matrix1, matrix2))

        matrix1 = np.array([[1.0, 2, 3, 44.44],
                            [5, 6, 4, 8],
                            [9, 10, 11, 12.12]])
        matrix2 = np.array([[1.0, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12.12]])

        self.assertEquals(2, count_different_cells(matrix1, matrix2))

    def test_mse(self):
        matrix1 = np.array([[1.0, 2, 3, 44.44],
                            [5, 6, 4, 8],
                            [9, 10, 11, 12.12]])
        matrix2 = np.array([[1.0, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12.12]])

        self.assertEquals(137.03279999999998, mse(matrix1, matrix2))

    def test_getPSNR_f(self):
        matrix1 = np.array([[1.0, 2, 3, 44.44],
                            [5, 6, 4, 8],
                            [9, 10, 11, 12.12]])
        matrix2 = np.array([[1.0, 2, 3, 44.44],
                            [5, 6, 4, 8],
                            [9, 10, 11, 12.12]])

        self.assertEquals(0, getPSNR_f(matrix1, matrix2))

        matrix2 = np.array([[1.0, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12.12]])

        self.assertEquals(0.5352511658166736, getPSNR_f(matrix1, matrix2))

    def test_getCapacity_f(self):
        matrix1 = np.array([[1.0, 2, 3, 44.44],
                            [5, 6, 4, 8],
                            [9, 10, 11, 12.12]])
        matrix2 = np.array([[1.1, 3, 4, 4],
                            [5, 4, 7, 3],
                            [9, 10, 11, 12.12]])

        self.assertEquals(0.3333333333333333, getCapacity_f(matrix1, matrix2))
