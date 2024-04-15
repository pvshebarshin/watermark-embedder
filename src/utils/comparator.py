import numpy as np
import math

C_MAX = 21
PSNR_MAX = 50

K_E = 0.5
K_PSNR = 0.4
K_C = 0.1


def count_different_cells(matrix1, matrix2):
    count = 0
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            if matrix1[i][j] != matrix2[i][j]:
                count += 1
    return count


def mse(I, S):
    if I.shape != S.shape:
        raise ValueError("Изображения должны иметь одинаковый размер.")

    diff = I - S
    squared_diff = diff ** 2
    sum_squared_diff = np.sum(squared_diff)
    return sum_squared_diff / (I.shape[0] * I.shape[1])


def getPSNR_f(block_before, block_after):
    _mse = mse(block_before, block_after)
    if _mse == 0:
        return 0
    psnr_block = 10 * math.log10((255 ** 2) / _mse)
    return psnr_block / PSNR_MAX


def getCapacity_f(block_before, block_after):
    return count_different_cells(block_before, block_after) / C_MAX



# def optimization_function(block_before, block_after):
#     return (
#             K_C * getCapacity_f(block_before, block_after)
#             + K_PSNR * getPSNR_f(block_before, block_after)
#             + K_E * getErrors(block_before, block_after)
#     )
