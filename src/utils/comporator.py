import numpy as np
import math
from scipy import signal

C_MAX = 21
PSNR_MAX = 50

K_E = 0.5
K_PSNR = 0.4
K_C = 0.1


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
    psnr_block = 10 * math.log10((255**2) / _mse)
    return psnr_block / PSNR_MAX


def optimization_function(block_before, block_after):
    return (
            K_C * getCapacity_f(block_before, block_after)
            + K_PSNR * getPSNR_f(block_before, block_after)
            + K_E * getErrors(block_before, block_after)
    )


def correlation(image1, image2):
    return signal.correlate2d(image1, image2)


def meanSquareError(image1, image2):
    error = np.sum((image1.astype('float') - image2.astype('float')) ** 2)
    error /= float(image1.shape[0] * image1.shape[1])
    return error


def psnr(image1, image2):
    _mse = meanSquareError(image1, image2)
    if _mse == 0:
        return 100
    pixel_max = 255.0
    return 20 * math.log10(pixel_max / math.sqrt(_mse))
