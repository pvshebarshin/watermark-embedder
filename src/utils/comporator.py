import numpy as np
import math
from scipy import signal


def correlation(image1, image2):
    return signal.correlate2d(image1, image2)


def meanSquareError(image1, image2):
    error = np.sum((image1.astype('float') - image2.astype('float')) ** 2)
    error /= float(image1.shape[0] * image1.shape[1])
    return error


def psnr(image1, image2):
    mse = meanSquareError(image1, image2)
    if mse == 0:
        return 100
    pixel_max = 255.0
    return 20 * math.log10(pixel_max / math.sqrt(mse))
