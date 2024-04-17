import math
from copy import deepcopy

import cv2
import numpy as np

from utils.comparator import getCapacity_f, getPSNR_f
from utils.type_utils import string_to_bitstring, getBitPosition, getBitInversePosition, merge_matrices

C_MAX = 21
PSNR_MAX = 50

K_E = 0.5
K_PSNR = 0.4
K_C = 0.1

TEMP_BLOCK = [[0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]]

TEMP_MAG = [[0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]]


def getErrors(block_after):
    real, imag = cv2.polarToCart(TEMP_MAG, block_after + math.pi)
    back = cv2.merge([real, imag])
    back_ishift = np.fft.ifftshift(back)
    img_back = cv2.idft(back_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = cv2.normalize(img_back, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    dft = cv2.dft(np.float32(img_back), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    mag, phase = cv2.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])
    phase -= math.pi
    errors = 0
    for i in range(len(phase)):
        for j in range(len(phase[0])):
            if phase[i][j] != block_after[i][j]:
                errors += 1
    return errors


def optimization_function(solution):
    block_before = deepcopy(TEMP_BLOCK)
    for i in range(len(solution)):
        pos = getBitPosition(i)
        inv_pos = getBitInversePosition(i)
        TEMP_BLOCK[pos[1]][pos[0]] = solution[i]
        TEMP_BLOCK[inv_pos[1]][inv_pos[0]] = -TEMP_BLOCK[pos[1]][pos[0]]
    TEMP_BLOCK[7][7] = -TEMP_BLOCK[1][1]
    TEMP_BLOCK[6][7] = -TEMP_BLOCK[2][1]
    TEMP_BLOCK[7][6] = -TEMP_BLOCK[1][2]

    return (
            K_C * getCapacity_f(block_before, TEMP_BLOCK)
            + K_PSNR * getPSNR_f(block_before, TEMP_BLOCK)
            + K_E * getErrors(TEMP_BLOCK)
    )


def embed_secret_message(image_container, secret_message, model, epsilon=1, phi_0=-1.57, phi_1=1.57, iterations=5):
    # define problem
    problem = {
        "fit_func": optimization_function,
        "lb": [-math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi,
               -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi,
               -math.pi],
        "ub": [math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi,
               math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi],
        "minmax": "min",
        "log_to": None,
        "save_population": False,
    }

    # convert image to floats and do dft saving as complex output
    dft = cv2.dft(np.float32(image_container), flags=cv2.DFT_COMPLEX_OUTPUT)

    # apply shift of origin from upper left corner to center of image
    dft_shift = np.fft.fftshift(dft)

    # extract magnitude and phase images
    mag, phase = cv2.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])
    phase -= math.pi

    # получение размеров изображения
    height, width = phase.shape

    # количество блоков по вертикали и горизонтали
    blocks_v = height // 8
    blocks_h = width // 8

    # создание массива блоков
    blocks = mags = np.zeros((blocks_v * blocks_h, 8, 8))

    # Получение эпсилона
    calculate_epsilon(len(blocks), len(secret_message))

    # Извлекаем биты из сообщения
    bits = string_to_bitstring(secret_message)

    # разделение изображения на блоки
    for i in range(blocks_v):
        for j in range(blocks_h):
            blocks[i * blocks_h + j] = phase[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            mags[i * blocks_h + j] = mag[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]

    # Идем по блокам изображения
    for i in range(len(blocks)):
        global TEMP_BLOCK
        global TEMP_MAG
        TEMP_BLOCK = blocks[i]
        TEMP_MAG = mags[i]
        best_position, best_fitness = model.solveProblem(problem)

        blocks[i] = TEMP_BLOCK
        for position_number in range(len(best_position)):
            pos = getBitPosition(position_number)
            _block = blocks[position_number][pos[1]][pos[0]]
            if len(bits) == 0:
                break

            bit = bits[0]
            if phi_1 - epsilon <= _block <= phi_1 - epsilon:
                if bit == '0':
                    blocks[position_number][pos[1]][pos[0]] = -_block
                    bits = bits[1:]
                else:
                    bits = bits[1:]
            else:
                continue

    # Собираем блоки в изображение
    phase_for_image = merge_matrices(blocks)
    phase_for_image += math.pi

    # convert magnitude and phase into cartesian real and imaginary components
    real, imag = cv2.polarToCart(mag, phase_for_image)

    # combine cartesian components into one complex image
    back = cv2.merge([real, imag])

    # shift origin from center to upper left corner
    back_ishift = np.fft.ifftshift(back)

    # do idft saving as complex output
    img_back = cv2.idft(back_ishift)

    # combine complex components into original image again
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # re-normalize to 8-bits
    img_back = cv2.normalize(img_back, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return img_back


def calculate_epsilon(k, message_length):
    epsilon = math.pi / 2 * message_length / (21 * k)
    if epsilon > math.pi:
        raise ValueError("Message length exceeds capacity of the stego-container")
    return epsilon
