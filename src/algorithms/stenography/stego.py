import numpy as np
import math
import cv2

from algorithms.metaheuristics.hho import HHO
from utils.comparator import optimization_function


def embed_secret_message(image_container, secret_message, phi_0=-1.57, phi_1=1.57, iterations=5):
    # define problem
    problem = {
        "fit_func": optimization_function,
        "lb": [-10, -10],
        "ub": [10, 10],
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
    height, width, channels = phase.shape

    # количество блоков по вертикали и горизонтали
    blocks_v = height // 8
    blocks_h = width // 8

    # создание массива блоков
    blocks = np.zeros((blocks_v * blocks_h, 8, 8, channels))

    # Получение эпсилона
    epsilon = calculate_epsilon(len(blocks), len(secret_message))

    # Извлекаем биты из сообщения
    bits = extract_bits_from_word(secret_message)

    pop_size = 50
    epoch = 100
    # HHO
    model = HHO(epoch, pop_size)

    # разделение изображения на блоки
    for i in range(blocks_v):
        for j in range(blocks_h):
            block = phase[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8, :]
            blocks[i * blocks_h + j] = block

    # Идем по блокам изображения
    for _block in blocks:
        dft_coeffs = np.fft.fft(_block)

    # Создаем пустое изображение для сборки блоков
    image = np.zeros((image_height, image_width, 3))

    # Собираем блоки в изображение
    for i in range(blocks_v):
        for j in range(blocks_h):
            block = blocks[i * blocks_h + j]
            image[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8, :] = block


def calculate_epsilon(k, message_length):
    epsilon = math.pi / 2 * message_length / (21 * k)
    if epsilon > math.pi:
        raise ValueError("Message length exceeds capacity of the stego-container")
    return epsilon


def extract_bits_from_word(word):
    """
    Извлекает биты из произвольного слова по 21 штуке.

    Args:
      word: Произвольное слово.

    Returns:
      Массив битов.
    """

    # Преобразуйте произвольное слово в двоичную строку.
    binary_string = bin(int.from_bytes(word.encode(), 'big'))[2:]

    # Определите начальную позицию в строке.
    start_position = 0

    # Создайте массив для хранения извлекаемых битов.
    bits = []

    # Войдите в цикл, который будет выполняться до тех пор, пока вы не достигнете конца строки.
    while start_position < len(binary_string):
        # Извлеките 21 бит из строки, начиная с текущей позиции.
        bits_to_extract = binary_string[start_position:start_position + 21]

        # Добавьте извлеченные биты в массив.
        bits.append(bits_to_extract)

        # Обновите текущую позицию в строке.
        start_position += 21

    # Верните массив битов.
    return bits


def getBitPosition(i):
    match i:
        case 0:
            return [3, 1]
        case 1:
            return [4, 1]
        case 2:
            return [2, 2]
        case 3:
            return [3, 2]
        case 4:
            return [4, 2]
        case 5:
            return [1, 3]
        case 6:
            return [2, 3]
        case 7:
            return [3, 3]
        case 8:
            return [4, 3]
        case 9:
            return [1, 4]
        case 10:
            return [2, 4]
        case 11:
            return [3, 4]
        case 12:
            return [1, 5]
        case 13:
            return [2, 5]
        case 14:
            return [3, 5]
        case 15:
            return [1, 6]
        case 16:
            return [2, 6]
        case 17:
            return [3, 6]
        case 18:
            return [1, 7]
        case 19:
            return [2, 7]
        case 20:
            return [3, 7]


def getBitInversePosition(i):
    match i:
        case 0:
            return [5, 7]
        case 1:
            return [4, 7]
        case 2:
            return [6, 6]
        case 3:
            return [5, 6]
        case 4:
            return [4, 6]
        case 5:
            return [7, 5]
        case 6:
            return [6, 5]
        case 7:
            return [5, 5]
        case 8:
            return [4, 5]
        case 9:
            return [7, 4]
        case 10:
            return [6, 4]
        case 11:
            return [5, 4]
        case 12:
            return [7, 3]
        case 13:
            return [6, 3]
        case 14:
            return [5, 3]
        case 15:
            return [7, 2]
        case 16:
            return [6, 2]
        case 17:
            return [5, 2]
        case 18:
            return [7, 1]
        case 19:
            return [6, 1]
        case 20:
            return [5, 1]
