import numpy as np
import math
import cv2


def embed_secret_message(image_container, secret_message, phi_0, phi_1, A_crit, iterations, p):
    # конвертирование изображения в массив numpy
    img_arr = np.array(image_container)

    # получение размеров изображения
    height, width, channels = img_arr.shape

    # количество блоков по вертикали и горизонтали
    blocks_v = height // 8
    blocks_h = width // 8

    # создание массива блоков
    blocks = np.zeros((blocks_v * blocks_h, 8, 8, channels))

    # Получение эпсилона
    epsilon = calculate_epsilon(len(blocks), len(secret_message))

    # Извлекаем биты из сообщения
    bits = extract_bits_from_word(secret_message)

    # разделение изображения на блоки
    for i in range(blocks_v):
        for j in range(blocks_h):
            block = img_arr[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8, :]
            blocks[i * blocks_h + j] = block

    # Идем по блокам изображения
    for _block in blocks:
        dft_coeffs = np.fft.fft(_block)


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
