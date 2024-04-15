import math

import numpy as np


def merge_matrices(matrices):

    # Get the size of the merged matrix
    size = int(math.sqrt(len(matrices)))

    # Create the merged matrix
    merged_matrix = np.zeros((size * 8, size * 8))

    # Merge the matrices
    start_row = 0
    start_col = 0
    for matrix in matrices:
        end_row = start_row + len(matrix)
        end_col = start_col + len(matrix[0])
        merged_matrix[start_row:end_row, start_col:end_col] = matrix
        start_col = end_col

        if size * 8 == start_col:
            start_col = 0
            start_row += len(matrix[0])

    return merged_matrix


def string_to_bitstring(_string):
    """Converts a string to a bitstring.

    Args:
      _string: The string to convert.

    Returns:
      A bitstring representation of the string.
    """

    bitstring = ""
    for char in _string:
        # Convert the character to its ASCII code.
        ascii_code = ord(char)

        # Convert the ASCII code to a binary string.
        binary_string = bin(ascii_code)[2:]

        # Pad the binary string with zeros to make it 8 bits long.
        binary_string = binary_string.zfill(8)

        # Add the binary string to the bitstring.
        bitstring += binary_string

    return bitstring


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
