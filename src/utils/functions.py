from numpy import exp, sin, cos, pi, sqrt, fabs


def easom_function(x):
    return -(cos(x[0]) * cos(x[1])) * exp(-(x[0] - pi) ** 2 - (x[1] - pi) ** 2)


def ackley_function(x):
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))) \
        - exp(0.5 * (cos(2 * pi * x[0]) + cos(2 * pi * x[1]))) + exp(1) + 20


# Composite
def ackley_function_3(x):
    return -200 * exp(-0.2 * sqrt(x[0] ** 2 + x[1] ** 2)) + 5 * exp(cos(3 * x[0]) + sin(3 * x[1]))


# Composite
def keane_function(x):
    return -sin(x[0] - x[1]) ** 2 * sin(x[0] + x[1]) ** 2 / sqrt(x[0] * x[0] + x[1] * x[1])


def michaelwicz_function(x):
    return -1 * ((sin(x[0]) * sin((1 * x[0] ** 2) / pi) ** 20) + (sin(x[1]) * sin((2 * x[1] ** 2) / pi) ** 20))


def rosenbrook_function(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[1] ** 2) ** 2


def nonsmooth_multipeak_function(x):
    return (abs(x[0]) + abs(x[1])) * exp(-0.0625 * (x[0] ** 2 + x[1] ** 2))


def brown_function(x):
    return (x[0] * x[0]) ** (x[1] * x[1] + 1) + (x[1] * x[1]) ** (x[0] * x[1] + 1)


# Composite
def levin_function_13(x):
    return sin(3 * pi * x[0]) ** 2 + (x[0] - 1) ** 2 * (1 + sin(3 * pi * x[1]) * sin(3 * pi * x[1])) \
        + (x[1] - 1) * (x[1] - 1) * (1 + sin(2 * pi * x[1]) * sin(2 * pi * x[1]))


# Composite
def bird_function(x):
    return sin(x[0]) * (exp(1 - cos(x[1])) ** 2) + cos(x[1]) * (exp(1 - sin(x[0])) ** 2) + (x[0] - x[1]) ** 2


# Composite
def schwefel_function(x):
    return 418.9829 * 2 - x[0] * sin(sqrt(abs(x[0]))) - x[1] * sin(sqrt(abs(x[1])))


def brent_function(x):
    return (x[0] + 10) ** 2 + (x[1] + 10) ** 2 + exp(-x[0] ** 2 - x[1] ** 2)


def deckkers_aarts_function(x):
    return (10 ** 5) * (x[0] ** 2) + (x[1] ** 2) - ((x[0] ** 2) + (x[1] ** 2)) ** 2 + (10 ** (-5)) * (
            (x[0] ** 2) + (x[1] ** 2)) ** 4


# Composite
def xin_she_yang_function_4(x):
    return (sin(x[0]) * sin(x[0]) - exp(-(x[0] ** 2 + x[1] ** 2))) * exp(-(pow(sin(sqrt(abs(x[0]))), 2))) \
        + (sin(x[1]) * sin(x[1]) - exp(-(x[0] ** 2 + x[1] ** 2))) * exp(-pow(sin(sqrt(abs(x[1]))), 2))


# Composite
def eggholder_function(x):
    return -(x[1] + 47) * sin(sqrt(fabs(x[1] + x[0] / 2 + 47))) - x[0] * sin(sqrt(fabs(x[0] - (x[1] + 47))))


# Composite
def shubert_function(x):
    sum1 = 0
    sum2 = 0
    for i in range(1, 6):
        sum1 = sum1 + i * cos((i + 1) * x[0] + i)
        sum2 = sum2 + i * cos((i + 1) * x[1] + i)
    return sum1 * sum2


# Composite 0
def schaffer_function(x):
    num = sin((x[0] ** 2 + x[1] ** 2) ** 2) ** 2 - 0.5
    den = (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2
    return 0.5 + num / den


# Composite −25.740858
def shubert_function_4(x):
    y = 0
    for j in range(1, 6):
        y += j * cos((j + 1) * x[0] + j) + j * cos((j + 1) * x[1] + j)
    return y


# Composite −1
def drop_wave_function(x):
    return -(1 + cos(12 * sqrt(x[0] * x[0] + x[1] * x[1]))) \
        / (0.5 * (x[0] * x[0] + x[1] * x[1]) + 2)


# Composite -2.06261
def cross_in_tray_function(x):
    a = fabs(100 - sqrt(x[0] * x[0] + x[1] * x[1]) / pi)
    b = fabs(sin(x[0]) * sin(x[1]) * exp(a)) + 1
    return -0.0001 * b ** 0.1


# [6.16547952 6.16547958] 249.1391477271118
def composite_xin_she_yang_4_schwefel_function(x):
    return 0.3 * schwefel_function(x) + 0.7 * xin_she_yang_function_4(x)


# -1.268712223626797
def composite_cross_in_tray_drop_wave_function(x):
    return 0.4 * cross_in_tray_function(x) + 0.6 * drop_wave_function(x)


# -0.8
def composite_schaffer_drop_wave_function(x):
    return 0.2 * schaffer_function(x) + 0.8 * drop_wave_function(x)


# -92.54689320077729
def composite_shubert_function(x):
    return 0.5 * shubert_function_4(x) + 0.5 * shubert_function(x)


# -13.128066255311966
def composite_shubert_4_levin_function_13(x):
    return 0.7 * shubert_function_4(x) + 0.3 * levin_function_13(x)
