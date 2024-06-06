import unittest

from python.algorithms.metaheuristics.single.abc import ABC
from python.utils.functions import brent_function


class AbcTest(unittest.TestCase):

    problem_brent = {
        "fit_func": brent_function,
        "lb": [-20, -20],
        "ub": [0, 0],
        "minmax": "min",
        "log_to": None,
        "save_population": False,
    }

    def test_abc(self):
        epoch = 500
        n_limits = 50
        pop_size = 50

        # ABC
        model = ABC(epoch, pop_size, n_limits)
        best_position, best_fitness = model.solve(AbcTest.problem_brent)
        self.assertTrue(best_fitness < 0.001)
