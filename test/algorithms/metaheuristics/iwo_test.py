import unittest

from algorithms.metaheuristics.iwo import IWO
from utils.functions import keane_function


class IwoTest(unittest.TestCase):

    problem_keane = {
        "fit_func": keane_function,
        "lb": [-10, -10],
        "ub": [10, 10],
        "minmax": "min",
        "log_to": None,
        "save_population": False,
    }

    def test_iwo(self):
        pop_size = 50
        seed_min = 3
        seed_max = 9
        exponent = 3
        sigma_start = 0.6
        sigma_end = 0.01
        epoch = 500

        model = IWO(epoch, pop_size, seed_min, seed_max, exponent, sigma_start, sigma_end)
        best_position, best_fitness = model.solve(IwoTest.problem_keane)
        self.assertTrue(best_fitness < -0.5)
