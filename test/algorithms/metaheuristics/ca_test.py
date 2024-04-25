import unittest

from algorithms.metaheuristics.single.ca import CA
from utils.functions import brent_function


class CaTest(unittest.TestCase):

    problem_brent = {
        "fit_func": brent_function,
        "lb": [-20, -20],
        "ub": [0, 0],
        "minmax": "min",
        "log_to": None,
        "save_population": False,
    }

    def test_ca(self):
        epoch = 500
        accepted_rate = 0.15
        pop_size = 50

        model = CA(epoch, pop_size, accepted_rate)
        best_position, best_fitness = model.solve(CaTest.problem_brent)
        self.assertTrue(best_fitness < 0.001)
