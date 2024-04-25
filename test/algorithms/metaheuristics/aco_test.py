import unittest

from algorithms.metaheuristics.single.aco import ACO
from utils.functions import brent_function


class AcoTest(unittest.TestCase):
    problem_brent = {
        "fit_func": brent_function,
        "lb": [-20, -20],
        "ub": [0, 0],
        "minmax": "min",
        "log_to": None,
        "save_population": False,
    }

    def test_aco(self):
        epoch = 500
        pop_size = 50
        sample_count = 25
        intent_factor = 0.5
        zeta = 1.0

        model = ACO(epoch, pop_size, sample_count, intent_factor, zeta)
        best_position, best_fitness = model.solve(AcoTest.problem_brent)
        self.assertTrue(best_fitness < 0.001)
