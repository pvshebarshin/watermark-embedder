import unittest

from algorithms.metaheuristics.gbo import GBO
from utils.functions import ackley_function


class GboTest(unittest.TestCase):

    problem_ackley = {
        "fit_func": ackley_function,
        "lb": [-32.768, -32.768],
        "ub": [32.768, 32.768],
        "minmax": "min",
        "log_to": None,
        "save_population": False,
    }

    def test_gbo(self):
        epoch = 500
        pop_size = 50
        pr = 0.5
        beta_min = 0.2
        beta_max = 1.2

        model = GBO(epoch, pop_size, pr, beta_min, beta_max)
        best_position, best_fitness = model.solve(GboTest.problem_ackley)
        self.assertTrue(best_fitness < 0.1)
