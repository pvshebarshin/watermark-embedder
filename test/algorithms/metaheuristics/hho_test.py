import unittest

from algorithms.metaheuristics.single.hho import HHO
from utils.functions import easom_function


class HhoTest(unittest.TestCase):

    problem_easom = {
        "fit_func": easom_function,
        "lb": [-100, -100],
        "ub": [100, 100],
        "minmax": "min",
        "log_to": None,
        "save_population": False,
    }

    def test_hho(self):
        epoch = 500
        pop_size = 50

        model = HHO(epoch, pop_size)
        best_position, best_fitness = model.solve(HhoTest.problem_easom)
        self.assertTrue(best_fitness < -0.9)
