import unittest

from algorithms.metaheuristics.single.fbio import FBIO
from utils.functions import eggholder_function


class FbioTest(unittest.TestCase):
    # -959.6407
    problem_eggholder = {
        "fit_func": eggholder_function,
        "lb": [-512, -512],
        "ub": [512, 512],
        "minmax": "min",
        "log_to": None,
        "save_population": False,
    }

    def test_fbio(self):
        epoch = 150
        pop_size = 50

        # FBIO
        model = FBIO(epoch, pop_size)
        best_position, best_fitness = model.solve(FbioTest.problem_eggholder)
        self.assertTrue(best_fitness < -958)
