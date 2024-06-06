import unittest

from python.algorithms.metaheuristics.single.de import DE
from python.utils.functions import brent_function


class DeTest(unittest.TestCase):

    problem_brent = {
        "fit_func": brent_function,
        "lb": [-20, -20],
        "ub": [0, 0],
        "minmax": "min",
        "log_to": None,
        "save_population": False,
    }

    def test_de(self):
        epoch = 500
        pop_size = 50
        wf = 0.7
        cr = 0.9
        strategy = 0

        model = DE(epoch, pop_size, wf, cr, strategy)
        best_position, best_fitness = model.solve(DeTest.problem_brent)
        self.assertTrue(best_fitness < 0.001)
