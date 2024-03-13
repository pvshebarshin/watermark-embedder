import unittest

from algorithms.metaheuristics.bbo import BBO
from utils.functions import xin_she_yang_function_4


class BboTest(unittest.TestCase):

    problem_xin_she_yang_4 = {
        "fit_func": xin_she_yang_function_4,
        "lb": [-10, -10],
        "ub": [10, 10],
        "minmax": "min",
        "log_to": None,
        "save_population": False,
    }

    def test_bbo(self):
        epoch = 1000
        pop_size = 50
        p_m = 0.01
        elites = 2

        model = BBO(epoch, pop_size, p_m, elites)
        best_position, best_fitness = model.solve(BboTest.problem_xin_she_yang_4)
        print(best_fitness)
        self.assertTrue(best_fitness < -1.9)
