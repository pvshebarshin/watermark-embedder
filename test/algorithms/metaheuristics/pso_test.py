import unittest

from algorithms.metaheuristics.pso import PSO
from utils.functions import shubert_function


class PsoTest(unittest.TestCase):
    # -186.7309
    problem_shubert = {
        "fit_func": shubert_function,
        "lb": [-10, -10],
        "ub": [10, 10],
        "minmax": "min",
        "log_to": None,
        "save_population": False,
    }

    def test_pso(self):
        epoch = 600
        pop_size = 50
        c1 = 2.05
        c2 = 2.05
        w_min = 0.4
        w_max = 0.9

        model = PSO(epoch, pop_size, c1, c2, w_min, w_max)
        best_position, best_fitness = model.solve(PsoTest.problem_shubert)
        self.assertTrue(best_fitness < -185.0)
