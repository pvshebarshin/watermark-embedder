import unittest

from algorithms.metaheuristics.ga import GA
from utils.functions import rosenbrook_function


class GaTest(unittest.TestCase):

    problem_rosenbrook = {
        "fit_func": rosenbrook_function,
        "lb": [-5, -5],
        "ub": [10, 10],
        "minmax": "min",
        "log_to": None,
        "save_population": False,
    }

    def test_ga(self):
        epoch = 1000
        pop_size = 50
        pc = 0.9
        pm = 0.05
        selection = "roulette"
        crossover = "uniform"
        mutation = "swap"

        model = GA(epoch, pop_size, pc, pm, selection, crossover, mutation)
        best_position, best_fitness = model.solve(GaTest.problem_rosenbrook)
        print(best_fitness)
        self.assertTrue(best_fitness < 0.99)
