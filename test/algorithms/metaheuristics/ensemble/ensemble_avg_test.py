import unittest

from algorithms.metaheuristics.single.abc import ABC
from algorithms.metaheuristics.single.de import DE
from algorithms.metaheuristics.ensemble.ensemble_avg import MetaheuristicEnsembleAvg
from algorithms.metaheuristics.single.hho import HHO
from utils.functions import brent_function, rosenbrook_function, ackley_function


class EnsembleAvgTest(unittest.TestCase):
    problem_brent = {
        "fit_func": brent_function,
        "lb": [-20, -20],
        "ub": [0, 0],
        "minmax": "min",
        "log_to": None,
        "save_population": False,
    }

    problem_ackley = {
        "fit_func": ackley_function,
        "lb": [-32.768, -32.768],
        "ub": [32.768, 32.768],
        "minmax": "min",
        "log_to": None,
        "save_population": False,
    }

    problem_rosenbrook = {
        "fit_func": rosenbrook_function,
        "lb": [-5, -5],
        "ub": [10, 10],
        "minmax": "min",
        "log_to": None,
        "save_population": False,
    }

    def test_ensemble_one_evr(self):
        term_dict = {
            "max_time": 60  # seconds
        }
        epoch = 500
        pop_size = 50
        wf = 0.7
        cr = 0.9
        strategy = 0

        model = DE(epoch, pop_size, wf, cr, strategy)
        ensemble = MetaheuristicEnsembleAvg(model, term_dict=term_dict)
        best_position, best_fitness = ensemble.solveProblem(EnsembleAvgTest.problem_brent)
        print(f"Best solution: {best_position}, Best fitness: {best_fitness}")

        self.assertTrue(best_fitness < 0.1)

    def test_ensemble_two_evr(self):
        epoch = 400
        pop_size = 50
        wf = 0.7
        cr = 0.9
        strategy = 0

        de = DE(epoch, pop_size, wf, cr, strategy)

        epoch = 400
        pop_size = 50

        hho = HHO(epoch, pop_size)

        ensemble = MetaheuristicEnsembleAvg(de, hho)
        best_position, best_fitness = ensemble.solveProblem(EnsembleAvgTest.problem_ackley)
        print(f"Best solution: {best_position}, Best fitness: {best_fitness}")
        self.assertTrue(best_fitness < 0.1)

    def test_ensemble_three_evr(self):
        epoch = 400
        pop_size = 50
        wf = 0.7
        cr = 0.9
        strategy = 0

        de = DE(epoch, pop_size, wf, cr, strategy)

        epoch = 200
        pop_size = 50

        hho = HHO(epoch, pop_size)

        n_limits = 50
        pop_size = 50
        epoch = 200

        abc = ABC(epoch, pop_size, n_limits)

        ensemble = MetaheuristicEnsembleAvg(de, hho, abc)
        best_position, best_fitness = ensemble.solveProblem(EnsembleAvgTest.problem_rosenbrook)
        print(f"Best solution: {best_position}, Best fitness: {best_fitness}")
        self.assertTrue(best_fitness < 0.1)
