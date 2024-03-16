import unittest

from algorithms.metaheuristics.abc import ABC
from algorithms.metaheuristics.de import DE
from algorithms.metaheuristics.ensemble import MetaheuristicEnsemble
from algorithms.metaheuristics.hho import HHO
from utils.functions import brent_function, schwefel_function, rosenbrook_function


class EnsembleTest(unittest.TestCase):

    problem_brent = {
        "fit_func": brent_function,
        "lb": [-20, -20],
        "ub": [0, 0],
        "minmax": "min",
        "log_to": None,
        "save_population": False,
    }

    problem_schwefel = {
        "fit_func": schwefel_function,
        "lb": [-500, -500],
        "ub": [500, 500],
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
        epoch = 500
        pop_size = 50
        wf = 0.7
        cr = 0.9
        strategy = 0

        model = DE(epoch, pop_size, wf, cr, strategy)
        ensemble = MetaheuristicEnsemble(model)
        best_position, best_fitness = ensemble.solveProblem(EnsembleTest.problem_brent)
        print(f"Best solution: {best_position}, Best fitness: {best_fitness}")

        self.assertTrue(best_fitness < 0.001)

    def test_ensemble_two_evr(self):
        epoch = 400
        pop_size = 50
        wf = 0.7
        cr = 0.9
        strategy = 0

        de = DE(epoch, pop_size, wf, cr, strategy)

        epoch = 100
        pop_size = 50

        hho = HHO(epoch, pop_size)

        ensemble = MetaheuristicEnsemble(de, hho)
        best_position, best_fitness = ensemble.solveProblem(EnsembleTest.problem_schwefel)
        print(f"Best solution: {best_position}, Best fitness: {best_fitness}")
        self.assertTrue(best_fitness < 0.001)

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

        ensemble = MetaheuristicEnsemble(de, hho, abc)
        best_position, best_fitness = ensemble.solveProblem(EnsembleTest.problem_rosenbrook)
        print(f"Best solution: {best_position}, Best fitness: {best_fitness}")
        self.assertTrue(best_fitness < 0.1)
