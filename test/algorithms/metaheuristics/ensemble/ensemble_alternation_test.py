import unittest

from algorithms.metaheuristics.single.abc import ABC
from algorithms.metaheuristics.single.de import DE
from algorithms.metaheuristics.ensemble.ensemble_alternation import MetaheuristicEnsembleAlternation
from algorithms.metaheuristics.single.hho import HHO
from utils.functions import brent_function, rosenbrook_function, nonsmooth_multipeak_function


class EnsembleAlternationTest(unittest.TestCase):
    problem_brent = {
        "fit_func": brent_function,
        "lb": [-20, -20],
        "ub": [0, 0],
        "minmax": "min",
        "log_to": None,
        "save_population": False,
    }

    problem_nonsmooth_multipeak = {
        "fit_func": nonsmooth_multipeak_function,
        "lb": [0, 0],
        "ub": [3, 3],
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
        epoch = 40
        pop_size = 50
        wf = 0.7
        cr = 0.9
        strategy = 0

        model = DE(epoch, pop_size, wf, cr, strategy)
        ensemble = MetaheuristicEnsembleAlternation(model)
        best_position, best_fitness = ensemble.solveProblem(EnsembleAlternationTest.problem_brent)
        print(f"Best solution: {best_position}, Best fitness: {best_fitness}")

        self.assertTrue(best_fitness < 0.1)

    def test_ensemble_two_evr(self):
        epoch = 20
        pop_size = 50
        wf = 0.7
        cr = 0.9
        strategy = 0

        de = DE(epoch, pop_size, wf, cr, strategy)

        epoch = 20
        pop_size = 50

        hho = HHO(epoch, pop_size)

        ensemble = MetaheuristicEnsembleAlternation(de, hho)
        best_position, best_fitness = ensemble.solveProblem(EnsembleAlternationTest.problem_nonsmooth_multipeak)
        print(f"Best solution: {best_position}, Best fitness: {best_fitness}")
        self.assertTrue(best_fitness < 0.1)

    def test_ensemble_three_evr(self):
        epoch = 20
        pop_size = 50
        wf = 0.7
        cr = 0.9
        strategy = 0

        de = DE(epoch, pop_size, wf, cr, strategy)

        epoch = 20
        pop_size = 50

        hho = HHO(epoch, pop_size)

        n_limits = 50
        pop_size = 50
        epoch = 20

        abc = ABC(epoch, pop_size, n_limits)

        ensemble = MetaheuristicEnsembleAlternation(de, hho, abc)
        best_position, best_fitness = ensemble.solveProblem(EnsembleAlternationTest.problem_rosenbrook)
        print(f"Best solution: {best_position}, Best fitness: {best_fitness}")
        self.assertTrue(best_fitness < 0.1)
