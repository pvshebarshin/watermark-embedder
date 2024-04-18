import random

import numpy as np


def rnd(choices):
    return random.choice(choices)


class MetaheuristicEnsembleLearn:

    def __init__(self, model1, model2=None, model3=None):
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

    def switchModel3(self, problem, choices, starting_positions=None):
        term_dict = {
            "max_time": 20  # 60 seconds to run this algorithm only
        }
        dice = rnd(choices)
        if starting_positions is not None:
            if dice == 1:
                choices.remove(dice)
                return self.model1.solve(problem, termination=term_dict)
            elif dice == 2:
                choices.remove(dice)
                return self.model2.solve(problem)
            else:
                choices.remove(dice)
                return self.model3.solve(problem)
        else:
            if dice == 1:
                choices.remove(dice)
                return self.model1.solve(problem, starting_positions=starting_positions, termination=term_dict)
            elif dice == 2:
                choices.remove(dice)
                return self.model2.solve(problem, starting_positions=starting_positions)
            else:
                return self.model3.solve(problem, starting_positions=starting_positions)

    def solveProblem(self, problem):
        if self.model3 is not None:
            if self.model2 is not None:
                choices = [1, 2, 3]
                best_position, best_fitness = self.switchModel3(problem, choices)
                best_position, best_fitness = self.switchModel3(problem, choices, starting_positions=[
                    best_position + np.random.uniform(-1, 1) * 0.1 for _ in range(self.model1.pop_size)])
                best_position, best_fitness = self.switchModel3(problem, choices, starting_positions=[
                    best_position + np.random.uniform(-1, 1) * 0.01 for _ in range(self.model1.pop_size)])
                return best_position, best_fitness
            else:
                best_position, best_fitness = self.switchModel3(problem, [1])
                return best_position, best_fitness

        else:

            if self.model2 is not None:
                choices = [1, 2]
                best_position, best_fitness = self.switchModel3(problem, choices)
                best_position, best_fitness = self.switchModel3(problem, choices, starting_positions=[
                    best_position + np.random.uniform(-1, 1) * 0.1 for _ in range(self.model1.pop_size)])
                return best_position, best_fitness
            else:
                best_position, best_fitness = self.switchModel3(problem, [1])
                return best_position, best_fitness
