import random

import numpy as np


def switchModel(model, problem, starting_positions=None):
    if starting_positions is not None:
        return model.solve(problem)
    else:
        return model.solve(problem, starting_positions=starting_positions)


class MetaheuristicEnsembleAlternation:

    def __init__(self, model1, model2=None, model3=None, epochs=60, step=5):
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.epochs = epochs
        self.step = step

    def solveProblem(self, problem):
        if self.model3 is not None:
            if self.model2 is not None:
                choices = [self.model1, self.model2, self.model3]
                return self.switchEvr(choices, problem)
            else:
                return self.switchOneEvr(problem)
        else:
            if self.model2 is not None:
                choices = [self.model1, self.model2]
                return self.switchEvr(choices, problem)
            else:
                return self.switchOneEvr(problem)

    def switchOneEvr(self, problem):
        best_position, best_fitness = switchModel(self.model1, problem)
        epoch = self.epochs
        epoch -= self.step
        while epoch > 0:
            best_position, best_fitness = switchModel(self.model1, problem, starting_positions=[
                best_position + np.random.uniform(-1, 1) * 0.05 for _ in range(self.model1.pop_size)])
            epoch -= self.step
        return best_position, best_fitness

    def switchEvr(self, choices, problem):
        best_position, best_fitness = switchModel(random.choice(choices), problem)
        epoch = self.epochs
        epoch -= self.step
        while epoch > 0:
            best_position, best_fitness = switchModel(random.choice(choices), problem, starting_positions=[
                best_position + np.random.uniform(-1, 1) * 0.05 for _ in range(self.model1.pop_size)])
            epoch -= self.step
        return best_position, best_fitness
