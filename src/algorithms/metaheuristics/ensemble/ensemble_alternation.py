import random

import numpy as np


class MetaheuristicEnsembleAlternation:

    def __init__(self, model1, model2=None, model3=None, epochs=60, step=5, term_dict=None):
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.epochs = epochs
        self.step = step
        self.term_dict = term_dict

    def switchModel(self, model, problem, starting_positions=None):
        return model.solve(problem, starting_positions=starting_positions, termination=self.term_dict)

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
        best_position, best_fitness = self.switchModel(self.model1, problem)
        epoch = self.epochs
        epoch -= self.step
        while epoch > 0:
            best_position, best_fitness = self.switchModel(self.model1, problem, starting_positions=[
                best_position + np.random.uniform(-1, 1) * 0.05 for _ in range(self.model1.pop_size)])
            epoch -= self.step
        return best_position, best_fitness

    def switchEvr(self, choices, problem):
        best_position, best_fitness = self.switchModel(random.choice(choices), problem)
        epoch = self.epochs
        epoch -= self.step
        while epoch > 0:
            best_position, best_fitness = self.switchModel(random.choice(choices), problem, starting_positions=[
                best_position + np.random.uniform(-1, 1) * 0.05 for _ in range(self.model1.pop_size)])
            epoch -= self.step
        return best_position, best_fitness
