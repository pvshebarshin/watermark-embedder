import numpy as np
from copy import deepcopy

from python.optimizer import Optimizer


class BBO(Optimizer):

    def __init__(self, epoch=10000, pop_size=100, p_m=0.01, elites=2, **kwargs):
        """
        Args:
            epoch (int): Maximum number of iterations
            pop_size (int): Number of population size
            p_m (float): Mutation probability
            elites (int): Number of elites will be keep for next generation
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.p_m = self.validator.check_float("p_m", p_m, (0, 1.0))
        self.elites = self.validator.check_int("elites", elites, [2, int(self.pop_size / 2)])
        self.set_parameters(["epoch", "pop_size", "p_m", "elites"])
        self.sort_flag = False
        self.mu = (self.pop_size + 1 - np.array(range(1, self.pop_size + 1))) / (self.pop_size + 1)
        self.mr = 1 - self.mu

    def evolve(self, epoch):
        _, pop_elites, _ = self.get_special_solutions(self.pop, best=self.elites)
        pop = []

        for i in range(0, self.pop_size):
            new_position = deepcopy(self.pop[i][self.ID_POS])

            for j in range(self.problem.n_dims):
                if np.random.uniform() < self.mr[i]:
                    random_number = np.random.uniform() * np.sum(self.mu)
                    select = self.mu[0]
                    select_index = 0

                    while (random_number > select) and (select_index < self.pop_size - 1):
                        select_index += 1
                        select += self.mu[select_index]

                    new_position[j] = self.pop[select_index][self.ID_POS][j]

            noise = np.random.uniform(self.problem.lb, self.problem.ub)
            condition = np.random.random(self.problem.n_dims) < self.p_m
            new_position = np.where(condition, noise, new_position)
            new_position = self.amend_position(new_position, self.problem.lb, self.problem.ub)
            pop.append([new_position, None])

            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(new_position)
                self.pop[i] = self.get_better_solution([new_position, target], self.pop[i])

        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_wrapper_population(pop)
            self.pop = self.greedy_selection_population(self.pop, pop)

        self.pop = self.get_sorted_strim_population(self.pop + pop_elites, self.pop_size)
