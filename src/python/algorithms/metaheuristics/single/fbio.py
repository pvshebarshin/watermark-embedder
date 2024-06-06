import numpy as np
from copy import deepcopy
from python.optimizer import Optimizer


class FBIO(Optimizer):

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])

        self.nfe_per_epoch = 4 * self.pop_size
        self.sort_flag = False

    def probability__(self, list_fitness=None):
        max1 = np.max(list_fitness)
        min1 = np.min(list_fitness)
        return (max1 - list_fitness) / (max1 - min1 + self.EPSILON)

    def amend_position(self, position=None, lb=None, ub=None):
        rand_pos = np.random.uniform(lb, ub)
        condition = np.logical_and(lb <= position, position <= ub)
        return np.where(condition, position, rand_pos)

    def evolve(self, epoch):
        # Step A1
        pop_new = []
        for idx in range(0, self.pop_size):
            n_change = np.random.randint(0, self.problem.n_dims)
            nb1, nb2 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)

            pos_a = deepcopy(self.pop[idx][self.ID_POS])
            pos_a[n_change] = self.pop[idx][self.ID_POS][n_change] + (np.random.uniform() - 0.5) * 2 * \
                              (self.pop[idx][self.ID_POS][n_change] - (
                                      self.pop[nb1][self.ID_POS][n_change] + self.pop[nb2][self.ID_POS][n_change]) / 2)

            pos_a = self.amend_position(pos_a, self.problem.lb, self.problem.ub)
            pop_new.append([pos_a, None])

            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_a)
                self.pop[idx] = self.get_better_solution([pos_a, target], self.pop[idx])

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)

        # Step A2
        list_fitness = np.array([item[self.ID_TAR][self.ID_FIT] for item in self.pop])
        prob = self.probability__(list_fitness)
        pop_child = []

        for idx in range(0, self.pop_size):
            if np.random.uniform() > prob[idx]:
                r1, r2, r3 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
                pos_a = deepcopy(self.pop[idx][self.ID_POS])
                rnd = np.floor(np.random.uniform() * self.problem.n_dims) + 1

                for j in range(0, self.problem.n_dims):
                    if np.random.uniform() < np.random.uniform() or rnd == j:
                        pos_a[j] = self.g_best[self.ID_POS][j] + self.pop[r1][self.ID_POS][j] + \
                                   np.random.uniform() * (self.pop[r2][self.ID_POS][j] - self.pop[r3][self.ID_POS][j])

            else:

                pos_a = np.random.uniform(self.problem.lb, self.problem.ub)

            pos_a = self.amend_position(pos_a, self.problem.lb, self.problem.ub)
            pop_child.append([pos_a, None])

            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_a)
                self.pop[idx] = self.get_better_solution([pos_a, target], self.pop[idx])

        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_wrapper_population(pop_child)
            self.pop = self.greedy_selection_population(pop_child, self.pop)

        ## Step B1
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_b = deepcopy(self.pop[idx][self.ID_POS])

            for j in range(0, self.problem.n_dims):
                pos_b[j] = np.random.uniform() * self.pop[idx][self.ID_POS][j] + \
                           np.random.uniform() * (self.g_best[self.ID_POS][j] - self.pop[idx][self.ID_POS][j])

            pos_b = self.amend_position(pos_b, self.problem.lb, self.problem.ub)
            pop_new.append([pos_b, None])

            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_b)
                self.pop[idx] = self.get_better_solution([pos_b, target], self.pop[idx])

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)

        ## Step B2
        pop_child = []
        for idx in range(0, self.pop_size):

            rr = np.random.randint(0, self.pop_size)
            while rr == idx:
                rr = np.random.randint(0, self.pop_size)
            if self.compare_agent(self.pop[idx], self.pop[rr]):

                pos_b = self.pop[idx][self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * (
                        self.pop[rr][self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                        np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[rr][self.ID_POS])
            else:

                pos_b = self.pop[idx][self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * (
                        self.pop[idx][self.ID_POS] - self.pop[rr][self.ID_POS]) + \
                        np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])

            pos_b = self.amend_position(pos_b, self.problem.lb, self.problem.ub)
            pop_child.append([pos_b, None])

            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_b)
                self.pop[idx] = self.get_better_solution([pos_b, target], self.pop[idx])

        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_wrapper_population(pop_child)
            self.pop = self.greedy_selection_population(pop_child, self.pop)
