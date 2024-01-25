import numpy as np

from src.optimizer import Optimizer


class ABC(Optimizer):

    def __init__(self, epoch=10000, pop_size=100, n_limits=25, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations
            pop_size (int): number of population size = onlooker bees = employed bees
            n_limits (int): Limit of trials before abandoning a food source
        """
        super().__init__(**kwargs)
        self.trials = None
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.n_limits = self.validator.check_int("n_limits", n_limits, [1, 1000])
        self.support_parallel_modes = False
        self.set_parameters(["epoch", "pop_size", "n_limits"])
        self.sort_flag = False

    def initialize_variables(self):
        self.trials = np.zeros(self.pop_size)

    def evolve(self, epoch):
        for idx in range(0, self.pop_size):
            t = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
            self.execute(idx, t)

        employed_fits = np.array([agent[self.ID_TAR][self.ID_FIT] for agent in self.pop])

        for idx in range(0, self.pop_size):
            selected_bee = self.get_index_roulette_wheel_selection(employed_fits)
            t = np.random.choice(list(set(range(0, self.pop_size)) - {idx, selected_bee}))
            self.execute(selected_bee, t)

        abandoned = np.where(self.trials >= self.n_limits)[0]
        for idx in abandoned:
            self.pop[idx] = self.create_solution(self.problem.lb, self.problem.ub)
            self.trials[idx] = 0

    def execute(self, idx, t):
        phi = np.random.uniform(low=-1, high=1, size=self.problem.n_dims)
        pos_new = self.pop[idx][self.ID_POS] + phi * (self.pop[t][self.ID_POS] - self.pop[idx][self.ID_POS])
        pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
        target = self.get_target_wrapper(pos_new)
        if self.compare_agent([pos_new, target], self.pop[idx]):
            self.pop[idx] = [pos_new, target]
            self.trials[idx] = 0
        else:
            self.trials[idx] += 1
