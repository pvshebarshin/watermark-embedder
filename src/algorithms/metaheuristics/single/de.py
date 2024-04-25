import numpy as np
from src.optimizer import Optimizer


class DE(Optimizer):

    def __init__(self, epoch=10000, pop_size=100, wf=1.0, cr=0.9, strategy=0, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            wf (float): weighting factor, default = 1.5
            cr (float): crossover rate, default = 0.9
            strategy (int): Different variants of DE, default = 0
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.wf = self.validator.check_float("wf", wf, (0, 3.0))
        self.cr = self.validator.check_float("cr", cr, (0, 1.0))
        self.strategy = self.validator.check_int("strategy", strategy, [0, 5])
        self.set_parameters(["epoch", "pop_size", "wf", "cr", "strategy"])
        self.sort_flag = False

    def mutation__(self, current_pos, new_pos):
        condition = np.random.random(self.problem.n_dims) < self.cr
        pos_new = np.where(condition, new_pos, current_pos)
        return self.amend_position(pos_new, self.problem.lb, self.problem.ub)

    def evolve(self, epoch):

        pop = []
        if self.strategy == 0:
            # Choose 3 random element and different to i
            for idx in range(0, self.pop_size):
                idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
                pos_new = self.pop[idx_list[0]][self.ID_POS] + self.wf * \
                          (self.pop[idx_list[1]][self.ID_POS] - self.pop[idx_list[2]][self.ID_POS])
                pos_new = self.mutation__(self.pop[idx][self.ID_POS], pos_new)
                pop.append([pos_new, None])

                if self.mode not in self.AVAILABLE_MODES:
                    target = self.get_target_wrapper(pos_new)
                    self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])

        elif self.strategy == 1:

            for idx in range(0, self.pop_size):
                idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                pos_new = self.g_best[self.ID_POS] + self.wf * (
                            self.pop[idx_list[0]][self.ID_POS] - self.pop[idx_list[1]][self.ID_POS])
                pos_new = self.mutation__(self.pop[idx][self.ID_POS], pos_new)
                pop.append([pos_new, None])

                if self.mode not in self.AVAILABLE_MODES:
                    target = self.get_target_wrapper(pos_new)
                    self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])

        elif self.strategy == 2:

            for idx in range(0, self.pop_size):
                idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 4, replace=False)
                pos_new = self.g_best[self.ID_POS] + self.wf * (
                            self.pop[idx_list[0]][self.ID_POS] - self.pop[idx_list[1]][self.ID_POS]) + \
                          self.wf * (self.pop[idx_list[2]][self.ID_POS] - self.pop[idx_list[3]][self.ID_POS])
                pos_new = self.mutation__(self.pop[idx][self.ID_POS], pos_new)
                pop.append([pos_new, None])

                if self.mode not in self.AVAILABLE_MODES:
                    target = self.get_target_wrapper(pos_new)
                    self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])

        elif self.strategy == 3:

            for idx in range(0, self.pop_size):
                idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 5, replace=False)
                pos_new = self.pop[idx_list[0]][self.ID_POS] + self.wf * \
                          (self.pop[idx_list[1]][self.ID_POS] - self.pop[idx_list[2]][self.ID_POS]) + \
                          self.wf * (self.pop[idx_list[3]][self.ID_POS] - self.pop[idx_list[4]][self.ID_POS])
                pos_new = self.mutation__(self.pop[idx][self.ID_POS], pos_new)
                pop.append([pos_new, None])

                if self.mode not in self.AVAILABLE_MODES:
                    target = self.get_target_wrapper(pos_new)
                    self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])

        elif self.strategy == 4:

            for idx in range(0, self.pop_size):
                idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                pos_new = self.pop[idx][self.ID_POS] + self.wf * (
                            self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                          self.wf * (self.pop[idx_list[0]][self.ID_POS] - self.pop[idx_list[1]][self.ID_POS])
                pos_new = self.mutation__(self.pop[idx][self.ID_POS], pos_new)
                pop.append([pos_new, None])

                if self.mode not in self.AVAILABLE_MODES:
                    target = self.get_target_wrapper(pos_new)
                    self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])

        else:

            for idx in range(0, self.pop_size):
                idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
                pos_new = self.pop[idx][self.ID_POS] + self.wf * (
                            self.pop[idx_list[0]][self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                          self.wf * (self.pop[idx_list[1]][self.ID_POS] - self.pop[idx_list[2]][self.ID_POS])
                pos_new = self.mutation__(self.pop[idx][self.ID_POS], pos_new)
                pop.append([pos_new, None])

                if self.mode not in self.AVAILABLE_MODES:
                    target = self.get_target_wrapper(pos_new)
                    self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])

        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_wrapper_population(pop)
            self.pop = self.greedy_selection_population(self.pop, pop)
