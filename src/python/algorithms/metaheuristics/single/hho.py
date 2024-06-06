import numpy as np
from math import gamma
from copy import deepcopy

from python.optimizer import Optimizer


class HHO(Optimizer):

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
        self.sort_flag = False

    def evolve(self, epoch):
        pop_new = []
        for idx in range(0, self.pop_size):
            e0 = 2 * np.random.uniform() - 1
            e = 2 * e0 * (1 - (epoch + 1) * 1.0 / self.epoch)
            j = 2 * (1 - np.random.uniform())

            if np.abs(e) >= 1:
                if np.random.rand() >= 0.5:
                    x_rand = deepcopy(self.pop[np.random.randint(0, self.pop_size)][self.ID_POS])
                    pos_new = x_rand - np.random.uniform() * np.abs(x_rand - 2 * np.random.uniform()
                                                                    * self.pop[idx][self.ID_POS])

                else:
                    x_m = np.mean([x[self.ID_POS] for x in self.pop])
                    pos_new = (self.g_best[self.ID_POS] - x_m) - np.random.uniform() * \
                              (self.problem.lb + np.random.uniform() * (self.problem.ub - self.problem.lb))

                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                pop_new.append([pos_new, None])
            else:
                if np.random.rand() >= 0.5:
                    delta_x = self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]

                    if np.abs(e) >= 0.5:
                        pos_new = delta_x - e * np.abs(j * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                    else:
                        pos_new = self.g_best[self.ID_POS] - e * np.abs(delta_x)
                    pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                    pop_new.append([pos_new, None])
                else:
                    xichma = np.power((gamma(1 + 1.5) * np.sin(np.pi * 1.5 / 2.0)) /
                                      (gamma((1 + 1.5) * 1.5 * np.power(2, (1.5 - 1) / 2))
                                       / 2.0), 1.0 / 1.5)
                    lf_d = 0.01 * np.random.uniform() * xichma / np.power(np.abs(np.random.uniform()), 1.0 / 1.5)
                    if np.abs(e) >= 0.5:
                        y = self.g_best[self.ID_POS] - e * np.abs(j * self.g_best[self.ID_POS]
                                                                  - self.pop[idx][self.ID_POS])
                    else:
                        x_m = np.mean([x[self.ID_POS] for x in self.pop])
                        y = self.g_best[self.ID_POS] - e * np.abs(j * self.g_best[self.ID_POS] - x_m)

                    pos_y = self.amend_position(y, self.problem.lb, self.problem.ub)
                    target_y = self.get_target_wrapper(pos_y)
                    z = y + np.random.uniform(self.problem.lb, self.problem.ub) * lf_d
                    pos_z = self.amend_position(z, self.problem.lb, self.problem.ub)
                    target_z = self.get_target_wrapper(pos_z)

                    if self.compare_agent([pos_y, target_y], self.pop[idx]):
                        pop_new.append([pos_y, target_y])
                        continue

                    if self.compare_agent([pos_z, target_z], self.pop[idx]):
                        pop_new.append([pos_z, target_z])
                        continue
                    pop_new.append(deepcopy(self.pop[idx]))

        if self.mode not in self.AVAILABLE_MODES:
            for idx, agent in enumerate(pop_new):
                pop_new[idx][self.ID_TAR] = self.get_target_wrapper(agent[self.ID_POS])
        else:
            pop_new = self.update_target_wrapper_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)
