import numpy as np
from python.optimizer import Optimizer


class GBO(Optimizer):

    def __init__(self, epoch=10000, pop_size=100, pr=0.5, beta_min=0.2, beta_max=1.2, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pr (float): Probability Parameter, default = 0.5
            beta_min (float): Fixed parameter, default = 0.2
            beta_max (float): Fixed parameter, default = 1.2
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.pr = self.validator.check_float("pr", pr, (0, 1.0))
        self.beta_min = self.validator.check_float("beta_min", beta_min, (0, 2.0))
        self.beta_max = self.validator.check_float("beta_max", beta_max, (0, 5.0))
        self.set_parameters(["epoch", "pop_size", "pr", "beta_min", "beta_max"])

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def evolve(self, epoch):

        beta = self.beta_min + (self.beta_max - self.beta_min) * (1 - ((epoch + 1) / self.epoch) ** 3) ** 2
        alpha = np.abs(beta * np.sin(3 * np.pi / 2 + np.sin(beta * 3 * np.pi / 2)))

        pop_new = []
        for idx in range(0, self.pop_size):
            p1 = 2 * np.random.rand() * alpha - alpha
            p2 = 2 * np.random.rand() * alpha - alpha
            #  Four positions randomly selected from population
            r1, r2, r3, r4 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 4, replace=False)
            # Average of Four positions randomly selected from population
            r0 = (self.pop[r1][self.ID_POS] + self.pop[r2][self.ID_POS] + self.pop[r3][self.ID_POS] + self.pop[r4][
                self.ID_POS]) / 4
            # Randomization Epsilon
            epsilon = 5e-3 * np.random.rand()

            delta = 2 * np.random.rand() * np.abs(r0 - self.pop[idx][self.ID_POS])
            step = (self.g_best[self.ID_POS] - self.pop[r1][self.ID_POS] + delta) / 2
            delta_x = np.random.choice(range(0, self.pop_size)) * np.abs(step)

            x1 = self.pop[idx][self.ID_POS] - np.random.normal() * p1 * 2 * delta_x * \
                 self.pop[idx][self.ID_POS] / (self.g_worst[self.ID_POS] - self.g_best[self.ID_POS] + epsilon) + \
                 np.random.rand() * p2 * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])

            z = self.pop[idx][self.ID_POS] - np.random.normal() * 2 * delta_x * \
                self.pop[idx][self.ID_POS] / (self.g_worst[self.ID_POS] - self.g_best[self.ID_POS] + epsilon)

            y_p = np.random.rand() * ((z + self.pop[idx][self.ID_POS]) / 2 + np.random.rand() * delta_x)
            y_q = np.random.rand() * ((z + self.pop[idx][self.ID_POS]) / 2 - np.random.rand() * delta_x)
            x2 = self.g_best[self.ID_POS] - np.random.normal() * p1 * 2 * delta_x * self.pop[idx][self.ID_POS] / (
                    y_p - y_q + epsilon) + \
                 np.random.rand() * p2 * (self.pop[r1][self.ID_POS] - self.pop[r2][self.ID_POS])

            x3 = self.pop[idx][self.ID_POS] - p1 * (x2 - x1)
            ra = np.random.rand()
            rb = np.random.rand()
            pos_new = ra * (rb * x1 + (1 - rb) * x2) + (1 - ra) * x3

            # Local escaping operator
            if np.random.rand() < self.pr:
                f1 = np.random.uniform(-1, 1)
                f2 = np.random.normal(0, 1)
                L1 = np.round(1 - np.random.rand())
                u1 = L1 * 2 * np.random.rand() + (1 - L1)
                u2 = L1 * np.random.rand() + (1 - L1)
                u3 = L1 * np.random.rand() + (1 - L1)

                L2 = np.round(1 - np.random.rand())
                x_rand = self.generate_position(self.problem.lb, self.problem.ub)
                x_p = self.pop[np.random.choice(range(0, self.pop_size))][self.ID_POS]
                x_m = L2 * x_p + (1 - L2) * x_rand

                if np.random.rand() < 0.5:
                    pos_new = pos_new + f1 * (u1 * self.g_best[self.ID_POS] - u2 * x_m) + \
                              f2 * p1 * (u3 * (x2 - x1) + u2 * (
                            self.pop[r1][self.ID_POS] - self.pop[r2][self.ID_POS])) / 2
                else:
                    pos_new = self.g_best[self.ID_POS] + f1 * (u1 * self.g_best[self.ID_POS] - u2 * x_m) + f2 * p1 * (
                            u3 * (x2 - x1) + u2 * (self.pop[r1][self.ID_POS] - self.pop[r2][self.ID_POS])) / 2

            # Check if solutions go outside the search space and bring them back
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])

            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(pop_new, self.pop)

        self.nfe_per_epoch = self.pop_size
        _, best, worst = self.get_special_solutions(self.pop, best=1, worst=1)
        self.g_best, self.g_worst = best[0], worst[0]
