import numpy as np

from src.optimizer import Optimizer


class IWO(Optimizer):

    def __init__(self,
                 epoch=10000,
                 pop_size=100,
                 seed_min=2,
                 seed_max=10,
                 exponent=2,
                 sigma_start=1.0,
                 sigma_end=0.01,
                 **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            seed_min (int): Number of Seeds (min)
            seed_max (int): Number of seeds (max)
            exponent (int): Variance Reduction Exponent
            sigma_start (float): The initial value of standard deviation
            sigma_end (float): The final value of standard deviation
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.seed_min = self.validator.check_int("seed_min", seed_min, [1, 3])
        self.seed_max = self.validator.check_int("seed_max", seed_max, [4, int(self.pop_size / 2)])
        self.exponent = self.validator.check_int("exponent", exponent, [2, 4])
        self.sigma_start = self.validator.check_float("sigma_start", sigma_start, [0.5, 5.0])
        self.sigma_end = self.validator.check_float("sigma_end", sigma_end, (0, 0.5))
        self.set_parameters(["epoch", "pop_size", "seed_min", "seed_max", "exponent", "sigma_start", "sigma_end"])
        self.sort_flag = True

    def evolve(self, epoch=None):
        sigma = ((self.epoch - epoch) / (self.epoch - 1)) ** self.exponent * (
                self.sigma_start - self.sigma_end) + self.sigma_end
        pop, best, worst = self.get_special_solutions(self.pop)

        pop_new = []
        for idx in range(0, self.pop_size):
            temp = best[0][self.ID_TAR][self.ID_FIT] - worst[0][self.ID_TAR][self.ID_FIT]

            if temp == 0:
                ratio = np.random.rand()
            else:
                ratio = (pop[idx][self.ID_TAR][self.ID_FIT] - worst[0][self.ID_TAR][self.ID_FIT]) / temp

            s = int(np.ceil(self.seed_min + (self.seed_max - self.seed_min) * ratio))

            if s > int(np.sqrt(self.pop_size)):
                s = int(np.sqrt(self.pop_size))

            pop_local = []
            for j in range(s):
                new_position = pop[idx][self.ID_POS] + sigma * np.random.normal(0, 1, self.problem.n_dims)
                new_position = self.amend_position(new_position, self.problem.lb, self.problem.ub)
                pop_local.append([new_position, None])

                if self.mode not in self.AVAILABLE_MODES:
                    pop_local[-1][self.ID_TAR] = self.get_target_wrapper(new_position)

            if self.mode in self.AVAILABLE_MODES:
                pop_local = self.update_target_wrapper_population(pop_local)

            pop_new += pop_local
        self.pop = self.get_sorted_strim_population(pop_new, self.pop_size)
