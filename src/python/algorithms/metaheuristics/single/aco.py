import numpy as np
from python.optimizer import Optimizer


class ACO(Optimizer):

    def __init__(self, epoch=10000, pop_size=100, sample_count=25, intent_factor=0.5, zeta=1.0, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations
            pop_size (int): number of population size
            sample_count (int): Number of Newly Generated Samples
            intent_factor (float): Intensification Factor (Selection Pressure) (q in the paper)
            zeta (float): Deviation-Distance Ratio
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.sample_count = self.validator.check_int("sample_count", sample_count, [2, 10000])
        self.intent_factor = self.validator.check_float("intent_factor", intent_factor, (0, 1.0))
        self.zeta = self.validator.check_float("zeta", zeta, (0, 5))
        self.set_parameters(["epoch", "pop_size", "sample_count", "intent_factor", "zeta"])
        self.sort_flag = True

    def evolve(self, epoch):
        pop_rank = np.array([i for i in range(1, self.pop_size + 1)])
        qn = self.intent_factor * self.pop_size
        matrix_w = 1 / (np.sqrt(2 * np.pi) * qn) * np.exp(-0.5 * ((pop_rank - 1) / qn) ** 2)
        matrix_p = matrix_w / np.sum(matrix_w)

        matrix_position = np.array([solution[self.ID_POS] for solution in self.pop])
        matrix_sigma = []

        for i in range(0, self.pop_size):
            matrix_index = np.repeat(self.pop[i][self.ID_POS].reshape((1, -1)), self.pop_size, axis=0)
            d = np.sum(np.abs(matrix_position - matrix_index), axis=0)
            temp = self.zeta * d / (self.pop_size - 1)
            matrix_sigma.append(temp)

        matrix_sigma = np.array(matrix_sigma)

        pop_new = []
        for i in range(0, self.sample_count):
            child = np.zeros(self.problem.n_dims)

            for j in range(0, self.problem.n_dims):
                idx = self.get_index_roulette_wheel_selection(matrix_p)
                child[j] = self.pop[idx][self.ID_POS][j] + np.random.normal() * matrix_sigma[idx, j]

            new_position = self.amend_position(child, self.problem.lb, self.problem.ub)
            pop_new.append([new_position, None])

            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(new_position)

        pop_new = self.update_target_wrapper_population(pop_new)
        self.pop = self.get_sorted_strim_population(self.pop + pop_new, self.pop_size)
