import numpy as np
from copy import deepcopy
from src.optimizer import Optimizer


class PSO(Optimizer):
    ID_POS = 0
    ID_TAR = 1
    ID_VEC = 2  # Velocity
    ID_LOP = 3  # Local position
    ID_LOF = 4  # Local fitness

    def __init__(self,
                 epoch=10000,
                 pop_size=100,
                 c1=2.05,
                 c2=2.05,
                 w_min=0.4,
                 w_max=0.9,
                 **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c1 (float): [0-2] local coefficient
            c2 (float): [0-2] global coefficient
            w_min (float): Weight min of bird, default = 0.4
            w_max (float): Weight max of bird, default = 0.9
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.c1 = self.validator.check_float("c1", c1, (0, 5.0))
        self.c2 = self.validator.check_float("c2", c2, (0, 5.0))
        self.w_min = self.validator.check_float("w_min", w_min, (0, 0.5))
        self.w_max = self.validator.check_float("w_max", w_max, [0.5, 2.0])
        self.set_parameters(["epoch", "pop_size", "c1", "c2", "w_min", "w_max"])
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def initialize_variables(self):
        self.v_max = 0.5 * (self.problem.ub - self.problem.lb)
        self.v_min = -self.v_max

    def create_solution(self, lb=None, ub=None, pos=None):
        if pos is None:
            pos = self.generate_position(lb, ub)

        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        velocity = np.random.uniform(self.v_min, self.v_max)
        local_pos = deepcopy(position)
        local_fit = deepcopy(target)

        return [position, target, velocity, local_pos, local_fit]

    def amend_position(self, position=None, lb=None, ub=None):
        condition = np.logical_and(lb <= position, position <= ub)
        pos_rand = np.random.uniform(lb, ub)
        return np.where(condition, position, pos_rand)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Update weight after each move count  (weight down)
        w = (self.epoch - epoch) / self.epoch * (self.w_max - self.w_min) + self.w_min
        pop_new = []

        for idx in range(0, self.pop_size):
            agent = deepcopy(self.pop[idx])
            v_new = w * self.pop[idx][self.ID_VEC] + self.c1 * np.random.rand() * (
                        self.pop[idx][self.ID_LOP] - self.pop[idx][self.ID_POS]) + \
                    self.c2 * np.random.rand() * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            x_new = self.pop[idx][self.ID_POS] + v_new
            pos_new = self.amend_position(x_new, self.problem.lb, self.problem.ub)
            agent[self.ID_POS] = pos_new
            agent[self.ID_VEC] = v_new
            pop_new.append(agent)

            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)

        # Update current position, current velocity and compare with past position, past fitness (local best)
        for idx in range(0, self.pop_size):

            if self.compare_agent(pop_new[idx], self.pop[idx]):
                self.pop[idx] = deepcopy(pop_new[idx])

                if self.compare_agent(pop_new[idx], [None, self.pop[idx][self.ID_LOF]]):
                    self.pop[idx][self.ID_LOP] = deepcopy(pop_new[idx][self.ID_POS])
                    self.pop[idx][self.ID_LOF] = deepcopy(pop_new[idx][self.ID_TAR])
