import numpy as np

from python.optimizer import Optimizer


class CA(Optimizer):

    def __init__(self, epoch=10000, pop_size=100, accepted_rate=0.15, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations
            pop_size (int): number of population size
            accepted_rate (float): probability of accepted rate
        """
        super().__init__(**kwargs)

        self.dyn_accepted_num = None
        self.dyn_belief_space = None
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.accepted_rate = self.validator.check_float("accepted_rate", accepted_rate, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "accepted_rate"])
        self.support_parallel_modes = False
        self.sort_flag = True

    def initialize_variables(self):
        self.dyn_belief_space = {
            "lb": self.problem.lb,
            "ub": self.problem.ub,
        }
        self.dyn_accepted_num = int(self.accepted_rate * self.pop_size)

    def create_faithful__(self, lb, ub):
        position = self.generate_position(lb, ub)
        position = self.amend_position(position, lb, ub)
        target = self.get_target_wrapper(position)
        return [position, target]

    def update_belief_space__(self, belief_space, pop_accepted):
        pos_list = np.array([solution[self.ID_POS] for solution in pop_accepted])
        belief_space["lb"] = np.min(pos_list, axis=0)
        belief_space["ub"] = np.max(pos_list, axis=0)
        return belief_space

    def evolve(self, epoch):
        pop_child = [self.create_faithful__(self.dyn_belief_space["lb"],
                                            self.dyn_belief_space["ub"]) for _ in range(0, self.pop_size)]

        new_position = []
        pop_full = self.pop + pop_child
        size_new = len(pop_full)
        for _ in range(0, self.pop_size):
            id1, id2 = np.random.choice(list(range(0, size_new)), 2, replace=False)
            new_position.append(self.get_better_solution(pop_full[id1], pop_full[id2]))

        self.pop = self.get_sorted_strim_population(new_position)
        accepted = self.pop[:self.dyn_accepted_num]
        self.dyn_belief_space = self.update_belief_space__(self.dyn_belief_space, accepted)
