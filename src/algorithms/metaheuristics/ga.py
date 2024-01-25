import numpy as np

from src.optimizer import Optimizer, crossover_arithmetic


class GA(Optimizer):

    def __init__(self, epoch=10000, pop_size=100, pc=0.95, pm=0.025,
                 selection="roulette", crossover="arithmetic", mutation="flip", k_way=0.2, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pc (float): cross-over probability, default = 0.95
            pm (float): mutation probability, default = 0.025
            selection (str): Optional, can be ["roulette", "tournament", "random"], default = "tournament"
            crossover (str): Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
            mutation (str): Optional, can be ["flip", "swap"] for multipoints
            k_way (float): Optional, set it when use "tournament" selection, default = 0.2
        """
        super().__init__(**kwargs)

        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.pc = self.validator.check_float("pc", pc, (0, 1.0))
        self.pm = self.validator.check_float("pm", pm, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "pc", "pm"])
        self.sort_flag = False
        self.selection = "tournament"
        self.k_way = 0.2
        self.crossover = "uniform"
        self.mutation = "flip"
        self.mutation_multipoints = True

        if "selection" in kwargs:
            self.selection = self.validator.check_str("selection", kwargs["selection"],
                                                      ["tournament", "random", "roulette"])
        if "k_way" in kwargs:
            self.k_way = self.validator.check_float("k_way", kwargs["k_way"], (0, 1.0))
        if "crossover" in kwargs:
            self.crossover = self.validator.check_str("crossover", kwargs["crossover"],
                                                      ["one_point", "multi_points", "uniform", "arithmetic"])
        if "mutation_multipoints" in kwargs:
            self.mutation_multipoints = self.validator.check_bool("mutation_multipoints",
                                                                  kwargs["mutation_multipoints"])
        if self.mutation_multipoints:
            if "mutation" in kwargs:
                self.mutation = self.validator.check_str("mutation", kwargs["mutation"], ["flip", "swap"])
        else:
            if "mutation" in kwargs:
                self.mutation = self.validator.check_str("mutation", kwargs["mutation"],
                                                         ["flip", "swap", "scramble", "inversion"])

        self.selection = self.validator.check_str("selection", selection, ["tournament", "random", "roulette"])
        self.crossover = self.validator.check_str("crossover", crossover, ["one_point", "multi_points",
                                                                           "uniform", "arithmetic"])
        self.mutation = self.validator.check_str("mutation", mutation, ["flip", "swap"])
        self.k_way = self.validator.check_float("k_way", k_way, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "pc", "pm", "selection", "crossover", "mutation", "k_way"])

    def selection_process__(self, list_fitness):
        if self.selection == "roulette":
            id_c1 = self.get_index_roulette_wheel_selection(list_fitness)
            id_c2 = self.get_index_roulette_wheel_selection(list_fitness)
        elif self.selection == "random":
            id_c1, id_c2 = np.random.choice(range(self.pop_size), 2, replace=False)
        else:  ## tournament
            id_c1, id_c2 = self.get_index_kway_tournament_selection(self.pop, k_way=self.k_way, output=2)
        return self.pop[id_c1][self.ID_POS], self.pop[id_c2][self.ID_POS]

    def selection_process_00__(self, pop_selected):
        if self.selection == "roulette":
            list_fitness = np.array([agent[self.ID_TAR][self.ID_FIT] for agent in pop_selected])
            id_c1 = self.get_index_roulette_wheel_selection(list_fitness)
            id_c2 = self.get_index_roulette_wheel_selection(list_fitness)
        elif self.selection == "random":
            id_c1, id_c2 = np.random.choice(range(len(pop_selected)), 2, replace=False)
        else:
            id_c1, id_c2 = self.get_index_kway_tournament_selection(pop_selected, k_way=self.k_way, output=2)
        return pop_selected[id_c1][self.ID_POS], pop_selected[id_c2][self.ID_POS]

    def selection_process_01__(self, pop_dad, pop_mom):
        if self.selection == "roulette":
            list_fit_dad = np.array([agent[self.ID_TAR][self.ID_FIT] for agent in pop_dad])
            list_fit_mom = np.array([agent[self.ID_TAR][self.ID_FIT] for agent in pop_mom])
            id_c1 = self.get_index_roulette_wheel_selection(list_fit_dad)
            id_c2 = self.get_index_roulette_wheel_selection(list_fit_mom)

        elif self.selection == "random":
            id_c1 = np.random.choice(range(len(pop_dad)))
            id_c2 = np.random.choice(range(len(pop_mom)))

        else:
            id_c1 = self.get_index_kway_tournament_selection(pop_dad, k_way=self.k_way, output=1)[0]
            id_c2 = self.get_index_kway_tournament_selection(pop_mom, k_way=self.k_way, output=1)[0]
        return pop_dad[id_c1][self.ID_POS], pop_mom[id_c2][self.ID_POS]

    def crossover_process__(self, dad, mom):
        if self.crossover == "arithmetic":
            w1, w2 = crossover_arithmetic(dad, mom)
        elif self.crossover == "one_point":
            cut = np.random.randint(1, self.problem.n_dims - 1)
            w1 = np.concatenate([dad[:cut], mom[cut:]])
            w2 = np.concatenate([mom[:cut], dad[cut:]])
        elif self.crossover == "multi_points":
            indexes = np.random.choice(range(1, self.problem.n_dims - 1), 2, replace=False)
            cut1, cut2 = np.min(indexes), np.max(indexes)
            w1 = np.concatenate([dad[:cut1], mom[cut1:cut2], dad[cut2:]])
            w2 = np.concatenate([mom[:cut1], dad[cut1:cut2], mom[cut2:]])
        else:
            flip = np.random.randint(0, 2, self.problem.n_dims)
            w1 = dad * flip + mom * (1 - flip)
            w2 = mom * flip + dad * (1 - flip)
        return w1, w2

    def mutation_process__(self, child):
        if self.mutation == "swap":
            for i in range(self.problem.n_dims):
                idx_swap = np.random.choice(list(set(range(0, self.problem.n_dims)) - {i}))
                child[i], child[idx_swap] = child[idx_swap], child[i]
                return child
        else:
            mutation_child = self.generate_position(self.problem.lb, self.problem.ub)
            flag_child = np.random.uniform(0, 1, self.problem.n_dims) < self.pm
            return np.where(flag_child, mutation_child, child)

    def survivor_process__(self, pop, pop_child):
        pop_new = []
        for i in range(0, self.pop_size):
            child_id = self.get_index_kway_tournament_selection(pop, k_way=0.1, output=1, reverse=True)[0]
            pop_new.append(self.get_better_solution(pop_child[i], pop[child_id]))
        return pop_new

    def evolve(self, epoch):
        list_fitness = np.array([agent[self.ID_TAR][self.ID_FIT] for agent in self.pop])
        pop_new = []
        for i in range(0, int(self.pop_size / 2)):
            child1, child2 = self.selection_process__(list_fitness)

            if np.random.uniform() < self.pc:
                child1, child2 = self.crossover_process__(child1, child2)

            child1 = self.mutation_process__(child1)
            child2 = self.mutation_process__(child2)

            pop_new.append([self.amend_position(child1, self.problem.lb, self.problem.ub), None])
            pop_new.append([self.amend_position(child2, self.problem.lb, self.problem.ub), None])

            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-2][self.ID_TAR] = self.get_target_wrapper(child1)
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(child2)

        pop_new = self.update_target_wrapper_population(pop_new)
        self.pop = self.survivor_process__(self.pop, pop_new)
