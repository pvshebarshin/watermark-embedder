import numpy as np
from copy import deepcopy
from mealpy.utils.history import History
from mealpy.utils.problem import Problem
from mealpy.utils.termination import Termination
from mealpy.utils.logger import Logger
from mealpy.utils.validator import Validator
import concurrent.futures as parallel
from functools import partial
import os
import time


def bounded_position(position=None, lb=None, ub=None):
    return np.clip(position, lb, ub)


def crossover_arithmetic(dad_pos=None, mom_pos=None):
    r = np.random.uniform()  # w1 = w2 when r =0.5
    w1 = np.multiply(r, dad_pos) + np.multiply((1 - r), mom_pos)
    w2 = np.multiply(r, mom_pos) + np.multiply((1 - r), dad_pos)
    return w1, w2


class Optimizer:
    ID_POS = 0  # Index of position/location of solution/agent
    ID_TAR = 1  # Index of target list, (includes fitness value and objectives list)
    ID_FIT = 0  # Index of target (the final fitness) in fitness
    ID_OBJ = 1  # Index of objective list in target
    EPSILON = 10E-10

    def __init__(self, **kwargs):
        super(Optimizer, self).__init__()
        self.generate_position = None
        self.epoch, self.pop_size, self.solution = None, None, None
        self.mode, self.n_workers, self.name = None, None, None
        self.pop, self.g_best, self.g_worst = None, None, None
        self.problem, self.logger, self.history = None, None, None
        self.__set_keyword_arguments(kwargs)
        self.validator = Validator(log_to="console", log_file=None)

        if self.name is None: self.name = self.__class__.__name__
        self.sort_flag = False
        self.nfe_counter = -1  # The first one is tested in Problem class
        self.parameters, self.params_name_ordered = {}, None
        self.AVAILABLE_MODES = ["process", "thread", "swarm"]
        self.support_parallel_modes = True

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_parameters(self, parameters):
        if type(parameters) in (list, tuple):
            self.params_name_ordered = tuple(parameters)
            self.parameters = {}
            for name in parameters:
                self.parameters[name] = self.__dict__[name]

        if type(parameters) is dict:
            valid_para_names = set(self.parameters.keys())
            new_para_names = set(parameters.keys())
            if new_para_names.issubset(valid_para_names):
                for key, value in parameters.items():
                    setattr(self, key, value)
                    self.parameters[key] = value
            else:
                raise ValueError(f"Invalid input parameters: {new_para_names} for {self.get_name()} optimizer. "
                                 f"Valid parameters are: {valid_para_names}.")

    def get_name(self):
        return self.name

    def __str__(self):
        temp = ""
        for key in self.params_name_ordered:
            temp += f"{key}={self.parameters[key]}, "
        temp = temp[:-2]
        return f"{self.__class__.__name__}({temp})"

    def before_initialization(self, starting_positions=None):
        if starting_positions is None:
            pass
        elif type(starting_positions) in [list, np.ndarray] and len(starting_positions) == self.pop_size:
            if isinstance(starting_positions[0], np.ndarray) and len(starting_positions[0]) == self.problem.n_dims:
                self.pop = [self.create_solution(self.problem.lb, self.problem.ub, pos) for pos in starting_positions]
            else:
                raise ValueError("Starting positions should be a list of positions or 2D matrix of positions only.")
        else:
            raise ValueError(
                "Starting positions should be a list/2D matrix of positions with same length as pop_size hyper-parameter.")

    def initialization(self):
        if self.pop is None:
            self.pop = self.create_population(self.pop_size)

    def after_initialization(self):
        # The initial population is sorted or not depended on algorithm's strategy
        pop_temp, best, worst = self.get_special_solutions(self.pop, best=1, worst=1)
        self.g_best, self.g_worst = best[0], worst[0]
        # pop_temp, self.g_best = self.get_global_best_solution(self.pop)
        if self.sort_flag: self.pop = pop_temp
        ## Store initial best and worst solutions
        self.history.store_initial_best_worst(self.g_best, self.g_worst)

    def before_main_loop(self):
        pass

    def initialize_variables(self):
        pass

    def get_target_wrapper(self, position, counted=True):
        if counted:
            self.nfe_counter += 1
        objs = self.problem.fit_func(position)
        if not self.problem.obj_is_list:
            objs = [objs]
        fit = np.dot(objs, self.problem.obj_weights)
        return [fit, objs]

    def create_solution(self, lb=None, ub=None, pos=None):
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        return [position, target]

    def evolve(self, epoch):
        pass

    def amend_position(self, position=None, lb=None, ub=None):
        pos = bounded_position(position, lb, ub)
        return self.problem.amend_position(pos, lb, ub)

    def check_problem(self, problem):
        self.problem = problem if isinstance(problem, Problem) else Problem(**problem)
        self.generate_position = self.problem.generate_position
        self.logger = Logger(self.problem.log_to, log_file=self.problem.log_file).create_logger(
            name=f"{self.__module__}.{self.__class__.__name__}")
        self.logger.info(self.problem.msg)
        self.history = History(log_to=self.problem.log_to, log_file=self.problem.log_file)
        self.pop, self.g_best, self.g_worst = None, None, None

    def check_mode_and_workers(self, mode, n_workers):
        self.mode = self.validator.check_str("mode", mode, ["single", "swarm", "thread", "process"])
        if self.mode in ("process", "thread"):
            if not self.support_parallel_modes:
                self.logger.warning(
                    f"{self.__class__.__name__} doesn't support parallelization. The default mode 'single' is activated.")
                self.mode = "single"
            elif n_workers is not None:
                if self.mode == "process":
                    self.n_workers = self.validator.check_int("n_workers", n_workers, [2, min(61, os.cpu_count() - 1)])
                if self.mode == "thread":
                    self.n_workers = self.validator.check_int("n_workers", n_workers, [2, min(32, os.cpu_count() + 4)])
            else:
                self.logger.warning(
                    f"The parallel mode: {self.mode} is selected. But n_workers is not set. The default n_workers = 4 is used.")
                self.n_workers = 4

    def check_termination(self, mode="start", termination=None, epoch=None):
        if mode == "start":
            self.termination = termination
            if termination is not None:
                if isinstance(termination, Termination):
                    self.termination = termination
                elif type(termination) == dict:
                    self.termination = Termination(log_to=self.problem.log_to, log_file=self.problem.log_file,
                                                   **termination)
                else:
                    raise ValueError("Termination needs to be a dict or an instance of Termination class.")
                self.nfe_counter = 0
                self.termination.set_start_values(0, self.nfe_counter, time.perf_counter(), 0)
        else:
            finished = False
            if self.termination is not None:
                es = self.history.get_global_repeated_times(self.ID_TAR, self.ID_FIT, self.termination.epsilon)
                finished = self.termination.should_terminate(epoch, self.nfe_counter, time.perf_counter(), es)
                if finished:
                    self.logger.warning(self.termination.message)
            return finished

    def solve(self, problem=None, mode='single', starting_positions=None, n_workers=None, termination=None):
        self.check_problem(problem)
        self.check_mode_and_workers(mode, n_workers)
        self.check_termination("start", termination, None)
        self.initialize_variables()

        self.before_initialization(starting_positions)
        self.initialization()
        self.after_initialization()

        self.before_main_loop()
        for epoch in range(0, self.epoch):
            time_epoch = time.perf_counter()
            self.evolve(epoch)

            pop_temp, self.g_best = self.update_global_best_solution(self.pop)
            if self.sort_flag:
                self.pop = pop_temp

            time_epoch = time.perf_counter() - time_epoch
            self.track_optimize_step(self.pop, epoch + 1, time_epoch)
            if self.check_termination("end", None, epoch + 1):
                break
        self.track_optimize_process()
        return self.solution[self.ID_POS], self.solution[self.ID_TAR][self.ID_FIT]

    def track_optimize_step(self, population=None, epoch=None, runtime=None):
        pop = deepcopy(population)
        if self.problem.save_population:
            self.history.list_population.append(pop)
        self.history.list_epoch_time.append(runtime)
        self.history.list_global_best_fit.append(self.history.list_global_best[-1][self.ID_TAR][self.ID_FIT])
        self.history.list_current_best_fit.append(self.history.list_current_best[-1][self.ID_TAR][self.ID_FIT])
        # Save the exploration and exploitation data for later usage
        pos_matrix = np.array([agent[self.ID_POS] for agent in pop])
        div = np.mean(np.abs(np.median(pos_matrix, axis=0) - pos_matrix), axis=0)
        self.history.list_diversity.append(np.mean(div, axis=0))
        self.logger.info(
            f">Problem: {self.problem.name}, Epoch: {epoch}, Current best: "
            f"{self.history.list_current_best[-1][self.ID_TAR][self.ID_FIT]}, "
            f"Global best: {self.history.list_global_best[-1][self.ID_TAR][self.ID_FIT]},"
            f" Runtime: {runtime:.5f} seconds")

    def track_optimize_process(self):
        self.history.epoch = len(self.history.list_diversity)
        div_max = np.max(self.history.list_diversity)
        self.history.list_exploration = 100 * (np.array(self.history.list_diversity) / div_max)
        self.history.list_exploitation = 100 - self.history.list_exploration
        self.history.list_global_best = self.history.list_global_best[1:]
        self.history.list_current_best = self.history.list_current_best[1:]
        self.solution = self.history.list_global_best[-1]
        self.history.list_global_worst = self.history.list_global_worst[1:]
        self.history.list_current_worst = self.history.list_current_worst[1:]

    def create_population(self, pop_size=None):
        if pop_size is None:
            pop_size = self.pop_size
        pop = []
        if self.mode == "thread":
            with parallel.ThreadPoolExecutor(self.n_workers) as executor:
                list_executors = [executor.submit(self.create_solution, self.problem.lb, self.problem.ub) for _ in
                                  range(pop_size)]
                for f in parallel.as_completed(list_executors):
                    pop.append(f.result())
        elif self.mode == "process":
            with parallel.ProcessPoolExecutor(self.n_workers) as executor:
                list_executors = [executor.submit(self.create_solution, self.problem.lb, self.problem.ub) for _ in
                                  range(pop_size)]
                for f in parallel.as_completed(list_executors):
                    pop.append(f.result())
        else:
            pop = [self.create_solution(self.problem.lb, self.problem.ub) for _ in range(0, pop_size)]
        return pop

    def update_target_wrapper_population(self, pop=None):
        pos_list = [agent[self.ID_POS] for agent in pop]
        if self.mode == "thread":
            with parallel.ThreadPoolExecutor(self.n_workers) as executor:
                list_results = executor.map(partial(self.get_target_wrapper, counted=False), pos_list)
                for idx, target in enumerate(list_results):
                    pop[idx][self.ID_TAR] = target
        elif self.mode == "process":
            with parallel.ProcessPoolExecutor(self.n_workers) as executor:
                list_results = executor.map(partial(self.get_target_wrapper, counted=False), pos_list)
                for idx, target in enumerate(list_results):
                    pop[idx][self.ID_TAR] = target
        elif self.mode == "swarm":
            for idx, pos in enumerate(pos_list):
                pop[idx][self.ID_TAR] = self.get_target_wrapper(pos, counted=False)
        else:
            return pop
        self.nfe_counter += len(pop)
        return pop

    def get_better_solution(self, agent1: list, agent2: list, reverse=False):
        if self.problem.minmax == "min":
            if agent1[self.ID_TAR][self.ID_FIT] < agent2[self.ID_TAR][self.ID_FIT]:
                return deepcopy(agent1) if reverse is False else deepcopy(agent2)
            return deepcopy(agent2) if reverse is False else deepcopy(agent1)
        else:
            if agent1[self.ID_TAR][self.ID_FIT] < agent2[self.ID_TAR][self.ID_FIT]:
                return deepcopy(agent2) if reverse is False else deepcopy(agent1)
            return deepcopy(agent1) if reverse is False else deepcopy(agent2)

    def compare_agent(self, agent_new: list, agent_old: list):
        if self.problem.minmax == "min":
            if agent_new[self.ID_TAR][self.ID_FIT] < agent_old[self.ID_TAR][self.ID_FIT]:
                return True
            return False
        else:
            if agent_new[self.ID_TAR][self.ID_FIT] < agent_old[self.ID_TAR][self.ID_FIT]:
                return False
            return True

    def get_special_solutions(self, pop=None, best=3, worst=3):
        if self.problem.minmax == "min":
            pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT])
        else:
            pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT], reverse=True)
        if best is None:
            if worst is None:
                raise ValueError("Best and Worst can not be None in get_special_solutions function!")
            else:
                return pop, None, deepcopy(pop[::-1][:worst])
        else:
            if worst is None:
                return pop, deepcopy(pop[:best]), None
            else:
                return pop, deepcopy(pop[:best]), deepcopy(pop[::-1][:worst])

    def update_global_best_solution(self, pop=None, save=True):
        if self.problem.minmax == "min":
            sorted_pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT])
        else:
            sorted_pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT], reverse=True)
        current_best = sorted_pop[0]
        current_worst = sorted_pop[-1]
        if save:
            ## Save current best
            self.history.list_current_best.append(current_best)
            better = self.get_better_solution(current_best, self.history.list_global_best[-1])
            self.history.list_global_best.append(better)

            ## Save current worst
            self.history.list_current_worst.append(current_worst)
            worse = self.get_better_solution(current_worst, self.history.list_global_worst[-1], reverse=True)
            self.history.list_global_worst.append(worse)
            return deepcopy(sorted_pop), deepcopy(better)
        else:
            ## Handle current best
            local_better = self.get_better_solution(current_best, self.history.list_current_best[-1])
            self.history.list_current_best[-1] = local_better
            global_better = self.get_better_solution(current_best, self.history.list_global_best[-1])
            self.history.list_global_best[-1] = global_better

            ## Handle current worst
            local_worst = self.get_better_solution(current_worst, self.history.list_current_worst[-1], reverse=True)
            self.history.list_current_worst[-1] = local_worst
            global_worst = self.get_better_solution(current_worst, self.history.list_global_worst[-1], reverse=True)
            self.history.list_global_worst[-1] = global_worst
            return deepcopy(sorted_pop), deepcopy(global_better)

    def get_index_roulette_wheel_selection(self, list_fitness: np.array):
        if type(list_fitness) in [list, tuple, np.ndarray]:
            list_fitness = np.array(list_fitness).flatten()
        if list_fitness.ptp() == 0:
            return int(np.random.randint(0, len(list_fitness)))
        if np.any(list_fitness) < 0:
            list_fitness = list_fitness - np.min(list_fitness)
        final_fitness = list_fitness
        if self.problem.minmax == "min":
            final_fitness = np.max(list_fitness) - list_fitness
        prob = final_fitness / np.sum(final_fitness)
        return int(np.random.choice(range(0, len(list_fitness)), p=prob))

    def get_index_kway_tournament_selection(self, pop=None, k_way=0.2, output=2, reverse=False):
        if 0 < k_way < 1:
            k_way = int(k_way * len(pop))
        list_id = np.random.choice(range(len(pop)), k_way, replace=False)
        list_parents = [[idx, pop[idx][self.ID_TAR][self.ID_FIT]] for idx in list_id]
        if self.problem.minmax == "min":
            list_parents = sorted(list_parents, key=lambda agent: agent[1])
        else:
            list_parents = sorted(list_parents, key=lambda agent: agent[1], reverse=True)
        if reverse:
            return [parent[0] for parent in list_parents[-output:]]
        return [parent[0] for parent in list_parents[:output]]

    def greedy_selection_population(self, pop_old=None, pop_new=None):
        len_old, len_new = len(pop_old), len(pop_new)
        if len_old != len_new:
            raise ValueError("Greedy selection of two population with different length.")
        if self.problem.minmax == "min":
            return [pop_new[i] if pop_new[i][self.ID_TAR][self.ID_FIT] < pop_old[i][self.ID_TAR][self.ID_FIT]
                    else pop_old[i] for i in range(len_old)]
        else:
            return [pop_new[i] if pop_new[i][self.ID_TAR] > pop_old[i][self.ID_TAR]
                    else pop_old[i] for i in range(len_old)]

    def get_sorted_strim_population(self, pop=None, pop_size=None, reverse=False):
        if self.problem.minmax == "min":
            pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT], reverse=reverse)
        else:
            pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT], reverse=reverse)
        return pop[:pop_size]
