import concurrent.futures


class MetaheuristicEnsembleAvg:

    def __init__(self, model1, model2=None, model3=None):
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

    @staticmethod
    def average(lst):
        return sum(lst) / len(lst)

    @staticmethod
    def find_average_arrays3(arr1, arr2, arr3):
        if len(arr1) != len(arr2) or len(arr1) != len(arr3):
            raise ValueError("Arrays should be of the same length")

        average_array = []
        for i in range(len(arr1)):
            avg_value = (arr1[i] + arr2[i] + arr3[i]) / 3
            average_array.append(avg_value)

        return average_array

    @staticmethod
    def find_average_arrays2(arr1, arr2):
        if len(arr1) != len(arr2):
            raise ValueError("Arrays should be of the same length")

        average_array = []
        for i in range(len(arr1)):
            avg_value = (arr1[i] + arr2[i]) / 2
            average_array.append(avg_value)

        return average_array

    def solveProblem(self, problem):
        with (concurrent.futures.ThreadPoolExecutor() as executor):

            future1 = executor.submit(self.model1.solve, problem)

            if self.model2 is not None:

                future2 = executor.submit(self.model2.solve, problem)

                if self.model3 is not None:

                    future3 = executor.submit(self.model3.solve, problem)

                    best_position1, best_fitness1 = future1.result()
                    best_position2, best_fitness2 = future2.result()
                    best_position3, best_fitness3 = future3.result()
                    avg_cor = self.average((best_fitness1, best_fitness2, best_fitness3))
                    avg_arr = self.find_average_arrays3(best_position1, best_position2, best_position3)
                    return avg_arr, avg_cor

                else:

                    best_position1, best_fitness1 = future1.result()
                    best_position2, best_fitness2 = future2.result()
                    avg_cor = self.average((best_fitness1, best_fitness2))
                    avg_arr = self.find_average_arrays2(best_position1, best_position2)
                    return avg_arr, avg_cor

            else:

                best_position1, best_fitness1 = future1.result()
                return best_position1, best_fitness1

# берем популяцию, отрабатываем первой по ней, затем другую и популяю передаем второй

# Во время оптимизации по очереди выбираем метаэвристики (чередуем по поколениям)
