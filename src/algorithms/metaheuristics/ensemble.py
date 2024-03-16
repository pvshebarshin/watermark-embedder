class MetaheuristicEnsemble:

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
        best_position1, best_fitness1 = self.model1.solve(problem)
        print(f"Best solution: {best_position1}, Best fitness: {best_fitness1}")

        if self.model2 is not None:

            best_position2, best_fitness2 = self.model2.solve(problem)
            print(f"Best solution: {best_position2}, Best fitness: {best_fitness2}")

            if self.model3 is not None:

                best_position3, best_fitness3 = self.model3.solve(problem)
                print(f"Best solution: {best_position3}, Best fitness: {best_fitness3}")
                avg_cor = self.average((best_fitness1, best_fitness2, best_fitness3))
                avg_arr = self.find_average_arrays3(best_position1, best_position2, best_position3)
                return avg_arr, avg_cor

            else:

                avg_cor = self.average((best_fitness1, best_fitness2))
                avg_arr = self.find_average_arrays2(best_position1, best_position2)
                return avg_arr, avg_cor

        else:

            return best_position1, best_fitness1
