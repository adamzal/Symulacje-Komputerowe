import numpy as np


class GeneticOptimizer:

    def __init__(self, n_min: int, n_max: int, n_dig: int = 32) -> None:
        """
        Initializes the GeneticOptimizer.

        Parameters:
            - n_min (int): Minimum value for initialization.
            - n_max (int): Maximum value for initialization.
            - n_dig (int): Number of bits in the binary representation (default is 32).
        """
        self.n_min = n_min
        self.n_max = n_max
        self.n_dig = n_dig

        self.points_x_list = []
        self.points_y_list = []
        self.values_list = []

    def function(self, x, y):
        """
        Objective function to be minimized or maximized.

        Parameters:
            - x (float): Input variable.
            - y (float): Input variable.

        Returns:
            float: Result of the objective function.
        """
        return x**4 + y**4 - 4*x*y + x

    def get_dec(self, float_number):
        """
        Converts a float number to decimal representation.

        Parameters:
            - float_number (float): The float number to be converted.

        Returns:
            int: Decimal representation of the float number.
        """
        a = -2**self.n_dig/(self.n_min-self.n_max)
        b = -self.n_min*a

        return int(a*float_number + b)

    def get_float(self, int_number):
        """
        Converts an integer to its float representation.

        Parameters:
            - int_number (int): The integer to be converted.

        Returns:
            float: Float representation of the integer.
        """
        a = -2 ** self.n_dig / (self.n_min - self.n_max)
        b = -self.n_min * a

        return (int_number - b)/a

    def gray_to_binary(self, gray_code):
        """
        Converts Gray code to binary representation.

        Parameters:
            - gray_code (str): Gray code representation.

        Returns:
            str: Binary representation.
        """
        binary_representation = gray_code[0]

        for i in range(1, len(gray_code)):
            binary_bit = str(int(gray_code[i]) ^ int(binary_representation[i - 1]))
            binary_representation += binary_bit

        return binary_representation

    def binary_to_gray(self, binary_code):
        """
        Converts binary representation to Gray code.

        Parameters:
            - binary_code (str): Binary representation.

        Returns:
            str: Gray code representation.
        """
        gray_representation = binary_code[0]

        for i in range(1, len(binary_code)):
            gray_bit = str(int(binary_code[i]) ^ int(binary_code[i - 1]))
            gray_representation += gray_bit

        return gray_representation

    def mutateing(self, p: list | np.ndarray):
        """
        Mutates a population of individuals with a probability of 1/n_dig.

        Parameters:
            - p (list or np.ndarray): Population of individuals.

        Returns:
            list or np.ndarray: Mutated population.
        """
        if True: #
            for i in range(len(p)):
                p_gray = self.binary_to_gray(str(bin(self.get_dec(p[i])))[2:].zfill(self.n_dig))

                mutated_index = np.random.randint(len(p_gray))
                mutated_gray = p_gray[:mutated_index] + str(int(p_gray[mutated_index]) ^ 1) + p_gray[mutated_index+1:]

                p[i] = self.get_float(int((self.gray_to_binary(mutated_gray)), 2))

        return p

    def crossbreeding(self, p: list | np.ndarray, num_of_n: int, num_of_p: int, objective_method: str):
        """
        Performs crossbreeding between two populations.

        Parameters:
            - p (list or np.ndarray): Parent population.
            - num_of_n (int): Number of individuals in the new population.
            - num_of_p (int): Number of individuals in the parent population.
            - objective_method (str): Objective method ("min" or "max").

        Returns:
            list or np.ndarray: New population after crossbreeding.
        """
        n = np.random.uniform(self.n_min, self.n_max, (num_of_n, len(p[0])))

        if objective_method.lower() == "max":
            n = sorted(n, key=lambda point: self.function(*point), reverse=True)
            p = sorted(p, key=lambda point: self.function(*point), reverse=True)

        elif objective_method.lower() == "min":
            n = sorted(n, key=lambda point: self.function(*point), reverse=False)
            p = sorted(p, key=lambda point: self.function(*point), reverse=False)

        crossbreded = []
        for i in range(len(p)):
            for j in range(len(n)):
                cross_indiv = []
                for k in range(len(p[0])):
                    p_gray = self.binary_to_gray(str(bin(self.get_dec(p[i][k])))[2:].zfill(self.n_dig))
                    n_gray = self.binary_to_gray(str(bin(self.get_dec(n[j][k])))[2:].zfill(self.n_dig))

                    cross_index = np.random.randint(len(p_gray))
                    cross_gray = p_gray[:cross_index] + n_gray[cross_index:]

                    cross_indiv += [self.get_float(int((self.gray_to_binary(cross_gray)), 2))]
                crossbreded += [cross_indiv]

        for i in range(len(crossbreded)):
            crossbreded[i] = self.mutateing(crossbreded[i])

        if objective_method.lower() == "max":
            crossbreded = sorted(crossbreded, key=lambda point: self.function(*point), reverse=True)
        elif objective_method.lower() == "min":
            crossbreded = sorted(crossbreded, key=lambda point: self.function(*point), reverse=False)

        return crossbreded[:num_of_p]

    def fit(self, objective_method: str, genetic_method: str, num_of_p: int = 10, num_of_n: int = 4, initial_individual: list | np.ndarray = None, num_generations: int = 100):
        """
        Fits the GeneticOptimizer to find the optimal solution.

        Parameters:
            - objective_method (str): Objective method ("min" or "max").
            - genetic_method (str): Genetic method ("mutate" or "crossbreed").
            - num_of_p (int): Number of individuals in the parent population.
            - num_of_n (int): Number of individuals in the new population.
            - initial_individual (list or np.ndarray): Initial individual or population (default is None).
            - num_generations (int): Number of generations (default is 1000).

        Returns:
            tuple: Best individual and its fitness value.
        """
        self.points_x_list = []
        self.points_y_list = []
        self.values_list = []

        P = []
        if initial_individual:
            P = initial_individual
        elif genetic_method.lower() == "crossbreed":
            P = np.random.uniform(self.n_min, self.n_max, (num_of_p, 2))
        elif genetic_method.lower() == "mutate":
            P = np.random.uniform(self.n_min, self.n_max, 2)
            self.points_x_list += [P[0]]
            self.points_y_list += [P[1]]
            self.values_list += [self.function(*P)]

        for generation in range(num_generations):
            N = P.copy()

            if genetic_method == "mutate":
                N = self.mutateing(N)

                if objective_method.lower() == "min" and self.function(*P) > self.function(
                        *N) and genetic_method == "mutate" or \
                        objective_method.lower() == "max" and self.function(*P) < self.function(
                    *N) and genetic_method == "mutate":
                    P = N

                    self.points_x_list += [P[0]]
                    self.points_y_list += [P[1]]
                    self.values_list += [self.function(*P)]

            elif genetic_method == "crossbreed":
                N = self.crossbreeding(p=N, num_of_n=num_of_n, num_of_p=num_of_p, objective_method=objective_method)
                P = N

        if genetic_method == "mutate":
            return P, self.function(*P)
        elif genetic_method == "crossbreed":
            return P[0], self.function(*P[0])


if __name__ == "__main__":
    # Instantiate the optimizer with the specified range
    GO = GeneticOptimizer(n_min=-2, n_max=2)

    # Choose genetic method ("mutate" or "crossbreed")
    genetic_method = "mutate"
    objective_method = "min"

    # Run the optimization
    [x, y], best_value = GO.fit(objective_method=objective_method, genetic_method=genetic_method, num_generations=100)

    # Print results based on the genetic method
    print("The best individual:")
    print(f"x = {x}")
    print(f"y = {y}")
    print("Fitness value:", best_value)

