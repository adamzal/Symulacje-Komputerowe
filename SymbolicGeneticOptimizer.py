import numpy as np
from sympy.parsing.sympy_parser import parse_expr


class SymbolicGeneticOptimizer:

    def __init__(self,  function: str, n_min: int, n_max: int, n_dig: int = 32) -> None:
        """
        Initializes the SymbolicGeneticOptimizer class.

        Parameters:
            - function (str): The symbolic function in Python notation.
            - n_min (int): The minimum value for the function's variables.
            - n_max (int): The maximum value for the function's variables.
            - n_dig (int): The number of digits to represent the binary encoding (default is 32).
        """
        self.function = parse_expr(function)
        self.variables = list(self.function.free_symbols)

        self.n_min = n_min
        self.n_max = n_max
        self.n_dig = n_dig

        if len(self.variables) == 1:
            self.points_x_list = []
        elif len(self.variables) == 2:
            self.points_x_list = []
            self.points_y_list = []
        else:
            self.points_list = []
        self.values_list = []

    def get_dec(self, float_number):
        """
            Converts a floating-point number to its decimal representation.

            Parameters:
                - float_number: The floating-point number to convert.

            Returns:
                int: The decimal representation of the input float_number.
        """
        a = -2**self.n_dig/(self.n_min-self.n_max)
        b = -self.n_min*a
        return int(a*float_number + b)

    def get_float(self, int_number):
        """
            Converts a decimal number to its floating-point representation.

            Parameters:
                - int_number: The decimal number to convert.

            Returns:
                float: The floating-point representation of the input int_number.
        """
        a = -2 ** self.n_dig / (self.n_min - self.n_max)
        b = -self.n_min * a
        return (int_number - b)/a

    def gray_to_binary(self, gray_code):
        """
            Converts a Gray code to its binary representation.

            Parameters:
                - gray_code: The Gray code to convert.

            Returns:
                str: The binary representation of the input Gray code.
        """
        binary_representation = gray_code[0]

        for i in range(1, len(gray_code)):
            binary_bit = str(int(gray_code[i]) ^ int(binary_representation[i - 1]))
            binary_representation += binary_bit

        return binary_representation

    def binary_to_gray(self, binary_code):
        """
            Converts a binary code to its Gray code representation.

            Parameters:
                - binary_code: The binary code to convert.

            Returns:
                str: The Gray code representation of the input binary code.
        """
        gray_representation = binary_code[0]

        for i in range(1, len(binary_code)):
            gray_bit = str(int(binary_code[i]) ^ int(binary_code[i - 1]))
            gray_representation += gray_bit

        return gray_representation

    def mutateing(self, p):
        """
            Applies mutation to an individual in the population.

            Parameters:
                - p: The individual to mutate.

            Returns:
                list: The mutated individual.
        """
        if np.random.uniform(0, 1) <= 1 / self.n_dig:
            for i in range(len(self.variables)):
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
        n = np.random.uniform(self.n_min, self.n_max, (num_of_n, len(self.variables)))

        if objective_method.lower() == "max":
            n = sorted(n, key=lambda point: self.function.subs({var: val for var, val in zip(self.variables, point)}), reverse=True)
            p = sorted(p, key=lambda point: self.function.subs({var: val for var, val in zip(self.variables, point)}), reverse=True)

        elif objective_method.lower() == "min":
            n = sorted(n, key=lambda point: self.function.subs({var: val for var, val in zip(self.variables, point)}),
                       reverse=False)
            p = sorted(p,
                       key=lambda point: self.function.subs({var: val for var, val in zip(self.variables, point)}),
                       reverse=False)

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
            crossbreded = sorted(crossbreded, key=lambda point: self.function.subs({var: val for var, val in zip(self.variables, point)}), reverse=True)
        elif objective_method.lower() == "min":
            crossbreded = sorted(crossbreded, key=lambda point: self.function.subs({var: val for var, val in zip(self.variables, point)}), reverse=False)

        return crossbreded[:num_of_p]

    def fit(self, objective_method: str, genetic_method: str, num_of_p: int = 10, num_of_n: int = 4, initial_individual: np.ndarray = None, num_generations: int = 100):
        """
            Optimizes the symbolic function using genetic algorithms.

            Parameters:
                - objective_method (str): Objective method ("min" or "max").
                - genetic_method (str): Genetic method ("mutate" or "crossbreed").
                - num_of_p (int): Number of individuals in the population.
                - num_of_n (int): Number of individuals in the new population (for crossbreeding).
                - initial_individual (np.ndarray): Initial individual for optimization (optional).
                - num_generations (int): Number of generations.

            Returns:
                Tuple[np.ndarray, float]: The best individual and its fitness value.
        """
        if len(self.variables) == 1:
            self.points_x_list = []
        elif len(self.variables) == 2:
            self.points_x_list = []
            self.points_y_list = []
        else:
            self.points_list = []
        self.values_list = []

        P = []
        if initial_individual:
            P = initial_individual
        elif genetic_method.lower() == "crossbreed":
            P = np.random.uniform(self.n_min, self.n_max, (num_of_p, len(self.variables)))
        elif genetic_method.lower() == "mutate":
            P = np.random.uniform(self.n_min, self.n_max, len(self.variables))
            if len(self.variables) == 1:
                self.points_x_list += [P]
            elif len(self.variables) == 2:
                self.points_x_list += [P[0]]
                self.points_y_list += [P[1]]
            else:
                self.points_list += [P]
            self.values_list += [self.function.subs({var: val for var, val in zip(self.variables, P)})]

        for generation in range(num_generations):
            N = P.copy()

            if genetic_method == "mutate":
                N = self.mutateing(N)

                current_function_value = self.function.subs({var: val for var, val in zip(self.variables, P)})
                new_function_value = self.function.subs({var: val for var, val in zip(self.variables, N)})

                if objective_method.lower() == "min" and current_function_value > new_function_value and genetic_method == "mutate" or \
                        objective_method.lower() == "max" and current_function_value < new_function_value and genetic_method == "mutate":
                    P = N

                    if len(self.variables) == 1:
                        self.points_x_list += [P]
                    elif len(self.variables) == 2:
                        self.points_x_list += [P[0]]
                        self.points_y_list += [P[1]]
                    else:
                        self.points_list += [P]
                    self.values_list += [self.function.subs({var: val for var, val in zip(self.variables, P)})]

            elif genetic_method == "crossbreed":
                N = self.crossbreeding(p=N, num_of_n=num_of_n, num_of_p=num_of_p, objective_method=objective_method)

        if genetic_method == "mutate":
            return P, self.function.subs({var: val for var, val in zip(self.variables, P)})
        elif genetic_method == "crossbreed":
            return P[0], self.function.subs({var: val for var, val in zip(self.variables, P[0])})


if __name__ == "__main__":
    f = input('Input function in Python notation: ')
    SGO = SymbolicGeneticOptimizer(f, -2, 2)

    # Crossbreeding
    best_point, best_value = SGO.fit(objective_method="min", genetic_method="crossbreed", num_generations=100)
    print("Crossbreeding")
    print("The best individual:")
    for i in range(len(SGO.variables)):
        print(f"{SGO.variables[i]} = {best_point[i]}")
    print("Fitness value:", best_value)

    # Mutation
    best_point, best_value = SGO.fit(objective_method="min", genetic_method="mutate", num_generations=100)
    print()
    print("Mutating")
    print("The best individual:")
    for i in range(len(SGO.variables)):
        print(f"{SGO.variables[i]} = {best_point[i]}")
    print("Fitness value:", best_value)