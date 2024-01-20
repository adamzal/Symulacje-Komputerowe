import sympy as sp
import numpy as np
from sympy.parsing.sympy_parser import parse_expr


class SymbolicGradientDescentOptimizer:

    def __init__(self, function: str, n_min: int, n_max: int) -> None:
        """
        Initializes the SymbolicGradientDescentOptimizer.

        Parameters:
            - function (str): Mathematical function in Python notation.
            - n_min (int): Minimum value for initialization.
            - n_max (int): Maximum value for initialization.
        """
        self.function = parse_expr(function)
        self.variables = list(self.function.free_symbols)
        self.gradient_function = [sp.diff(function, var) for var in self.variables]

        self.n_min = n_min
        self.n_max = n_max

        if len(self.variables) == 1:
            self.points_x_list = []
        elif len(self.variables) == 2:
            self.points_x_list = []
            self.points_y_list = []
        else:
            self.points_list = []
        self.values_list = []

    def fit(self, objective_method: str, initial_point: np.ndarray = None, epsilon: float = 1e-6,
            max_iterations: int = 1000, initial_step_size: float = 0.01,
            step_size_reduction_method: str = "factor", step_size_reduction_factor: float = 0.5) \
            -> (np.ndarray, float):
        """
        Fits the SymbolicGradientDescentOptimizer to the mathematical function.

        Parameters:
            - objective_method (str): "min" to find the minimum, "max" to find the maximum.
            - initial_point (np.ndarray): Initial point for optimization. If None, a random point is selected.
            - epsilon (float): Convergence threshold.
            - max_iterations (int): Maximum number of iterations.
            - initial_step_size (int | float): Initial step size for gradient descent.
            - step_size_reduction_method (str): "factor", "sqrt_divide", or "n_divide" for step size reduction.
            - step_size_reduction_factor (int | float): Factor for reducing the step size.

        Returns:
            tuple: Final optimized point (np.ndarray) and the corresponding function value (int | float).
        """
        if len(self.variables) == 1:
            self.points_x_list = []
        elif len(self.variables) == 2:
            self.points_x_list = []
            self.points_y_list = []
        else:
            self.points_list = []
        self.values_list = []

        if initial_point:
            p = initial_point

        else:
            p = np.random.uniform(self.n_min, self.n_max, len(self.variables))

        iteration = 1
        step_size = initial_step_size

        if len(self.variables) == 1:
            self.points_x_list += [p]
        elif len(self.variables) == 2:
            self.points_x_list += [p[0]]
            self.points_y_list += [p[1]]
        else:
            self.points_list += [p]
        self.values_list += [self.function.subs({var: val for var, val in zip(self.variables, p)})]

        while iteration <= max_iterations:
            gradient_value = np.array([grd.subs({var: val for var, val in zip(self.variables, p)}) for grd in self.gradient_function])

            gradient_value = np.array(gradient_value)
            if objective_method.lower() == "min":
                p_new = p - step_size * gradient_value
            elif objective_method.lower() == "max":
                p_new = p + step_size * gradient_value

            if (abs(gradient_value) <= epsilon).any() or (abs(p_new - p) <= epsilon).any():
                break

            current_function_value = self.function.subs({var: val for var, val in zip(self.variables, p)})
            new_function_value = self.function.subs({var: val for var, val in zip(self.variables, p_new)})

            if objective_method.lower() == "min" and new_function_value < current_function_value or \
                    objective_method.lower() == "max" and new_function_value > current_function_value:
                p = p_new
                iteration += 1

                if len(self.variables) == 1:
                    self.points_x_list += [p]
                elif len(self.variables) == 2:
                    self.points_x_list += [p[0]]
                    self.points_y_list += [p[1]]
                else:
                    self.points_list += [p]
                self.values_list += [self.function.subs({var: val for var, val in zip(self.variables, p)})]

            else:
                if step_size_reduction_method == "factor":
                    step_size *= step_size_reduction_factor

                elif step_size_reduction_method == "sqrt_divide":
                    step_size /= iteration ** .5

                elif step_size_reduction_method == "n_divide":
                    step_size /= iteration

        return p, self.function.subs({var: val for var, val in zip(self.variables, p)})


if __name__ == "__main__":
    # Input the mathematical function in Python notation
    f = input('Input function in Python notation: ')

    # Instantiate the optimizer with the specified range
    optimizer = SymbolicGradientDescentOptimizer(function=f, n_min=-2, n_max=2)

    # Choose optimization objective ("min" or "max")
    objective_method = "min"

    # Run the optimization
    point, value = optimizer.fit(objective_method=objective_method)

    # Print results based on the optimization objective
    if float(value) in [-np.inf, np.inf]:
        print(f"The function does not have a global/local {objective_method.lower()}imum or {objective_method.lower()}imum is too large number.")

    elif len(optimizer.values_list) > 1 and objective_method.lower() == "max":
        print("Found maximum:")
        for i in range(len(optimizer.variables)):
            print(f"{optimizer.variables[i]} = {point[i]}")
        print("Function value at the maximum:", value)

    elif len(optimizer.values_list) > 1 and objective_method.lower() == "min":
        print("Found minimum:")
        for i in range(len(optimizer.variables)):
            print(f"{optimizer.variables[i]} = {point[i]}")
        print("Function value at the minimum:", value)
