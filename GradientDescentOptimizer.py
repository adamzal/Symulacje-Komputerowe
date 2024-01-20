import numpy as np


class GradientDescentOptimizer:

    def __init__(self, n_min, n_max) -> None:
        """
        Initializes the GradientDescentOptimizer.

        Parameters:
            - n_min (float): Minimum value for initialization.
            - n_max (float): Maximum value for initialization.
        """
        self.n_min = n_min
        self.n_max = n_max

        self.points_x_list = []
        self.points_y_list = []
        self.values_list = []

    @staticmethod
    def function(x: float, y: float) -> float:
        """
        Objective function to be minimized or maximized.

        Parameters:
            - x (float): Input variable.
            - y (float): Input variable.

        Returns:
            float: Result of the objective function.
        """
        return x ** 4 + y ** 4 - 4 * x * y + x

    @staticmethod
    def gradient_function(x: float, y: float) -> tuple:
        """
        Gradient function of the objective function.

        Parameters:
            - x (float): Input variable.
            - y (float): Input variable.

        Returns:
            tuple: Gradient values with respect to x and y.
        """
        return 4 * x ** 3 - 4 * y + 1, 4 * y ** 3 - 4 * x

    def fit(self, objective_method: str, initial_point: np.ndarray = None, epsilon: float = 1e-6, max_iterations: int = 1000,
            initial_step_size: float = 0.01, step_size_reduction_method: str = "factor",
            step_size_reduction_factor: float = 0.5) -> (np.ndarray, float):
        """
            Fits the GradientDescentOptimizer to the objective function.

            Parameters:
                - objective_method (str): "min" to find the minimum, "max" to find the maximum.
                - initial_point (np.ndarray): Initial point for optimization. If None, a random point is selected.
                - epsilon (float): Convergence threshold.
                - max_iterations (int): Maximum number of iterations.
                - initial_step_size (float): Initial step size for gradient descent.
                - step_size_reduction_method (str): "factor", "sqrt_divide", or "n_divide" for step size reduction.
                - step_size_reduction_factor (float): Factor for reducing the step size.

            Returns:
                tuple: Final optimized point (np.ndarray) and the corresponding function value (float).
        """
        self.points_x_list = []
        self.points_y_list = []
        self.values_list = []

        if initial_point:
            x, y = initial_point

        else:
            x, y = np.random.uniform(self.n_min, self.n_max, 2)

        iteration = 1
        step_size = initial_step_size

        self.points_x_list += [x]
        self.points_y_list += [y]
        self.values_list += [self.function(x, y)]

        while iteration <= max_iterations:
            gradient_value_x, gradient_value_y = self.gradient_function(x, y)

            if objective_method.lower() == "min":
                x_new = x - step_size * gradient_value_x
                y_new = y - step_size * gradient_value_y
            elif objective_method.lower() == "max":
                x_new = x + step_size * gradient_value_x
                y_new = y + step_size * gradient_value_y

            if abs(gradient_value_x) <= epsilon or abs(gradient_value_y) <= epsilon \
                    or abs(x_new - x) <= epsilon or abs(y_new - y) <= epsilon:
                break

            current_function_value = self.function(x, y)
            new_function_value = self.function(x_new, y_new)

            if objective_method.lower() == "min" and new_function_value < current_function_value or \
                    objective_method.lower() == "max" and new_function_value > current_function_value:
                x = x_new
                y = y_new

                self.points_x_list += [x]
                self.points_y_list += [y]
                self.values_list += [self.function(x, y)]

                iteration += 1
            else:
                if step_size_reduction_method == "factor":
                    step_size *= step_size_reduction_factor

                elif step_size_reduction_method == "sqrt_divide":
                    step_size /= iteration**.5

                elif step_size_reduction_method == "n_divide":
                    step_size /= iteration

        return np.array([x, y]), self.function(x, y)


if __name__ == "__main__":
    # Instantiate the optimizer with the specified range
    optimizer = GradientDescentOptimizer(n_min=-2, n_max=2)

    # Choose optimization objective ("min" or "max")
    objective_method = "min"

    # Run the optimization
    point, value = optimizer.fit(objective_method=objective_method, max_iterations=100)

    # Print results based on the optimization objective
    if objective_method.lower() == "max" and float(value) in [-np.inf, np.inf]:
        print("The function does not have a global/local maximum or maximum is too large number.")

    elif objective_method.lower() == "min" and float(value) in [-np.inf, np.inf]:
        print("The function does not have a global/local minimum or minimum is too large number")

    elif len(optimizer.values_list) > 1 and objective_method.lower() == "max":
        print("Found maximum:")
        print(f"x = {point[0]}")
        print(f"y = {point[1]}")
        print("Function value at the maximum:", value)

    elif len(optimizer.values_list) > 1 and objective_method.lower() == "min":
        print("Found minimum:")
        print(f"x = {point[0]}")
        print(f"y = {point[1]}")
        print("Function value at the minimum:", value)

