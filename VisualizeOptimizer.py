from SymbolicGradientDescentOptimizer import *
from GradientDescentOptimizer import *
import numpy as np
import matplotlib.pyplot as plt


class VisualizeOptimizer:

    def __init__(self, objective_method: str, optimizer: GradientDescentOptimizer | SymbolicGradientDescentOptimizer):
        """
            Initializes the VisualizeOptimizer class.

            Parameters:
                - objective_method (str): The optimization objective, either "min" for minimization or "max" for maximization.
                - optimizer (GradientDescentOptimizer | SymbolicGradientDescentOptimizer): An instance of the optimizer to visualize.
        """
        self.optimizer = optimizer
        self.objective_method = objective_method
        self.point, self.value = self.optimizer.fit(objective_method=objective_method)

    def show_3d(self, initial_points: list = None):
        """
            Visualizes the 3D plot of the objective function.

            Parameters:
                - initial_points (list): List of initial points for optimization trajectory (optional).
        """
        x_vals = np.linspace(self.optimizer.n_min, self.optimizer.n_max, 100)
        y_vals = np.linspace(self.optimizer.n_min, self.optimizer.n_max, 100)
        x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
        if isinstance(self.optimizer, SymbolicGradientDescentOptimizer):
            if len(self.optimizer.variables) != 2:
                print(f"You can not use this method. Your function has {len(self.optimizer.variables)} variables but "
                      f"expected 2")
                return

            numeric_function = sp.lambdify(self.optimizer.variables, self.optimizer.function, 'numpy')
            z_mesh = numeric_function(x_mesh, y_mesh)
        else:
            z_mesh = self.optimizer.function(x_mesh, y_mesh)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap='viridis', alpha=0.5, label=None)

        if initial_points:
            for p in initial_points:
                self.point, self.value = self.optimizer.fit(objective_method=self.objective_method, initial_point=p)

                ax.scatter(self.point[0], self.point[1], self.value, color='green',
                           marker='o', s=20)
                ax.scatter(self.optimizer.points_x_list[0], self.optimizer.points_y_list[0],
                           self.optimizer.values_list[0], color='red', marker='o', s=20)
                ax.plot(self.optimizer.points_x_list, self.optimizer.points_y_list, self.optimizer.values_list,
                        color='black')
        else:
            if self.objective_method.lower() == "min":
                ax.scatter(self.point[0], self.point[1], self.value, color='green', marker='o', s=20, label='Minimum')
            elif self.objective_method.lower() == "max":
                ax.scatter(self.point[0], self.point[1], self.value, color='green', marker='o', s=20, label='Maximum')

            ax.scatter(self.optimizer.points_x_list[0], self.optimizer.points_y_list[0],
                       self.optimizer.values_list[0], color='red', marker='o', s=20, label='Punkt startowy')
            ax.plot(self.optimizer.points_x_list, self.optimizer.points_y_list, self.optimizer.values_list,
                    color='black')

        if isinstance(self.optimizer, SymbolicGradientDescentOptimizer):
            x = str(self.optimizer.variables[0]).upper()
            y = str(self.optimizer.variables[1]).upper()
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_zlabel(f'f({x}, {y})')
            ax.set_title(f'Objective function: $f({x}, {y}) = {sp.latex(self.optimizer.function)}$')
        else:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('f(X, Y)')
            ax.set_title('Objective function: $f(x, y) = x^4 + y^4 - 4xy + x$')

        if self.objective_method.lower() == "min":
            ax.legend(["Function", "Minimum", "Start point"])

        elif self.objective_method.lower() == "max":
            ax.legend(["Function", "Maximum", "Start point"])

        plt.show()

    def show_contourf(self, initial_points: list = None):
        """
            Visualizes the contour plot of the objective function.

            Parameters:
                - initial_points (list): List of initial points for optimization trajectory (optional).
        """
        x_vals = np.linspace(self.optimizer.n_min, self.optimizer.n_max, 100)
        y_vals = np.linspace(self.optimizer.n_min, self.optimizer.n_max, 100)
        x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)

        if isinstance(self.optimizer, SymbolicGradientDescentOptimizer):
            if len(self.optimizer.variables) != 2:
                print(f"You can not use this method. Your function has {len(self.optimizer.variables)} variables but "
                      f"expected 2")
                return

            numeric_function = sp.lambdify(self.optimizer.variables, self.optimizer.function, 'numpy')
            z_mesh = numeric_function(x_mesh, y_mesh)
        else:
            z_mesh = self.optimizer.function(x_mesh, y_mesh)

        plt.contourf(x_vals, y_vals, z_mesh, cmap='viridis')

        if initial_points:
            for p in initial_points:
                self.point, self.value = self.optimizer.fit(objective_method=self.objective_method, initial_point=p)

                plt.scatter(self.point[0], self.point[1], color='green', marker='o', s=20)
                plt.scatter(self.optimizer.points_x_list[0], self.optimizer.points_y_list[0], color='red',
                            marker='o', s=20)
                plt.plot(self.optimizer.points_x_list, self.optimizer.points_y_list, color='black')
        else:
            plt.scatter(self.point[0], self.point[1], color='green', marker='o',
                        s=20, label='Minimum')
            plt.scatter(self.optimizer.points_x_list[0], self.optimizer.points_y_list[0], color='red', marker='o',
                        s=20, label='Start point')
            plt.plot(self.optimizer.points_x_list, self.optimizer.points_y_list, color='black')

        if isinstance(self.optimizer, SymbolicGradientDescentOptimizer):
            x = str(self.optimizer.variables[0]).upper()
            y = str(self.optimizer.variables[1]).upper()
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(f'Objective function: $f({x}, {y}) = {sp.latex(self.optimizer.function)}$')
        else:
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Objective function: $f(x, y) = x^4 + y^4 - 4xy + x$')

        if self.objective_method.lower() == "min":
            plt.legend(["Minimum", "Start point"])
        elif self.objective_method.lower() == "max":
            plt.legend(["Maximum", "Start point"])

        plt.show()

    def show_2d(self, initial_points: list = None):
        """
            Visualizes the 2D plot of the objective function.

            Parameters:
                - initial_points (list): List of initial points for optimization trajectory (optional).
        """
        x_vals = np.linspace(self.optimizer.n_min, self.optimizer.n_max, 100)

        if isinstance(self.optimizer, SymbolicGradientDescentOptimizer):

            if len(self.optimizer.variables) != 1:
                print(f"You can not use this method. Your function has {len(self.optimizer.variables)} variables but "
                      f"expected 1")
                return

            numeric_function = sp.lambdify(self.optimizer.variables, self.optimizer.function, 'numpy')
            y_vals = numeric_function(x_vals)

        plt.plot(x_vals, y_vals)

        if initial_points:
            for p in initial_points:
                self.point, self.value = self.optimizer.fit(objective_method=self.objective_method, initial_point=p)

                plt.scatter(self.point, self.value, color='green', marker='o', s=40)
                plt.scatter(self.optimizer.points_x_list[0],
                            self.optimizer.values_list[0], color='red', marker='o', s=40)
                plt.scatter(self.optimizer.points_x_list, self.optimizer.values_list, color='black', marker="x")
        else:
            if objective_method.lower() == "min":
                plt.scatter(self.point, self.value, color='green', marker='o', s=40, label='Minimum')
            elif objective_method.lower() == "max":
                plt.scatter(self.point, self.value, color='green', marker='o', s=40, label='Maximum')

            plt.scatter(self.optimizer.points_x_list[0], self.optimizer.values_list[0], color='red', marker='o',
                        s=40, label='Start point')
            plt.scatter(self.optimizer.points_x_list, self.optimizer.values_list, color='black', marker="x", s=20)

        if isinstance(self.optimizer, SymbolicGradientDescentOptimizer):
            x = str(self.optimizer.variables[0]).upper()

            plt.xlabel(x)
            plt.ylabel(f'f({x})')
            plt.title(f'Objective function: $f({x}) = {sp.latex(self.optimizer.function)}$')

        if self.objective_method.lower() == "min":
            plt.legend(["Function", "Minimum", "Start point"])

        elif self.objective_method.lower() == "max":
            plt.legend(["Function", "Maximum", "Start point"])

        plt.show()


if __name__ == "__main__":
    # Create instances of optimizers
    grad_desc = GradientDescentOptimizer(-2, 2)
    symb_grad_opt = SymbolicGradientDescentOptimizer("-x**4-y**2+4*x*y-y", -7, 4)
    symb_grad_opt_2 = SymbolicGradientDescentOptimizer("(x-3)*(x+4)*(x-1)", -6, 6)

    # List of initial points for optimization trajectory
    points = [[0, 2], [1, 1.5], [2, 0], [1, -1.5], [0, -2], [-1, -1.5], [-2, 0], [-1, 1.5], [1.e-4, 0.5], [1.e-4, 1.e-4]]

    # Create instances of VisualizeOptimizer
    vo_gd = VisualizeOptimizer(objective_method="min", optimizer=grad_desc)
    vo_sgd = VisualizeOptimizer(objective_method="max", optimizer=symb_grad_opt)
    vo_sgd_2 = VisualizeOptimizer(objective_method="min", optimizer=symb_grad_opt_2)

    # Visualize 3D plots
    vo_gd.show_3d()
    vo_sgd.show_3d(initial_points=points)

    # Visualize contour plots
    vo_gd.show_contourf()
    vo_sgd.show_contourf(initial_points=points)

    # Visualize 2D plots
    vo_sgd_2.show_2d(initial_points=[[-1], [5]])


