from interfaces import AbstractSolver, AbstractProblem
from utils import *


class SubgradientSteepestDescent(AbstractSolver):
    def __init__(self, problem: AbstractProblem, x_solution):
        super().__init__(problem.fun, problem.grad, problem.bregman, x_solution)

    def minimize(self, x_0, n_iter=500):
        iterations = []
        x = np.array(x_0)
        iterations.append(x)

        f = self.f
        g = self.grad
        x_sol = self.x_solution

        for i_iter in range(n_iter):
            gr = g(x)
            alpha = (gr.T @ (x - x_sol)) / np.square(np.linalg.norm(gr))
            x = x - alpha * gr
            iterations.append(x)
        return iterations

    def estimate(self, opt_params, x):
        raise RuntimeError("Not impl")
