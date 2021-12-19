from interfaces import AbstractSolver, AbstractProblem
from utils import *


class SubgradientDescent(AbstractSolver):
    def __init__(self, problem: AbstractProblem, x_solution, projection: callable, strongly_convex_fun_const):
        super().__init__(problem.fun, problem.grad, problem.bregman, x_solution)
        self.projection = projection
        self.mu = strongly_convex_fun_const
        # self.q_set_radius = q_set_radius

    # def projection_Q(self, h, gr, x_k):
    #     proj_arg = x_k - h * gr
    #     proj_norm = np.linalg.norm(proj_arg)
    #     if proj_norm <= self.q_set_radius:
    #         return proj_arg
    #     return self.q_set_radius * proj_arg / proj_norm

    def minimize(self, x_0, n_iter=500):
        iterations = []
        x = np.array(x_0)
        iterations.append(x)

        f = self.f
        g = self.grad
        project = self.projection

        for i_iter in range(1, n_iter):
            h = 2 / (self.mu * (i_iter + 1))
            gr = g(x)
            x = project(x, h, gr)
            if i_iter != len(iterations):
                print("i_iter: ", i_iter, " iterations: ", len(iterations))
                raise RuntimeError("wtf i_iter wrong")
            iterations.append(x)
        return iterations

    def estimate(self, opt_params, x):
        raise RuntimeError("Not impl")
