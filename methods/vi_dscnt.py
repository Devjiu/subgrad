from datetime import datetime

from interfaces import AbstractSolver, AbstractProblem
from utils import *


class SubgradientVIDescent(AbstractSolver):
    def __init__(self, problem: AbstractProblem, x_solution, subproblem_fun: callable, estimation_fun: callable,
                 strongly_convex_fun_const, vi_rel_limitation):
        super().__init__(problem.fun, problem.grad, problem.bregman, x_solution)
        self.subproblem = subproblem_fun
        self.estimation = estimation_fun
        self.mu = strongly_convex_fun_const
        self.M = vi_rel_limitation

    def minimize(self, x_0, n_iter=500):
        iterations = []
        optional_parameters = {"time": []}
        x = np.array(x_0)
        iterations.append(x)

        # frequently called function and variables should be saved to local variables (faster to acccess)
        f = self.f
        g = self.grad
        subproblem = self.subproblem

        start_time = datetime.now()
        for i_iter in range(1, n_iter):
            h = 2 / (self.mu * (i_iter + 1))
            gr = g(x)
            # difficult to calculate in general, I have to switch to current examples impl
            # bregman_divergence = lambda _x, _y: fun.proximal_f(_y) - fun.proximal_f(_x) - gr.T @ (_y - _x)
            x = subproblem(h, x, gr)
            if i_iter != len(iterations):
                print("i_iter: ", i_iter, " iterations: ", len(iterations))
                raise RuntimeError("wtf i_iter wrong")
            iterations.append(x)
        end_time = datetime.now()
        total = (end_time - start_time).total_seconds()
        print("total time: ", total)
        optional_parameters["time"].append(total)
        return np.array(iterations), optional_parameters

    def estimate(self, opt_params, x_avraged):
        def theoretical_est(n):
            return (2 * self.M) / (self.mu * (n + 1))

        theoretical = apply(range(0, len(x_avraged), 100), theoretical_est)
        opt_params['theoretical_est'] = theoretical
