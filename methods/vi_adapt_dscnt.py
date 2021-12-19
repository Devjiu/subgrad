from datetime import datetime

from interfaces import AbstractSolver, AbstractProblem
from utils import *


class SubgradientVIAdaptiveDescent(AbstractSolver):
    def __init__(self, problem: AbstractProblem, x_solution, subproblem_fun: callable, estimation_fun: callable,
                 strongly_convex_fun_const, bregman_limit):
        super().__init__(problem.fun, problem.grad, problem.bregman, x_solution)
        self.subproblem = subproblem_fun
        self.estimation = estimation_fun
        self.mu = strongly_convex_fun_const
        self.r = bregman_limit

    def minimize(self, x_0, n_iter=500):
        iterations = []
        optional_parameters = {"l": [], "delta": [], "time": []}
        x = np.array(x_0)
        iterations.append(x)
        l_k = 1
        delta = 0.2

        f = self.f
        g = self.grad
        breg = self.bregman
        subproblem = self.subproblem

        optional_parameters["l"].append(l_k)
        optional_parameters["delta"].append(delta)

        start_time = datetime.now()
        for i_iter in range(1, n_iter):
            gr = g(x)
            x_k = x
            x = subproblem(x, gr, l_k)
            if gr @ (x - x_k) + l_k * breg(x, x_k) + delta < 0:
                l_k *= 2
                delta *= 2
            else:
                l_k /= 2
                delta /= 2

            if i_iter != len(iterations):
                print("i_iter: ", i_iter, " iterations: ", len(iterations))
                raise RuntimeError("wtf i_iter wrong")
            iterations.append(x)
            optional_parameters["l"].append(l_k)
            optional_parameters["delta"].append(delta)
        end_time = datetime.now()
        total = (end_time - start_time).total_seconds()
        print("total time: ", total)
        optional_parameters["time"].append(total)
        return np.array(iterations), optional_parameters

    def estimate(self, opt_params, x_averaged):
        def theoretical_estimation(comb):
            (lk, dk, r_param) = (comb[0], comb[1], comb[2])
            rev = np.reciprocal(lk)
            s = np.sum(rev)
            partial = np.sum(rev * dk)
            return (r_param ** 2 + partial) / s

        ls = opt_params["l"]
        ds = opt_params["delta"]
        r = self.r
        est_params = [(lk, dk, r) for lk, dk in list(zip(ls, ds))[::100]]
        theoretical = apply(est_params, theoretical_estimation)
        opt_params["theoretical_est"] = theoretical
