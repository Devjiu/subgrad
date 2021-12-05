from datetime import datetime

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
