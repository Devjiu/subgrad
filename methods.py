import time

import numpy as np
from scipy import optimize

from interfaces import AbstractSolver, Fun


class SubgradientDescent(AbstractSolver):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def minimize(self, x_0, fun: Fun, n_iter=500):
        iterations = []
        f_vals = []
        x = np.array(x_0)
        iterations.append(x)
        f_vals.append(fun.call_f(x))
        for i_iter in range(n_iter):
            # print(f'x_k-1 : ', x)
            x = x - self.alpha * fun.call_grad(x)
            # print(f'x_k : ', x)
            iterations.append(x)
            f_vals.append(fun.call_f(x))
        return iterations, f_vals


class SubgradientSteepestDescent(AbstractSolver):
    def minimize(self, x_0, fun: Fun, n_iter=500):
        iterations = []
        f_vals = []
        x = np.array(x_0)
        iterations.append(x)
        f_vals.append(fun.call_f(x))
        for i_iter in range(n_iter):
            print(f'x_k-1 : ', x)
            gr = fun.call_grad(x)
            alpha = (gr.T @ (x - x_solution)) / np.square(np.linalg.norm(gr))
            x = x - alpha * gr
            print(f'x_k : ', x)
            iterations.append(x)
            f_vals.append(fun.call_f(x))
        return iterations, f_vals


class SubgradientMirrorDescent(AbstractSolver):
    def __init__(self, strongly_convex_fun_const):
        # print("mu parameter: ", strongly_convex_fun_const)
        self.mu = strongly_convex_fun_const

    @staticmethod
    def argmin_subtask(h, gr, x_k):
        def f(x):
            return h * (gr.T @ (x - x_k)) + np.square(np.linalg.norm(x - x_k)) / 2

        x_0 = np.zeros((5, 1))
        # set x in Q, where Q is simplex
        A_matrix = np.zeros((5, 5))
        A_matrix[0, :] = 1
        left_border = np.zeros(5)
        left_border[0] = 0
        right_border = np.zeros(5)
        right_border[0] = 1
        linear_constraint = optimize.LinearConstraint(A_matrix.tolist(), left_border, right_border)
        vec_res = optimize.minimize(f, x_0, tol=1e-6, constraints=linear_constraint)
        print("\t\tres default: ", vec_res.fun)
        return vec_res.x

    @staticmethod
    def projection_Q(h, gr, x_k):
        proj_arg = x_k - h * gr
        # return proj_arg
        proj_norm = np.linalg.norm(proj_arg)
        rad_q_sphere = 1
        if proj_norm <= rad_q_sphere:
            return proj_arg
        return rad_q_sphere * proj_arg / proj_norm

    def minimize(self, x_0, fun: Fun, n_iter=500):
        iterations = []
        f_vals = []
        x = np.array(x_0)
        iterations.append(x)
        f_vals.append(fun.call_f(x))
        start_time = round(time.time() * 1000)
        for i_iter in range(n_iter):
            h = 2 / (self.mu * (i_iter + 1))
            gr = fun.call_grad(x)
            # print("\t radius: ", np.linalg.norm(x - points_to_cover[0]))
            x = self.projection_Q(h, gr, x)
            # x = self.argmin_subtask(h, gr, x)
            # if i_iter % 100 == 0:
            #     print("\t iter:   ", i_iter)
            #     print("\t h:      ", h)
            #     print("\t gr:     ", gr)
            #     print("\t x:      ", x)
            iterations.append(x)
            f_vals.append(fun.call_f(np.sum([2 * i_iter * xk / (n_iter * (n_iter - 1)) for xk in iterations])))
            end_iter_time = round(time.time() * 1000)
            if i_iter % 1000 == 1:
                print("expected: ", (end_iter_time - start_time) * n_iter/(i_iter * 60_000))
        return iterations, f_vals
