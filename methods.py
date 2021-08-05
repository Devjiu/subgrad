import numpy as np

from interfaces import AbstractSolver, Fun


class SubgradientSteepestDescent(AbstractSolver):
    def __init__(self, x_solution):
        self.x_solution = x_solution

    def minimize(self, x_0, fun: Fun, n_iter=500):
        iterations = []
        f_vals = []
        x = np.array(x_0)
        iterations.append(x)
        f_vals.append(fun.call_f(x))
        for i_iter in range(n_iter):
            print(f'x_k-1 : ', x)
            gr = fun.call_grad(x)
            alpha = (gr.T @ (x - self.x_solution)) / np.square(np.linalg.norm(gr))
            x = x - alpha * gr
            print(f'x_k : ', x)
            iterations.append(x)
            f_vals.append(fun.call_f(x))
        return iterations, f_vals


class SubgradientMirrorDescent(AbstractSolver):
    def __init__(self, strongly_convex_fun_const, q_set_radius):
        self.mu = strongly_convex_fun_const
        self.q_set_radius = q_set_radius

    def projection_Q(self, h, gr, x_k):
        proj_arg = x_k - h * gr
        proj_norm = np.linalg.norm(proj_arg)
        if proj_norm <= self.q_set_radius:
            return proj_arg
        return self.q_set_radius * proj_arg / proj_norm

    def minimize(self, x_0, fun: Fun, n_iter=500):
        iterations = []
        f_raw_vals = []
        f_vals = []
        x = np.array(x_0)
        iterations.append(x)
        x_averaged = np.sum([k * xk for k, xk in
                             enumerate(iterations, 1)], axis=0)
        f_vals.append(fun.call_f(x_averaged))
        f_raw_vals.append(fun.call_f(x))
        for i_iter in range(1, n_iter):
            h = 2 / (self.mu * (i_iter + 1))
            gr = fun.call_grad(x)
            x = self.projection_Q(h, gr, x)
            if i_iter != len(iterations):
                print("i_iter: ", i_iter, " iterations: ", len(iterations))
                raise RuntimeError("wtf i_iter wrong")
            x_averaged = np.sum([2 * k * xk / (i_iter * (i_iter + 1)) for k, xk in
                                 enumerate(iterations, 1)], axis=0)
            iterations.append(x)
            f_vals.append(fun.call_f(x_averaged))
            f_raw_vals.append(fun.call_f(x))
        return iterations, f_vals, f_raw_vals


class SubgradientDescent(AbstractSolver):
    def __init__(self, strongly_convex_fun_const):
        self.mu = strongly_convex_fun_const

    def minimize(self, x_0, fun: Fun, n_iter=500):
        iterations = []
        f_raw_vals = []
        f_vals = []
        x = np.array(x_0)
        iterations.append(x)
        x_averaged = np.sum([k * xk for k, xk in
                             enumerate(iterations, 1)], axis=0)
        f_vals.append(fun.call_f(x_averaged))
        f_raw_vals.append(fun.call_f(x))
        for i_iter in range(1, n_iter):
            h = 2 / (self.mu * (i_iter + 1))
            gr = fun.call_grad(x)
            x = x - h * gr
            if i_iter != len(iterations):
                print("i_iter: ", i_iter, " iterations: ", len(iterations))
                raise RuntimeError("wtf i_iter wrong")
            x_averaged = np.sum([2 * k * xk / (i_iter * (i_iter + 1)) for k, xk in
                                 enumerate(iterations, 1)], axis=0)
            iterations.append(x)
            f_vals.append(fun.call_f(x_averaged))
            f_raw_vals.append(fun.call_f(x))
        return iterations, f_vals, f_raw_vals
