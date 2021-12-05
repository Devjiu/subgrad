import numpy as np

from interfaces import AbstractProblem


class least_squares_l1_reg_problem(AbstractProblem):
    def __init__(self, A, b, lam):
        super().__init__()
        self.A = A
        self.b = b
        self.lam = lam

    def __call__(self, x):
        return 0.5 * np.linalg.norm(self.A @ x - self.b, ord=2) ** 2 + self.lam * np.linalg.norm(x, ord=1)

    def grad(self, x):
        return self.A.T @ (self.A @ x - self.b) + self.lam * np.sign(x)


class covering_sphere_problem_strong_convex(AbstractProblem):
    def __init__(self, points_to_cover):
        super().__init__()
        self.points = points_to_cover

    def __call__(self, x):
        return np.max([np.square(np.linalg.norm(x - point)) for point in self.points])

    # f(x) >= f(x_0) + <g, x- x_0>
    def grad(self, x):
        fun_values = [np.square(np.linalg.norm(x - point)) for point in self.points]
        max_ind = np.concatenate(np.argwhere(fun_values == np.max(fun_values))).tolist()
        # print("f_vals: ", fun_values, " ret: ", 2 * (x - self.points[max_ind[0]]))
        return 2 * (x - self.points[max_ind[0]])

    def hess(self, x):
        return 2


class covering_sphere_problem_convex(AbstractProblem):
    def __init__(self, points_to_cover):
        super().__init__()
        self.points = points_to_cover

    def __call__(self, x):
        return np.max([np.linalg.norm(x - point) for point in self.points])

    # f(x) >= f(x_0) + <g, x- x_0>
    def grad(self, x):
        fun_values = [np.linalg.norm(x - point) for point in self.points]
        max_ind = np.concatenate(np.argwhere(fun_values == np.max(fun_values))).tolist()
        x_meets = x - self.points[max_ind[0]]
        return x_meets / np.linalg.norm(x_meets)

    def hess(self, x):
        return 2


# relative_lipschitz_relative_strongly_convex
class p_norm_zhou(AbstractProblem):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def fun(self, x):
        return (np.linalg.norm(x) ** self.p) / self.p

    # f(x) >= f(x_0) + <g, x- x_0>
    def grad(self, x):
        return (np.linalg.norm(x) ** (self.p - 2)) * x

    def hess(self, x):
        return (np.linalg.norm(x) ** (self.p - 2)) * np.eye(x.shape[0], x.shape[0]) + (self.p - 2) * (
                np.linalg.norm(x) ** (self.p - 4)) * (x.T * x)

    def bregman(self, x, y):
        x_norm = np.linalg.norm(x)
        y_norm = np.linalg.norm(y)
        return (1 / (2 * self.p)) * (
                x_norm ** (2 * self.p) + (2 * self.p - 1) * y_norm ** (2 * self.p) - 2 * self.p * y_norm ** (
                2 * self.p - 2) * (y @ x.T))
