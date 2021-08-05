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
