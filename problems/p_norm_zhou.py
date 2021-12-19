import numpy as np

from interfaces import AbstractProblem


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
