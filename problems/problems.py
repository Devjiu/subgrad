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
