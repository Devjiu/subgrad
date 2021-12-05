import abc


class AbstractProblem(abc.ABC):
    @abc.abstractmethod
    def fun(self, x):
        pass

    @abc.abstractmethod
    def grad(self, x):
        pass

    @abc.abstractmethod
    def bregman(self, x, y):
        pass


class AbstractSolver(abc.ABC):
    def __init__(self, fun: callable, grad: callable, bregman: callable, x_solution):
        self.f = fun
        self.grad = grad
        self.bregman = bregman
        self.x_solution = x_solution
        self.f_solution = self.f(x_solution)

    @abc.abstractmethod
    def minimize(self, x_0, n_iter=500):
        pass

    @abc.abstractmethod
    def estimate(self, opt_params, x):
        pass
