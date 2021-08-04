import abc


class AbstractProblem(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, x):
        raise RuntimeError

    @abc.abstractmethod
    def grad(self, x):
        raise RuntimeError


class Fun:
    def __init__(self, fun: AbstractProblem):
        self.f = fun

    def call_f(self, x):
        return self.f(x)

    def call_grad(self, x):
        return self.f.grad(x)


class AbstractSolver(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def minimize(self, x_0, fun: Fun, n_iter=500):
        raise RuntimeError
