import numpy as np

from interfaces import AbstractProblem


class fermat_toricelly_steiner(AbstractProblem):
    """
    Этот класс описывает оператор, потому значение функции не имеет особого смысла и плохо протестировано
    по факту fun - это функция Лагранжа
    """

    def __init__(self, points_to_cover, alpha_matrix):
        super().__init__()
        self.points = points_to_cover
        self.alpha_m = alpha_matrix

    def fun(self, x, lmbd):
        f = np.sum([np.linalg.norm(x - point) for point in self.points],  axis=0)
        phi = np.array([alpha @ np.abs(x) - 1 for alpha in self.alpha_m])
        return f + lmbd @ phi

    # f(x) >= f(x_0) + <g, x- x_0>
    def grad(self, x, lmbd):
        print("ps: ", self.points[0].shape)
        nabla_f = np.sum([(x - point) / np.linalg.norm(x - point) for point in self.points],  axis=0)
        nabla_phi = np.array([alpha * np.sign(x) for alpha in self.alpha_m]).tolist()
        print("grad: ", (nabla_f + lmbd @ nabla_phi).shape)
        phi = np.array([alpha @ np.abs(x) - 1 for alpha in self.alpha_m])
        print("alph: ", self.alpha_m[0].shape)
        print("phi 0: ", phi)
        concatenated = np.concatenate([nabla_f + lmbd @ nabla_phi, phi], axis=0)
        print("concated: ", concatenated.shape)
        return concatenated

    def hess(self, x):
        return 2

    def bregman(self, x, y):
        pass


if __name__ == '__main__':
    p = fermat_toricelly_steiner(np.array(np.ones((5, 1000))), np.array(np.ones((3, 1000))))
    x = np.random.randn(1000)
    l = np.array([0.5, 1, 2])
    p.grad(x, l)