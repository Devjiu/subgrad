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
        # these 2 matrices should be syncshronized in dimensons
        # assuming dim is dimension of original space
        # so we should have n points, so point should be (n * dim)
        # alpha matrix is connected with limitations for problems
        # so it should be (m * dim) for m constraints
        assert self.points.shape[1] == self.alpha_m.shape[
            1], "Input matrices are wrong shape {} != {}, types: {}, {}".format(self.points.shape[1],
                                                                                self.alpha_m.shape[1],
                                                                                type(self.points.shape[1]),
                                                                                type(self.alpha_m.shape[1]))
        self.constrains_num = self.alpha_m.shape[0]
        self.original_dim_num = self.alpha_m.shape[1]
        print("constrints num: ", self.constrains_num, " original dim: ", self.original_dim_num)

    def fun(self, x: np.array):
        assert x.shape[0] == (self.constrains_num + self.original_dim_num), "Unexpected shape dim: {} != {}".format(
            x.shape, self.constrains_num + self.original_dim_num)
        x_orig = x[: self.original_dim_num]
        lmbd = x[-self.constrains_num:]
        # print("x_orig: ", x_orig.shape, " lmbd: ", lmbd.shape)
        f = np.max([np.square(np.linalg.norm(x_orig - point)) for point in self.points])
        phi = (np.square(x_orig) @ self.alpha_m.T) - 5
        # print("shapes f: ", f.shape, " lmbd: ", lmbd.shape, " phi: ", phi.shape)
        return f + lmbd @ phi - (1 / 2) * (lmbd @ lmbd)

    # f(x) >= f(x_0) + <g, x- x_0>
    def grad(self, x):
        assert x.shape[0] == self.constrains_num + self.original_dim_num, "Unexpected shape dim: ({})".format(
            x.shape)
        x_orig = x[: self.original_dim_num]
        lmbd = x[-self.constrains_num:]
        # print("ps: ", self.points[0].shape)
        fun_values = [np.square(np.linalg.norm(x_orig - point)) for point in self.points]
        max_ind = np.concatenate(np.argwhere(fun_values == np.max(fun_values))).tolist()
        # print("f_vals: ", fun_values, " ret: ", 2 * (x - self.points[max_ind[0]]))
        nabla_f = 2 * (x_orig - self.points[max_ind[0]])
        nabla_phi = 2 * (np.diag(lmbd.T @ self.alpha_m) @ x_orig)
        # print("mul shape         : ", (self.alpha_m[:, 0] * x_orig[0]).shape)
        # print("lmbd shape        : ", lmbd.shape)
        # print("derivative 1 shape: ", nabla_f.shape)
        # print("derivative 2 shape: ", (lmbd @ nabla_phi).shape)
        # print("grad: ", (nabla_f + lmbd @ nabla_phi).shape)
        phi = np.array([alpha @ np.square(x_orig) - 5 for alpha in self.alpha_m])
        # print("alph: ", self.alpha_m[0].shape)
        # print("phi 0: ", phi)
        concatenated = np.concatenate([nabla_f + nabla_phi, lmbd - phi], axis=0)
        # print("concated: ", concatenated.shape)
        return concatenated

    def hess(self, x):
        return 2

    # Используется евклидова прокс функция. Необходимо доказать
    # относительную Липшицевость для оператора
    # По факту - в случае Евклидова прокса, относительная Липщицевость - это то же, что и обычная.
    def bregman(self, x, y):
        return (1 / 2) * np.linalg.norm(x) ** 2 + (1 / 2) * np.linalg.norm(y) ** 2 - y @ x


if __name__ == '__main__':
    p = fermat_toricelly_steiner(np.array(np.ones((5, 1000))), np.array(np.ones((3, 1000))))
    x = np.random.randn(1000)
    l = np.array([0.5, 1, 2])
    p.grad(x, l)
