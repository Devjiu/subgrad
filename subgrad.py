import abc

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize

np.random.seed(1)

# PARAMETERS
m = 50
n = 5
n_iter = 100
n_exp = 10
lam = 0.9
alpha = 0.01

f_vals_array = np.zeros((n_exp, n_iter + 1))
g_norm_array = np.zeros((n_exp, n_iter + 1))


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

    def draw_fg(self, shape, interval, points_num=500, save_fig=False):
        fig = plt.figure(figsize=(8, 4))

        x_points = []
        f_array = []
        g_array = []

        tweak = np.zeros(shape)
        print((interval[1] - interval[0]) / points_num)
        for p in np.linspace(interval[0], interval[1], points_num):
            x_points.append(p)
            tweak[0] = p
            f_array.append(self.call_f(tweak))
            g_array.append(self.call_grad(tweak))

        fig.suptitle(f'Values for f and it\'s grad. {points_num} points.')

        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylabel(f'$f(x_k) g(x_k)$')
        ax.set_xlabel('x')
        f_line, = ax.plot(x_points, f_array)
        g_line, = ax.plot(x_points, g_array)
        for x, k, y in zip(x_points, g_array, f_array):
            print("x: ", x, " k: ", k, " y: ", y)
        exit(0)
        ax.legend([f_line, g_line], [f'$f(x)$', f'$g(x)$'])
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


class least_squares_l1_reg_problem(AbstractProblem):
    def __init__(self, A, b, lam):
        super().__init__()
        self.A = A
        self.b = b
        self.lam = lam

    def __call__(self, x):
        return 0.5 * np.linalg.norm(self.A @ x - self.b, ord=2) ** 2 + lam * np.linalg.norm(x, ord=1)

    def grad(self, x):
        return self.A.T @ (self.A @ x - self.b) + lam * np.sign(x)


class covering_sphere_problem(AbstractProblem):
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


class AbstractSolver(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def minimize(self, x_0, fun: Fun, n_iter=500):
        raise RuntimeError


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
        print("mu parameter: ", strongly_convex_fun_const)
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
        return proj_arg
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
        for i_iter in range(n_iter):
            h = 2 / (self.mu * (i_iter + 1))
            gr = fun.call_grad(x)
            # print("\t radius: ", np.linalg.norm(x - points_to_cover[0]))
            x = self.projection_Q(h, gr, x)
            # x = self.argmin_subtask(h, gr, x)
            # print("\t iter:   ", i_iter)
            # print("\t h:      ", h)
            # print("\t gr:     ", gr)
            # print("\t x:      ", x)
            iterations.append(x)
            f_vals.append(fun.call_f(x))
        return iterations, f_vals


if __name__ == '__main__':
    # points_to_cover = np.array([
    #     [3, 4, 0, 0, 0],
    #     [2, 3, 0, 0, 0],
    #     [3, 2, 0, 0, 0],
    #     [4, 3, 0, 0, 0],
    #     [3, 3, 0, 0, 0]
    # ])
    # points_to_cover = np.random.rand(5, 5)
    # print(points_to_cover.shape)
    #
    # fun = Fun(covering_sphere_problem(points_to_cover))
    # method = SubgradientMirrorDescent(strongly_convex_fun_const=2)

    for exp in range(n_exp):
        points_to_cover = np.random.rand(500, 500)

        fun = Fun(covering_sphere_problem(points_to_cover))
        method = SubgradientMirrorDescent(strongly_convex_fun_const=2)

        x_0 = np.random.randn(500)
        xs, f_vals_array[exp, :] = method.minimize(x_0, fun, n_iter=n_iter)
        g_norm_array[exp, :] = np.array([np.linalg.norm(fun.call_grad(x)) for x in xs])

    print("last optimized value: ", f_vals_array[-1][-1])
    print("last optimized arg: ", xs[-1])
    print("radius: ", np.linalg.norm(xs[-1] - points_to_cover[0]))

    fig = plt.figure(figsize=(8, 4))
    fig.suptitle(f'Find minimal sphere, covering points. {n_exp} runs.')

    ax = fig.add_subplot(1, 2, 1)
    ax.set_ylabel(f'$f(x_k)$')
    ax.set_xlabel('iteration')
    ax.semilogy(f_vals_array.mean(axis=0))
    ax.fill_between(np.arange(n_iter + 1), f_vals_array.mean(axis=0) - f_vals_array.std(axis=0),
                    f_vals_array.mean(axis=0) + f_vals_array.std(axis=0), alpha=0.3)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_ylabel(f'$\|g(x_k)\|$')
    ax.set_xlabel('iteration')
    ax.semilogy(g_norm_array.mean(axis=0))
    ax.fill_between(np.arange(n_iter + 1), g_norm_array.mean(axis=0) - g_norm_array.std(axis=0),
                    g_norm_array.mean(axis=0) + g_norm_array.std(axis=0), alpha=0.3)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('SD.svg')
    # plt.show()
