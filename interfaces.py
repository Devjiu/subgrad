import abc
from matplotlib import pyplot as plt


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


class AbstractSolver(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def minimize(self, x_0, fun: Fun, n_iter=500):
        raise RuntimeError
