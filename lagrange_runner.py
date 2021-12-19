from hashlib import md5

from scipy.optimize import minimize
from datetime import datetime
from interfaces import AbstractSolver

from methods import *
from problems import fermat_toricelly_steiner
from utils import *


def single_opt_task(seed, solver: AbstractSolver, n_iter: int, dim: int, exp: int, meta_descr: str = ""):
    np.random.seed(seed)
    raw_rand = np.random.randn(dim)
    x_0 = 0.2 * raw_rand / np.linalg.norm(raw_rand)
    print("x0 shape: ", x_0.shape)
    xs, opt_params = solver.minimize(x_0, n_iter=n_iter)
    # local variables should be faster
    f = solver.f
    g = solver.grad

    print("xs shape: ", np.array(xs).shape)
    print("data saving")
    start_time = datetime.now()
    x_averaged = averaging(xs)
    print("avg : ", (datetime.now() - start_time).total_seconds())
    start_time = datetime.now()
    f_vals = apply(x_averaged, f)
    print("f_vals : ", (datetime.now() - start_time).total_seconds())
    start_time = datetime.now()
    g_norm = apply(xs[::100], lambda x: np.linalg.norm(g(x)))
    print("g_norm : ", (datetime.now() - start_time).total_seconds())

    start_time = datetime.now()
    solver.estimate(opt_params, x_averaged)
    print("estimation : ", (datetime.now() - start_time).total_seconds())

    start_time = datetime.now()
    id = np.concatenate((x_0, solver.x_solution, hash(solver), hash(solver.f), dim, n_iter), axis=None)
    print("solver: ", solver.__class__.__name__)
    save_result(base_dir="experimental_data", solver=solver.__class__.__name__, target_fun=f.__name__,
                id="{}_exp#{}_{}".format(meta_descr, exp, md5(id).hexdigest()), x_args=xs,
                f_vals=f_vals, g_norm=g_norm,
                x_solution=solver.x_solution, f_solution=solver.f_solution, optional_parameters=opt_params)
    print("save : ", (datetime.now() - start_time).total_seconds())


def project_to_n_dim_cube(x, alpha):
    x[x > alpha] = alpha
    x[x < -alpha] = -alpha
    return x


def in_cube(x, alpha):
    return all(-alpha <= val <= alpha for val in x)


def main():
    # PARAMETERS
    n_exp = 1
    dim = 1000

    p_param = 4
    alpha = 0.2
    mu = (p_param - 1) / ((2 * p_param - 1) * (alpha * (dim ** (1 / 2))) ** p_param)
    print("mu: ", mu)
    epsilon = 1e-1

    # np.linalg.norm(fun.call_grad(x_0)) <= M
    # M = 2 * Q_radius  # np.linalg.norm(fun.call_grad(x_0))
    n_iter = 350000  # 2 * (M ** 2) / (mu * epsilon)
    print("min n_iter: ", n_iter)
    # n_iter *= 1.05
    # n_iter = int(n_iter + 1)
    print("int n_iter: ", n_iter)

    # это все вспомогательное нужно вынести куда-то
    # проблема в том, что эти функции одновременно связаны и с минимизируемой функцией, и с методом
    def first_order_subproblem_analytical(x_k, gr_k, l_k):
        c_param = (1 / l_k) * gr_k - x_k * np.linalg.norm(x_k) ** (2 * p_param - 2)
        ret = - c_param / (np.linalg.norm(c_param) ** ((2 * p_param - 2) / (2 * p_param - 1)))
        return ret

    def estimation_adaptive_scipy(x_averaged):
        def estimation(x):
            return - (np.linalg.norm(x) ** (p_param - 2)) * (x @ x_averaged) + np.linalg.norm(x) ** p_param

        y_val = estimation(
            minimize(estimation, x0=np.array([0.5] * dim), method='SLSQP', bounds=[(-alpha, alpha)] * dim)['x'])
        return y_val

    def first_order_adaptive_subproblem_scipy(x_k, gr_k, l_k):
        x_k_norm = np.linalg.norm(x_k)

        def subproblem(x):
            return gr_k @ x + l_k * (1 / (2 * p_param)) * (
                    np.linalg.norm(x) ** (2 * p_param) + (2 * p_param - 1) * x_k_norm ** (
                    2 * p_param) - 2 * p_param * x_k_norm ** (
                            2 * p_param - 2) * (x_k @ x))

        x_val = minimize(subproblem, x0=np.array([0.5] * dim), method='SLSQP', bounds=[(-alpha, alpha)] * dim)['x']
        return x_val

    def first_order_adaptive_subproblem_combined(x_k, gr_k, l_k):
        if in_cube(x_k, alpha):
            return first_order_subproblem_analytical(x_k, gr_k, l_k)
        else:
            return first_order_adaptive_subproblem_scipy(x_k, gr_k, l_k)

    def calculate_bregman_max():
        R = (alpha ** p_param) * np.sqrt((3 + (((-1) ** p_param) / p_param)) * (dim ** p_param))
        print('unknown formula - R =', R)
        return R

    x_solution = np.zeros(dim)
    method = SubgradientVIAdaptiveDescent(problem=fermat_toricelly_steiner(p_param, np.array(np.ones((3, 1000)))),
                                          x_solution=x_solution,
                                          subproblem_fun=first_order_adaptive_subproblem_combined,
                                          estimation_fun=estimation_adaptive_scipy,
                                          strongly_convex_fun_const=mu,
                                          bregman_limit=calculate_bregman_max())

    single_opt_task(0, method, n_iter, dim, 0, "sub#{}_p_param#{}".format(method.subproblem.__name__, p_param))


if __name__ == '__main__':
    init_globals()
    main()
    close_ppol()
