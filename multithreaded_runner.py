import logging
from hashlib import md5

import cvxpy as cp
from scipy.optimize import minimize

from methods import *
from problems import *
from utils import *


def single_opt_task(seed, solver: AbstractSolver, n_iter: int, dim: int, exp: int, meta_descr: str = ""):
    np.random.seed(seed)
    raw_rand = np.random.randn(dim)
    x_0 = 0.2 * raw_rand / np.linalg.norm(raw_rand)
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
    # start_time = datetime.now()
    # g_norm = apply(xs[::100], lambda x: np.linalg.norm(g(x)))
    # print("g_norm : ", (datetime.now() - start_time).total_seconds())
    #
    # start_time = datetime.now()
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
    # print("min n_iter: ", n_iter)
    # n_iter *= 1.05
    # n_iter = int(n_iter + 1)
    # print("int n_iter: ", n_iter)

    x_args_array = np.zeros((n_exp, n_iter, dim))
    f_vals_array = np.zeros((n_exp, n_iter))
    f_raw_vals_array = np.zeros((n_exp, n_iter))
    g_norm_array = np.zeros((n_exp, n_iter))
    x_solutions = np.zeros((n_exp, dim))
    f_solutions = np.zeros(n_exp)

    format = "%(asctime)s [%(threadName)s]: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    logging.info("Main    : all done")

    def first_order_subproblem(h_k, x_k, gr_k):
        # print("h: ", h_k, " x: ", x_k, " gr_k: ", gr_k)
        c_param = h_k * gr_k - x_k * np.linalg.norm(x_k) ** (2 * p_param - 2)
        # print("c: ", c_param)
        ret = - c_param / (np.linalg.norm(c_param) ** ((2 * p_param - 2) / (2 * p_param - 1)))
        # print("ret: ", ret)
        return project_to_n_dim_cube(ret, alpha)

    def first_order_subproblem_cvx(h_k, x_k, gr_k):
        # print("h_k: ", h_k, " gr_k: ", np.linalg.norm(gr_k))
        A = np.eye(dim, dim)
        b = np.ones(dim) * alpha
        x = cp.Variable(dim)
        x_k_norm = np.linalg.norm(x_k)
        cost = cp.scalar_product(h_k * gr_k - x_k_norm ** (2 * p_param - 2) * x_k, x) + (
                1 / (2 * p_param)) * cp.norm2(x) ** (2 * p_param) + (
                       (2 * p_param - 1) / (2 * p_param)) * x_k_norm ** (2 * p_param)
        objective = cp.Minimize(cost)
        # <=, >=, == are overloaded to construct CVXPY constraints.
        constraints = [A @ x >= -b, A @ x <= b]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)
        # The optimal value for x is stored in x.value.
        # print("Updated: ", x.value[0])
        return x.value

    def first_order_subproblem_scipy(h_k, x_k, gr_k):
        x_k_norm = np.linalg.norm(x_k)

        def subproblem(x):
            return (h_k * gr_k - x_k_norm ** (2 * p_param - 2) * x_k) @ x + (
                    1 / (2 * p_param)) * np.linalg.norm(x) ** (2 * p_param) + (
                           (2 * p_param - 1) / (2 * p_param)) * x_k_norm ** (2 * p_param)

        x_val = minimize(subproblem, x0=np.array([0.5] * dim), method='SLSQP', bounds=[(-alpha, alpha)] * dim)['x']
        return x_val

    def first_order_subproblem_combined(h_k, x_k, gr_k):
        if in_cube(x_k, alpha):
            return first_order_subproblem(h_k, x_k, gr_k)
        else:
            return first_order_subproblem_scipy(h_k, x_k, gr_k)

    def estimation_scipy(x_averaged):
        def estimation(x):
            return - (np.linalg.norm(x) ** (p_param - 2)) * (x @ x_averaged) + np.linalg.norm(x) ** p_param

        y_val = estimation(
            minimize(estimation, x0=np.array([alpha] * dim), method='SLSQP', bounds=[(-alpha, alpha)] * dim)['x'])
        return y_val

    def first_order_adaptive_subproblem_cvx(x_k, gr_k, l_k):
        # print("h_k: ", h_k, " gr_k: ", np.linalg.norm(gr_k))
        A = np.eye(dim, dim)
        b = np.ones(dim) * alpha
        x = cp.Variable(dim)
        x_k_norm = np.linalg.norm(x_k)
        cost = cp.scalar_product(gr_k, x) + l_k * (1 / (2 * p_param)) * (
                cp.norm2(x) ** (2 * p_param) + (2 * p_param - 1) * x_k_norm ** (
                2 * p_param) - 2 * p_param * x_k_norm ** (
                        2 * p_param - 2) * cp.scalar_product(x_k, x))
        objective = cp.Minimize(cost)
        # <=, >=, == are overloaded to construct CVXPY constraints.
        constraints = [A @ x >= -b, A @ x <= b]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)
        # The optimal value for x is stored in x.value.
        # print("Updated: ", x.value[0])
        return x.value

    x_solution = np.zeros(dim)
    # method = SubgradientVIAdaptiveDescent(problem=p_norm_zhou(p_param),
    #                                       x_solution=x_solution,
    #                                       subproblem_fun=first_order_adaptive_subproblem_combined,
    #                                       estimation_fun=estimation_adaptive_scipy,
    #                                       strongly_convex_fun_const=mu,
    #                                       bregman_limit=calculate_bregman_max())
    method = SubgradientVIDescent(problem=p_norm_zhou(p_param),
                                  x_solution=x_solution,
                                  subproblem_fun=first_order_subproblem_combined,
                                  estimation_fun=estimation_scipy,
                                  strongly_convex_fun_const=mu,
                                  vi_rel_limitation=1)

    single_opt_task(0, method, n_iter, dim, 0, "sub#{}_p_param#{}".format(method.subproblem.__name__, p_param))


if __name__ == '__main__':
    init_globals()
    main()
