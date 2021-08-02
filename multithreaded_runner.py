import concurrent.futures
import logging
import multiprocessing

from matplotlib import pyplot as plt

from methods import *
from problems import *


def single_opt_task(seed, solver: AbstractSolver, n_iter):
    np.random.seed(seed)
    fun = Fun(covering_sphere_problem(np.random.rand(500, 500)))
    x_0 = np.random.randn(500)
    xs, f_vals = solver.minimize(x_0, fun, n_iter=n_iter)
    g_norm = np.array([np.linalg.norm(fun.call_grad(x)) for x in xs])
    return f_vals, g_norm


if __name__ == '__main__':

    # PARAMETERS
    m = 50
    n = 5
    n_iter = 1_000
    n_exp = 100
    lam = 0.9
    alpha = 0.01

    f_vals_array = np.zeros((n_exp, n_iter + 1))
    g_norm_array = np.zeros((n_exp, n_iter + 1))

    format = "%(asctime)s [%(threadName)s]: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    logging.info("Main    : all done")

    n_exp = 3
    lam = 0.9
    alpha = 0.01
    dim = 500

    x_args_array = [n_exp * []]
    f_vals_array = [n_exp * []]
    g_norm_array = [n_exp * []]
    x_solutions = np.zeros((n_exp, dim))
    f_solutions = np.zeros(n_exp)

    for exp in range(n_exp):
        print("============exp#{}============".format(exp))
        x_solution = np.random.randn(dim)
        radius = np.random.randint(0, 1_000)
        # print("x_solution: ", x_solution, " radius: ", radius)
        points_to_cover = generate_points_to_cover(x_solution, radius, dim)

        mu = 2
        epsilon = 1e-1
        fun = Fun(covering_sphere_problem(points_to_cover))
        method = SubgradientMirrorDescent(strongly_convex_fun_const=mu)

        x_solutions[exp] = list(x_solution)
        f_solutions[exp] = fun.call_f(x_solution)

        x_0 = np.random.randn(dim)
        # np.linalg.norm(fun.call_grad(x_0)) <= M
        M = np.linalg.norm(fun.call_grad(x_0))
        n_iter = 2 * (M ** 2) / (mu * epsilon)
        print("min n_iter: ", n_iter)
        n_iter *= 1.2
        n_iter = int(n_iter + 1)
        print("int n_iter: ", n_iter)
        x_args_array[exp, :], f_vals_array[exp, :] = method.minimize(x_0, fun, n_iter=n_iter)
        g_norm_array[exp, :] = np.array([np.linalg.norm(fun.call_grad(x)) for x in x_args_array[exp]])

        # print("last optimized value: ", f_vals_array[exp][-1])
        # print("last optimized arg: ", xs[exp])
        print(f'$x_k - x_*$:', np.linalg.norm(x_args_array[exp] - x_solution))
        print("radius: ", np.linalg.norm(x_args_array[exp] - points_to_cover[0]))
        print(f'solution radius:', np.linalg.norm(x_solution - points_to_cover[0]))
    draw_result(x_args_array, f_vals_array, g_norm_array, covering_sphere_problem.__name__, n_exp, x_solutions,
                f_solutions, upper_bound=lambda i: 2 * (M ** 2) / (mu * (i + 1)))



    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        method = SubgradientMirrorDescent(strongly_convex_fun_const=2)
        future_to_exp_num = {
            executor.submit(single_opt_task, exp, method, n_iter): exp for exp in range(n_exp)}
        for future in concurrent.futures.as_completed(future_to_exp_num):
            exp_num = future_to_exp_num[future]
            try:
                f_vals, g_norm = future.result()
                f_vals_array[exp_num, :] = f_vals
                g_norm_array[exp_num, :] = g_norm
            except Exception as exc:
                print('%r generated an exception: %s' % (f_vals, exc))
            else:
                print("f: ", f_vals[-1], " g: ", g_norm[-1])

    fig = plt.figure(figsize=(8, 4))
    fig.suptitle(f'Solving rounding points. {n_exp} runs.')

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
    plt.savefig('multithreaded_SD.svg')
