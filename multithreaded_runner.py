import logging
import threading
import time
import numpy as np
import concurrent.futures
from matplotlib import pyplot as plt

from methods import *
from problems import *


def single_opt_task(solver: AbstractSolver, n_iter):
    np.random.seed(17)
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
    n_exp = 10
    lam = 0.9
    alpha = 0.01

    f_vals_array = np.zeros((n_exp, n_iter + 1))
    g_norm_array = np.zeros((n_exp, n_iter + 1))

    format = "%(asctime)s [%(threadName)s]: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    logging.info("Main    : all done")

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        method = SubgradientMirrorDescent(strongly_convex_fun_const=2)
        future_to_exp_num = {
            executor.submit(single_opt_task, method, n_iter): exp for exp in range(n_exp)}
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
