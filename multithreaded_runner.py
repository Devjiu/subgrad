import concurrent.futures
import logging
import multiprocessing
import os
from hashlib import md5

from methods import *
from problem_parmeters_generator import *
from problems import *


def save_result(base_dir: str, solver: str, target_fun: str, id: str, x_args, f_vals, g_norm,
                x_solution, f_solution):
    res_path = base_dir + "/" + solver + "/" + target_fun + "/"
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    full_res_path = res_path + id + ".npz"
    np.savez_compressed(full_res_path, x_args=x_args, f_vals=f_vals, g_norm=g_norm,
                        x_solution=x_solution, f_solution=f_solution)
    return full_res_path


def single_opt_task(seed, fun: Fun, solver: AbstractSolver, n_iter: int, dim: int, exp: int):
    np.random.seed(seed)
    x_0 = np.random.randn(dim)
    xs, f_vals, f_raw_vals = solver.minimize(x_0, fun, n_iter=n_iter)
    g_norm = np.array([np.linalg.norm(fun.call_grad(x)) for x in xs])
    id = np.concatenate((x_0, fun.x_solution, hash(solver), hash(fun), dim, n_iter), axis=None)
    save_result(base_dir="experimental_data", solver=solver.__class__.__name__, target_fun=fun.f.__class__.__name__,
                id="exp#{}_{}".format(exp, md5(id).hexdigest()), x_args=xs,
                f_vals=f_vals, g_norm=g_norm,
                x_solution=fun.x_solution, f_solution=fun.f_solution)
    return None, None


def main():
    # PARAMETERS
    n_exp = 3
    dim = 1000

    mu = 2
    epsilon = 1e-1

    # np.linalg.norm(fun.call_grad(x_0)) <= M
    Q_radius = 6
    M = 2 * Q_radius  # np.linalg.norm(fun.call_grad(x_0))
    n_iter = 2 * (M ** 2) / (mu * epsilon)
    print("min n_iter: ", n_iter)
    n_iter *= 1.2
    n_iter = int(n_iter + 1)
    print("int n_iter: ", n_iter)

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

    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        future_to_exp_num = dict()
        method = SubgradientMirrorDescent(strongly_convex_fun_const=2, q_set_radius=Q_radius)
        for exp in range(n_exp):
            x_solution = np.random.randn(dim)
            radius = np.random.randint(1, Q_radius // 2)
            if np.linalg.norm(x_solution) + radius > Q_radius:
                x_solution = (x_solution / np.linalg.norm(x_solution)) * (Q_radius - radius - 1)
            points_to_cover = generate_points_to_cover(x_solution, radius, dim // 10)

            fun = Fun(covering_sphere_problem_strong_convex(points_to_cover), x_solution)
            future_to_exp_num[executor.submit(single_opt_task, exp, fun, method, n_iter, dim, exp)] = exp

        for future in concurrent.futures.as_completed(future_to_exp_num):
            exp_num = future_to_exp_num[future]
            try:
                future.result()
                # f_vals_array[exp_num, :] = f_vals
                # g_norm_array[exp_num, :] = g_norm
            except Exception as exc:
                print('%r generated an exception in {}' % (exp_num, exc))
            else:
                print("succes for {}".format(exp_num))

    # plt.savefig('multithreaded_SD.svg')


if __name__ == '__main__':
    main()
