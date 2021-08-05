import numpy as np
from matplotlib import pyplot as plt
import os


def draw_result(figname: str, x_args, f_vals, f_raw_vals, g_norm, problem_name, x_solutions, f_solutions,
                y_limits: list = None,
                upper_bounds: dict = None,
                down_bound=None):
    fig = plt.figure(figsize=(8, 12))
    print("f_vals shape: ", f_vals.shape)
    n_exp = f_vals.shape[0]
    n_it = f_vals.shape[1]
    fig.suptitle(f'Solving {problem_name}. {n_exp} runs.')

    ax = fig.add_subplot(2, 1, 1)
    ax.set_ylabel(f'$f(\widehat{{x}}) - f(x_*)$')

    print("1 shapes: ", f_vals[0].shape)
    print([f_vals[exp] - f_solutions[exp] for exp in range(n_exp)][0][0:10])
    f_discrepancy = np.array([f_vals[exp] - f_solutions[exp] for exp in range(n_exp)][0])
    print("1 disc: ", f_discrepancy.shape)

    print("minimal f discrepancy: ", f_discrepancy.min())
    ax.set_xlabel('iteration')
    if y_limits is not None:
        ax.set_ylim(y_limits)
    ax.plot(np.arange(n_it), f_discrepancy, label="discrepancy", color="g")
    # ax.semilogy(f_vals.mean(axis=0), label="discrepancy", color="g")
    # ax.fill_between(np.arange(n_it), f_vals.mean(axis=0) - f_vals.std(axis=0),
    #                 f_vals.mean(axis=0) + f_vals.std(axis=0), alpha=0.3)
    if upper_bounds is not None:
        for up_label in upper_bounds.keys():
            ax.plot(list(range(n_it)), [upper_bounds[up_label](iter) for iter in range(n_it)],
                    label=up_label)
    ax.legend(loc="upper right")

    ax = fig.add_subplot(2, 1, 2)
    ax.set_ylabel(f'$f(x_k) - f(x_*)$')
    ax.set_xlabel('iteration')
    print("2 shapes: ", f_raw_vals.shape)
    f_raw_disc = np.array([f_raw_vals[exp] - f_solutions[exp] for exp in range(n_exp)][0])
    print("minimal raw f discrepancy: ", f_raw_disc.min())
    if y_limits is not None:
        ax.set_ylim(y_limits)
    ax.plot(np.arange(n_it), f_raw_disc, label="discrepancy", color="b")
    if upper_bounds is not None:
        for up_label in upper_bounds.keys():
            ax.plot(list(range(n_it)), [upper_bounds[up_label](iter) for iter in range(n_it)],
                    label=up_label)
    ax.legend(loc="upper right")

    plt.savefig(figname + '.svg')
    # plt.show()


def save_result(base_dir: str, solver: str, target_fun: str, id: str, x_args, f_vals, g_norm,
                x_solution, f_solution):
    res_path = base_dir + "/" + solver + "/" + target_fun + "/"
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    full_res_path = res_path + id + ".npz"
    np.savez_compressed(full_res_path, x_args=x_args, f_vals=f_vals, g_norm=g_norm,
                        x_solution=x_solution, f_solution=f_solution)
    return full_res_path
