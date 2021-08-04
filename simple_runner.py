from matplotlib import pyplot as plt

from methods import *
from problem_parmeters_generator import generate_points_to_cover
from problems import *

np.random.seed(1)


def draw_result(x_args, f_vals, f_raw_vals, g_norm, problem_name, x_solutions, f_solutions,
                upper_bounds: list = None,
                down_bound=None):
    fig = plt.figure(figsize=(8, 16))
    print("f_vals shape: ", f_vals.shape)
    n_exp = f_vals.shape[0]
    n_it = f_vals.shape[1]
    fig.suptitle(f'Solving {problem_name}. {n_exp} runs.')

    ax = fig.add_subplot(4, 1, 1)
    ax.set_ylabel(f'$f(\widehat{{x}}) - f(x_*)$')

    print("1 shapes: ", f_vals[0].shape)
    print([f_vals[exp] - f_solutions[exp] for exp in range(n_exp)][0][0:10])
    f_discrepancy = np.array([f_vals[exp] - f_solutions[exp] for exp in range(n_exp)][0])
    print("1 disc: ", f_discrepancy.shape)

    print("minimal f discrepancy: ", f_discrepancy.min())
    ax.set_xlabel('iteration')
    ax.set_ylim([0, 10])
    ax.plot(np.arange(n_it), f_discrepancy)
    # ax.semilogy(f_vals.mean(axis=0))
    # ax.fill_between(np.arange(n_it), f_vals.mean(axis=0) - f_vals.std(axis=0),
    #                 f_vals.mean(axis=0) + f_vals.std(axis=0), alpha=0.3)

    if upper_bounds is not None:
        for upper_bound in upper_bounds:
            ax.plot(list(range(n_it)), [upper_bound(iter) for iter in range(n_it)])

    ax = fig.add_subplot(4, 1, 2)
    ax.set_ylabel(f'$f(x_k) - f(x_*)$')
    ax.set_xlabel('iteration')
    print("2 shapes: ", f_raw_vals.shape)
    f_raw_disc = np.array([f_raw_vals[exp] - f_solutions[exp] for exp in range(n_exp)][0])
    print("minimal raw f discrepancy: ", f_raw_disc.min())
    ax.set_ylim([0, 10])
    ax.plot(np.arange(n_it), f_raw_disc)
    if upper_bounds is not None:
        for upper_bound in upper_bounds:
            ax.plot(list(range(n_it)), [upper_bound(iter) for iter in range(n_it)])

    ax = fig.add_subplot(4, 1, 3)
    ax.set_ylabel(f'$x_k - x_*$')
    ax.set_xlabel('iteration')
    print("3 shapes: ", x_args[0].shape, " x_sol: ", x_solutions[0].shape)
    x_discrepancy = [x_args[exp] - x_solutions[exp] for exp in range(n_exp)]
    x_discrepancy_norm = np.zeros((n_exp, n_iter))
    for exp in range(n_exp):
        for x_dicr_iter in x_discrepancy[exp]:
            x_discrepancy_norm[exp] = np.linalg.norm(x_dicr_iter)
    ax.plot(np.arange(n_it), x_discrepancy_norm[0])

    ax = fig.add_subplot(4, 1, 4)
    ax.set_ylabel(f'$f(x_k)$')
    ax.set_xlabel('iteration')
    ax.plot(np.arange(n_it), f_raw_vals[0])
    ax.plot(np.arange(n_it), f_solutions[0] * np.ones(n_it))

    # ax = fig.add_subplot(1, 3, 2)
    # ax.set_ylabel(f'$\|x_k - x_*\|$')
    # ax.set_xlabel('iteration')
    # x_args = np.array([x_args[exp] - x_solutions[exp] for exp in range(n_exp)])
    # ax.set_xlabel('iteration')
    # ax.semilogy(x_args.mean(axis=0))
    # ax.fill_between(np.arange(n_iter + 1), x_args.mean(axis=0) - x_args.std(axis=0),
    #                 x_args.mean(axis=0) + x_args.std(axis=0), alpha=0.3)

    # ax = fig.add_subplot(1, 2, 2)
    # ax.set_ylabel(f'$\|g(x_k)\|$')
    # ax.set_xlabel('iteration')
    # ax.semilogy(g_norm.mean(axis=0))
    # ax.fill_between(np.arange(n_iter + 1), g_norm.mean(axis=0) - g_norm.std(axis=0),
    #                 g_norm.mean(axis=0) + g_norm.std(axis=0), alpha=0.3)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('SD.svg')
    plt.show()


if __name__ == '__main__':

    # PARAMETERS
    m = 50
    n = 5
    n_exp = 1
    lam = 0.9
    alpha = 0.01
    dim = 1000

    mu = 2
    epsilon = 1e-2

    # np.linalg.norm(fun.call_grad(x_0)) <= M
    Q_radius = 10
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

    for exp in range(n_exp):
        print("============exp#{}============".format(exp))
        x_solution = np.random.randn(dim)
        radius = np.random.randint(1, Q_radius // 2)
        # solution + radius should fit into Q set.
        if np.linalg.norm(x_solution) + radius > Q_radius:
            x_solution = (x_solution / np.linalg.norm(x_solution)) * (Q_radius - radius - 1)
        print("x_solution: ", x_solution[0:6], " radius: ", radius, " solution norm: ", np.linalg.norm(x_solution),
              " sum: ", np.linalg.norm(x_solution) + radius)
        points_to_cover = generate_points_to_cover(x_solution, radius, dim // 10)

        fun = Fun(covering_sphere_problem(points_to_cover))
        method = SubgradientMirrorDescent(strongly_convex_fun_const=mu, q_set_radius=Q_radius)

        x_solutions[exp] = list(x_solution)
        f_solutions[exp] = fun.call_f(x_solution)

        x_0 = np.random.randn(dim)

        x_args_array[exp, :], f_vals_array[exp, :], f_raw_vals_array[exp, :] = method.minimize(x_0, fun, n_iter=n_iter)
        g_norm_array[exp, :] = np.array([np.linalg.norm(fun.call_grad(x)) for x in x_args_array[exp]])

        print("last optimized value: ", f_vals_array[exp][-1])
        print("solution value      : ", f_solutions[exp])
        # print("last optimized arg: ", x_args_array[exp][-1])
        # print("solution arg      : ", x_solution)
        print(f'$x_k - x_*$:', np.linalg.norm(x_args_array[exp] - x_solution))
        print("radius: ", np.linalg.norm(x_args_array[exp] - points_to_cover[0]))
        print(f'solution radius:', np.linalg.norm(x_solution - points_to_cover[0]))
    draw_result(x_args_array, f_vals_array, f_raw_vals_array, g_norm_array, covering_sphere_problem.__name__,
                x_solutions, f_solutions,
                upper_bounds=[lambda i: 2 * (M ** 2) / (mu * (i + 1)), lambda i: 2 * np.sum(
                    [k * np.square(g_norm_array[k]) / (k + 1) for k in range(n_iter)]) / (mu * (i + 1))])
