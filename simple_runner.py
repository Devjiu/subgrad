from matplotlib import pyplot as plt

from methods import *
from problems import *

np.random.seed(1)


def draw_result(f_vals, g_norm, problem_name, experiments_num):
    fig = plt.figure(figsize=(8, 4))
    fig.suptitle(f'Solving {problem_name}. {n_exp} runs.')

    ax = fig.add_subplot(1, 2, 1)
    ax.set_ylabel(f'$f(x_k)$')
    ax.set_xlabel('iteration')
    ax.semilogy(f_vals.mean(axis=0))
    ax.fill_between(np.arange(n_iter + 1), f_vals.mean(axis=0) - f_vals.std(axis=0),
                    f_vals.mean(axis=0) + f_vals.std(axis=0), alpha=0.3)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_ylabel(f'$\|g(x_k)\|$')
    ax.set_xlabel('iteration')
    ax.semilogy(g_norm.mean(axis=0))
    ax.fill_between(np.arange(n_iter + 1), g_norm.mean(axis=0) - g_norm.std(axis=0),
                    g_norm.mean(axis=0) + g_norm.std(axis=0), alpha=0.3)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('SD.svg')
    # plt.show()


if __name__ == '__main__':

    # PARAMETERS
    m = 50
    n = 5
    n_iter = 10_000
    n_exp = 100
    lam = 0.9
    alpha = 0.01

    f_vals_array = np.zeros((n_exp, n_iter + 1))
    g_norm_array = np.zeros((n_exp, n_iter + 1))

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
    draw_result(f_vals_array, g_norm_array, covering_sphere_problem.__name__, n_exp)
