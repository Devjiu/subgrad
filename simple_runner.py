from problems import *
from utils import *


def main(x_0_from_sol):
    # PARAMETERS
    n_exp = 1
    dim = 4

    p_param = 2
    alpha = 2
    mu = p_param / ((2 * p_param - 1) * (alpha * (dim ** (1 / 2))) ** p_param)
    epsilon = 1e-1

    # np.linalg.norm(fun.call_grad(x_0)) <= M
    # Q_radius = 6
    M = dim * alpha  # np.linalg.norm(fun.call_grad(x_0))
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
        x_solution = np.zeros(dim)
        # radius = np.random.randint(1, Q_radius // 2)
        # # solution + radius should fit into Q set.
        # if np.linalg.norm(x_solution) + radius > Q_radius:
        #     x_solution = (x_solution / np.linalg.norm(x_solution)) * (Q_radius - radius - 1)
        # print("x_solution: ", x_solution[0:6], " radius: ", radius, " solution norm: ", np.linalg.norm(x_solution),
        #       " sum: ", np.linalg.norm(x_solution) + radius)
        # points_to_cover = generate_points_to_cover(x_solution, radius, dim // 10)

        fun = Fun(p_norm_zhou(p_param), x_solution)

        # method = SubgradientDescent(strongly_convex_fun_const=mu)

        def first_order_subproblem(h_k, x_k, gr_k):
            print("h: ", h_k, " x: ", x_k, " gr_k: ", gr_k)
            c_param = h_k * gr_k - x_k * np.linalg.norm(x_k) ** (2 * p_param - 2)
            print("c: ", c_param)
            ret = (- np.linalg.norm(c_param) ** ((2 * p_param - 2) / (2 * p_param - 1))) * c_param
            print("ret: ", ret)
            return ret

        method = SubgradientVariationalInequalitiesDescent(strongly_convex_fun_const=mu,
                                                           subproblem_fun=first_order_subproblem)
        # Mirror
        # def projection_Q(x_k, h, gr, ):
        #     proj_arg = x_k - h * gr
        #     proj_norm = np.linalg.norm(proj_arg)
        #     if proj_norm <= Q_radius:
        #         return proj_arg
        #     return Q_radius * proj_arg / proj_norm
        #
        # method = SubgradientDescent(strongly_convex_fun_const=mu, projectin_Q_fun=projection_Q)
        #

        x_solutions[exp] = list(x_solution)
        f_solutions[exp] = fun.call_f(x_solution)

        x_0 = np.random.randn(dim)
        x_0 = (x_0 / np.linalg.norm(x_0)) * x_0_from_sol

        x_args_array[exp, :], f_vals_array[exp, :], f_raw_vals_array[exp, :] = method.minimize(x_0, fun, n_iter=n_iter)
        g_norm_array[exp, :] = np.array([np.linalg.norm(fun.call_grad(x)) for x in x_args_array[exp]])
        save_result("experiments", solver=method.__class__.__name__, target_fun=fun.f.__class__.__name__,
                    id="exp#{}_iter#{}_alpha#{}_dim#{}_r#{}".format(exp, n_iter, alpha, dim,
                                                                    int(np.linalg.norm(x_0 - x_solution))),
                    x_args=x_args_array, f_vals=f_vals_array, g_norm=g_norm_array, x_solution=x_solution,
                    f_solution=fun.f_solution)

    def adaptive(i):
        # this part cut due to small irrelevant values on a small amount of iterations
        if i < 70:
            return 10
        return (2 / (mu * i * (i + 1))) * np.sum(
            [k * np.square(g_norm_array[0][k]) / (k + 1) for k in range(i)], axis=0)

    draw_result("plots/SMD#{}".format(x_0_from_sol), x_args_array, f_vals_array, f_raw_vals_array, g_norm_array,
                covering_sphere_problem_strong_convex.__name__,
                x_solutions, f_solutions, [0, 0.1],
                upper_bounds={"non-adaptive": lambda i: 2 * (M ** 2) / (mu * (i + 1)), "adaptive": adaptive})
    draw_result("plots/SMD_unlimited#{}".format(x_0_from_sol), x_args_array, f_vals_array, f_raw_vals_array, g_norm_array,
                covering_sphere_problem_strong_convex.__name__,
                x_solutions, f_solutions, None,
                upper_bounds={"non-adaptive": lambda i: 2 * (M ** 2) / (mu * (i + 1)), "adaptive": adaptive})


if __name__ == '__main__':
    # [21, 3, 4].sort()
    # for rad in [5, 10, 20, 100, 1000]:
    #     main(rad)
    main(100)
