from datetime import datetime
from hashlib import md5

from scipy.optimize import minimize

from interfaces import AbstractSolver
from methods import *
from problems import fermat_toricelly_steiner
from problems.problem_parmeters_generator import generate_points_to_cover
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

    print("x solution: ", solver.x_solution)
    print("F solution: ", f(solver.x_solution))

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
    g_norm_f = apply(xs, lambda x: np.linalg.norm(g(x)))
    print("g_norm adpt: ", (datetime.now() - start_time).total_seconds())

    start_time = datetime.now()
    solver.estimate_adaptive(opt_params, g_norm_f)
    print("estimation adaptive : ", (datetime.now() - start_time).total_seconds())

    print("x_avg shape: ", x_averaged.shape)
    for x_it in range(0, x_averaged.shape[0], 100):
        n_it = np.linalg.norm(solver.x_solution - x_averaged[x_it])
        print(f"[{x_it:3d}] x - x*: {n_it:.2f}, f: {solver.f(x_averaged[x_it]):.3f}")

    for x_it in range(0, x_averaged.shape[0], 100):
        n_it = np.abs(solver.f(solver.x_solution) - solver.f(x_averaged[x_it]))
        print(f"[{x_it:3d}] f(x) - f(x*): {n_it:.3f}")

    start_time = datetime.now()
    id = np.concatenate((x_0, solver.x_solution, hash(solver), hash(solver.f), dim, n_iter), axis=None)
    print("solver: ", solver.__class__.__name__)
    save_result(base_dir="experimental_data", solver=solver.__class__.__name__, target_fun=f.__qualname__,
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
    original_dim = 10000
    constraints_dim = 1000
    dim = original_dim + constraints_dim

    Q_radius = 6
    x_solution = np.random.randn(original_dim)
    radius = np.random.randint(1, Q_radius // 2)
    # solution + radius should fit into Q set.
    if np.linalg.norm(x_solution) + radius > Q_radius:
        x_solution = (x_solution / np.linalg.norm(x_solution)) * (Q_radius - radius - 1)
    print("x_solution: ", x_solution[0:6], " radius: ", radius, " solution norm: ", np.linalg.norm(x_solution),
          " sum: ", np.linalg.norm(x_solution) + radius)
    points_to_cover = generate_points_to_cover(x_solution, radius, original_dim // 10)
    print("points to cover shape: ", points_to_cover.shape)
    phi_params_alpha = np.zeros((constraints_dim, original_dim))
    for c in range(constraints_dim):
        col = np.abs(np.random.randn(original_dim))
        col = col / np.sum(col)
        print("com sum: ", np.sum(col))
        phi_params_alpha[c] = col
    print("alpha shape: ", phi_params_alpha.shape)
    mu = 1
    print("mu: ", mu)
    epsilon = 1e-1

    # np.linalg.norm(fun.call_grad(x_0)) <= M
    M = 16 * (Q_radius ** 2)  # np.linalg.norm(fun.call_grad(x_0))
    print(f"M = {M:.3f}")
    n_iter = 1000  # 2 * (M ** 2) / (mu * epsilon)
    print("min n_iter: ", n_iter)
    # n_iter *= 1.05
    # n_iter = int(n_iter + 1)
    print("int n_iter: ", n_iter)

    def proj_Q(x):
        if np.linalg.norm(x) > Q_radius:
            # print("outside of Q, projecting original: ", ret, "norm: ", np.linalg.norm(ret))
            x = (x / np.linalg.norm(x)) * Q_radius
        for constr_ind in range(1, constraints_dim + 1):
            if x[-constr_ind] < 0:
                x[-constr_ind] = 0
        return x

    # это все вспомогательное нужно вынести куда-то
    # проблема в том, что эти функции одновременно связаны и с минимизируемой функцией, и с методом
    def first_order_subproblem_analytical(h_k, x_k, gr_k):
        ret = x_k - h_k * gr_k
        # new value should fit into Q set, so we check and project
        # print("first order ret: ", ret, " radius: ", Q_radius, " ret norm: ", np.linalg.norm(ret))
        return proj_Q(ret)

    phi = (np.square(x_solution) @ phi_params_alpha.T) - 5
    aw = np.argwhere(phi >= 0)
    print("first aw: ", aw, " phi: ", phi[phi >= 0])
    while len(aw) > 0:
        print("aw: ", aw)
        for c in aw:
            print(" c: ", c)
            col = np.abs(np.random.randn(original_dim))
            col = col / np.sum(col)
            phi_params_alpha[c] = col
        aw = np.argwhere(phi > 0)
    # exit(0)
    print("alpha: ", phi_params_alpha[0][:20])
    fun = fermat_toricelly_steiner(points_to_cover, phi_params_alpha)

    def estimation_adaptive_scipy(x_averaged):
        def estimation_x(x):
            lmbd = x_averaged[-constraints_dim:]
            f = np.max([np.square(np.linalg.norm(x - point)) for point in points_to_cover])
            phi = (np.square(x) @ phi_params_alpha.T) - 5
            return f + lmbd @ phi - (1 / 2) * (lmbd @ lmbd)

        lmbd = np.array(phi_params_alpha @ np.square(x_averaged[:original_dim]) - 5)
        for constr_ind in range(0, constraints_dim):
            if lmbd[constr_ind] < 0:
                lmbd[constr_ind] = 0
        l_hacked = np.concatenate([x_averaged[:original_dim], lmbd], axis=0)
        y_lmbd_max = fun.fun(l_hacked)
        # print("lmbd max: ",
        # np.array([alpha @ np.square(x_averaged[:original_dim]) - 5 for alpha in phi_params_alpha]))
        # print("lmbd max calc: ",
        #       np.array(phi_params_alpha @ np.square(x_averaged[:original_dim]) - 5))
        x_orig = x_averaged[:original_dim]
        lmbd_avg = x_averaged[-constraints_dim:]
        fun_values = [np.square(np.linalg.norm(x_orig - point)) for point in points_to_cover]
        max_ind = np.concatenate(np.argwhere(fun_values == np.max(fun_values))).tolist()
        x_mins = []
        for point in points_to_cover:
            tmp = np.linalg.inv(np.diag(lmbd_avg.T @ phi_params_alpha + 1)) @ point
            if np.linalg.norm(tmp) > Q_radius:
                print("projecting: ", tmp[:3], "norm: ", np.linalg.norm(tmp))
                tmp = (tmp / np.linalg.norm(tmp)) * Q_radius
            x_mins.append(tmp)
        max_norms = []
        for x_min in x_mins:
            norms = []
            for point in points_to_cover:
                norms.append(np.linalg.norm(x_min - point))
            # print("x min: ", x_min[:3], " norm: ", np.max(norms))
            max_norms.append(np.max(norms))
        print("x min: ", x_mins[np.concatenate(np.argwhere(max_norms == np.max(max_norms))).tolist()[0]][:3],
              " norm: ", np.max(max_norms))
        x_min = x_mins[np.concatenate(np.argwhere(max_norms == np.max(max_norms))).tolist()[0]]
        x_min_hacked = np.concatenate([x_min, lmbd_avg], axis=0)
        if np.linalg.norm(x_min_hacked) > Q_radius:
            # print("outside of Q, projecting original: ", ret, "norm: ", np.linalg.norm(ret))
            x_min_hacked = (x_min_hacked / np.linalg.norm(x_min_hacked)) * Q_radius
        x_min_scp = \
            minimize(estimation_x, x0=np.zeros(original_dim), method='SLSQP',
                     bounds=[(-Q_radius, Q_radius)] * original_dim)['x']
        x_hacked = np.concatenate([x_min_scp, lmbd_avg], axis=0)
        print("x_min    : ", x_min[:20])
        print("x_min_scp: ", x_min_scp[:20])
        y_x_min = fun.fun(x_hacked)
        y_val = y_lmbd_max - y_x_min
        print(
            f"max - min  : {y_lmbd_max:.3f} - {y_x_min:.3f} = {y_val:.3f} : f(x~): {fun.fun(x_averaged):.3f} : f(calc): {fun.fun(x_min_hacked):.3f}")
        return y_val

    lmbd_solution = np.zeros((constraints_dim))
    x_solution = np.concatenate([x_solution, lmbd_solution], axis=0)
    print("phi vals: ", ((np.square(x_solution[:original_dim]) @ phi_params_alpha.T) - 5)[:10])
    print("f val: ", fun.fun(x_solution))
    method = SubgradientVIDescent(
        problem=fun,
        x_solution=x_solution,
        subproblem_fun=first_order_subproblem_analytical,
        estimation_fun=estimation_adaptive_scipy,
        strongly_convex_fun_const=mu,
        vi_rel_limitation=M)

    single_opt_task(0, method, n_iter, dim, 0,
                    "sub#{}_num_points#{}".format(method.subproblem.__name__, points_to_cover.shape[0]))


if __name__ == '__main__':
    init_globals()
    main()
    close_ppol()
