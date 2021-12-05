import math
from datetime import datetime

import numpy as np
from scipy.optimize import minimize

n = 1000  # the dimension of the problem
p = 4  # this parameter in the prox function and Bregman divergence
M = 1  # f is M-relative Lipschitz
a = 0.5  # this is the upper boud of the cube Q = [-a,a]^n
mu = (p - 1) / ((2 * p - 1) * (np.power(a * math.sqrt(n), p)))
print('mu =', mu)


# the objective function. 1-relative Lipschitz and strongly relative
def f(x):
    return (np.linalg.norm(x) ** p) / p


# operator g = the gradient of the function f: grad(f)=g(x)
def g(x):
    return (np.linalg.norm(x) ** (p - 2)) * x


# the prox function
def h(x):
    return (1 / (2 * p)) * (np.power(np.linalg.norm(x), 2 * p))


# the gradient of the prox function h
def grad_h(x):
    return (np.linalg.norm(x) ** (2 * p - 2)) * x


# The Bregmann Divergence for the prox function h
def V(y, x):  # x,y in R^{n}.
    term_1 = np.power(np.linalg.norm(y), 2 * p)
    term_2 = (2 * p - 1) * np.power(np.linalg.norm(x), 2 * p)
    term_3 = (2 * p) * np.power(np.linalg.norm(x), 2 * p - 2) * np.dot(x, y)
    return (1 / (2 * p)) * (term_1 + term_2 - term_3)


# L_0
# L0 = np.linalg.norm(np.subtract(g([1] + [0] * (n - 1)), g([0, 1] + [0] * (n - 2)))) / math.sqrt(2)
# print("L_0 =", L0)

x0 = np.array([0.5] * n)
R = (a ** p) * math.sqrt((3 + (((-1) ** p) / p)) * (n ** p))

print('R =', R)
print('f(x0) =', f(x0))

bnds = [(-a, a)] * n


def arg_min_new(r, s):  # r=x_k is a vector in R^n, s=h_k is a constant in R
    initial = x0

    def subproblem(x):
        return s * (g(r) @ x) + V(x, r)

    return minimize(subproblem, initial, method='SLSQP', bounds=bnds)['x']


def arg_min_old(r, s):  # r=x_k is a vector in R^n, s=L_{k+1} is a constant in R
    initial = x0
    h = lambda x: np.dot(g(r), x) + s * V(x, r)
    return minimize(h, initial, method='SLSQP', bounds=bnds)['x']


def in_cube(x):
    return all(-a <= val <= a for val in x)


# Algorithm 2: Adaptation to Inexactness for Relatively Bounded VI's. this is algorithm 2 in the paper
delta_0 = 0.5


### New subgradient Algorithm for VI
def Subgrad_VT(iterations_list):
    Time_Subgrad_VT = []
    estimate_Subgrad_VT = []

    for K in iterations_list:
        x = x0  # this is for the first iteration
        start_time = datetime.now()

        for k in range(0, K):
            h_k = 2 / (mu * (k + 1))
            # print ('h_k = ' , h_k)

            if in_cube(x):
                c_sol = h_k * g(x) - grad_h(x)  # c in the solution for x^{k+1}
                theta = np.linalg.norm(c_sol) ** ((2 - 2 * p) / (2 * p - 1))
                x = -1 * theta * c_sol
            else:
                x = arg_min_new(x, h_k)

        print("f(x_N): ", f(x), " f(zero): ", f(np.zeros(n)), " f(x_0): ", f(x0))
        end_time = datetime.now()
        estimate = round(2 * (M ** 2) / (mu * (K + 1)), 8)
        Time_algorithm = (end_time - start_time).total_seconds()
        Time_Subgrad_VT.append(round(Time_algorithm, 3))
        estimate_Subgrad_VT.append(estimate)

    return Time_Subgrad_VT, estimate_Subgrad_VT


results_Subgrad_VT = Subgrad_VT([1000])

Time_Subgrad_VT = results_Subgrad_VT[0]
estimate_Subgrad_VT = results_Subgrad_VT[1]

print('The results of Subgrad for VT')
print('-----------------------------')

print('Time =', Time_Subgrad_VT)
print('estimate =', estimate_Subgrad_VT)
