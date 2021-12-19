import time

import cvxpy as cp
import numpy as np

start_tm = time.time()
# Problem data.
n = 7
alpha = 0.2

A = np.eye(n, n)
b = np.ones(n) * alpha

# Construct the problem.
x = cp.Variable(n)
# *, +, -, / are overloaded to construct CVXPY objects.
h_k = 1
g_k = np.random.randn(n)
x_k = np.random.randn(n)
cost = cp.scalar_product((h_k * g_k - np.linalg.norm(x_k) ** 2) * x_k, x) + (1 / 4) * cp.norm2(x) ** 2
objective = cp.Minimize(cost)
# <=, >=, == are overloaded to construct CVXPY constraints.
constraints = [-b <= A @ x, A @ x <= b]
prob = cp.Problem(objective, constraints)

# The optimal objective is returned by prob.solve().
result = prob.solve()
print("time for solution: ", time.time() - start_tm)
# The optimal value for x is stored in x.value.
print("argmin: ", x.value)
print("result: ", result)
