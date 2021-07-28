import cvxpy as cp
import numpy as np
from scipy import optimize

# Интересующий нас метод приведён в конце
# стр. 13. Существенно, что правая часть оценки скорости его сходимости не зависит
# от R (расстояния от точки старта до решения x_*). Можно это всё реализовать для
# задачи о нахождении шара наименьшего радиуса, покрывающего фиксированный набор
# точек (A_1, A_2, …, A_N) в пространстве R^n. То есть минимизируется функция
# f(X) = max{XA_1^2, XA_2^2, …, XA_N^2} (находим X —  центр шара). Такая функция негладкая
# за счёт максимизации, квадраты нужны, чтобы сильная выпуклость была. А условие Липшица можно
# гарантировать, если ограничить допустимое множество Q. Его можно взять шаром евклидовым
# (например, с центром в 0) содержащим точки A_1, A_2, …, A_N.


points_to_cover = np.array([
    [3, 2, 0, 0, 0],
    [4, 3, 0, 0, 0],
    [3, 4, 0, 0, 0],
    [2, 3, 0, 0, 0],
    [3, 3, 0, 0, 0]
])

# Create two scalar optimization variables.
x = cp.Variable((5, 1))
x_0 = np.array([-1, -1, 0, 0, 0]).reshape((5, 1))


def f(x, points):
    print("x: ", x, " point: ", points[0], " f val: ", np.max(np.hstack([np.square(x.T @ p) for p in points])))
    return np.max(np.hstack([np.square(x.T @ p) for p in points]))


vec_res = optimize.minimize(f, x_0, args=(points_to_cover,), method="BFGS")
print("vec res: ", vec_res)
exit(0)
func = cp.max(cp.hstack([cp.square(x.T @ p) for p in points_to_cover]))

x.value = np.array([-1, -1, 0, 0, 0]).reshape((5, 1))
print("expr.value = ", func.value)

# .
constraints = []

# Form objective.
obj = cp.Minimize(func)

# Form and solve problem.
prob = cp.Problem(obj)
prob.solve()  # Returns the optimal value.

print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value)
