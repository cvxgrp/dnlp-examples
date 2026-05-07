import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

from util_car_control import plot_car_results

# Parallel parking example
x_final = (0.5, 0.5, -np.pi/2)
speed_bounds = (-0.15, 0.6)
steering_bounds = (-np.pi/8, np.pi/8)
a_max = 0.35
omega_max = np.pi/10
h = 0.1
N = 50
L = 0.1
gamma = 10

s_min, s_max = speed_bounds
phi_min, phi_max = steering_bounds
u_min = np.broadcast_to(np.hstack((s_min, phi_min)), (N, 2))
u_max = np.broadcast_to(np.hstack((s_max, phi_max)), (N, 2))

x = cp.Variable((N+1, 3))
u = cp.Variable((N, 2), bounds=(u_min, u_max))
x_init = np.array([0, 0, 0])

cost = cp.sum_squares(u) + gamma * cp.sum_squares(u[1:, :] - u[:-1, :])
constr = [x[0, :] == x_init, x[N, :] == x_final]
constr += [x[1:, 0] == x[:-1, 0] + h * cp.multiply(u[:, 0], cp.cos(x[:-1, 2])),
           x[1:, 1] == x[:-1, 1] + h * cp.multiply(u[:, 0], cp.sin(x[:-1, 2])),
           x[1:, 2] == x[:-1, 2] + (h / L) * cp.multiply(u[:, 0], cp.tan(u[:, 1]))]
constr += [cp.abs(u[1:, 0] - u[:-1, 0]) <= a_max * h,
           cp.abs(u[1:, 1] - u[:-1, 1]) <= omega_max * h]

# sample bounds + best_of are needed for IPOPT to find a feasible solution
x.sample_bounds = ([-2, -2, -np.pi], [2, 2, np.pi])
prob = cp.Problem(cp.Minimize(cost), constr)
prob.solve(solver=cp.IPOPT, nlp=True, verbose=False, best_of=100)

x_opt, u_opt = x.value, u.value
print(f"Final: p1={x_opt[-1, 0]:.3f}, p2={x_opt[-1, 1]:.3f}, theta={x_opt[-1, 2]:.3f}")

fig = plot_car_results(x_opt, u_opt, L, h)
fig.savefig("parallel_parking.pdf", bbox_inches="tight", dpi=300)
print("Saved parallel_parking.pdf")
plt.show()
