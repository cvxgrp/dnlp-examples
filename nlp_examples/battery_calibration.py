import cvxpy as cp
import numpy as np

data = np.load("data/battery_data.npz")                                                                                                                                                                                                                
i, v_meas, q1, h = data["i"], data["v_meas"], data["q1"], data["h"]
K = len(i)

# precompute charge
q = np.zeros(K)
q[0] = q1
for k in range(K - 1):
    q[k + 1] = q[k] + h * i[k]

a_min, a_max = 1.0, 10.0
b_min, b_max = 100.0, 1000.0
Q_crit_min, Q_crit_max = 6000, 10000
R_min, R_max = 0.01, 0.3
C1_min, C1_max = 500.0, 2000.0

# using this lower bound halves number of iterations for IPOPT to converge
#Q_crit_min = np.max(q)

v = cp.Variable(K)
v_oc = cp.Variable(K)
a = cp.Variable(bounds=[a_min, a_max])
b = cp.Variable(bounds=[b_min, b_max])
Q_crit = cp.Variable(bounds=[Q_crit_min, Q_crit_max])
R0 = cp.Variable(bounds=[R_min, R_max])
R1 = cp.Variable(bounds=[R_min, R_max])
C1 = cp.Variable(bounds=[C1_min, C1_max])
U_RC = cp.Variable(K)

constrs = [v == v_oc + R0 * i + U_RC, v_oc == a + b / (Q_crit - q),
           U_RC[1:] == (1 - h / (R1 * C1)) * U_RC[:-1] + (h / C1) * i[:-1],
           U_RC[0] == 0.0]
obj = cp.Minimize(cp.sum_squares(v - v_meas))
prob = cp.Problem(obj, constrs)
prob.solve(nlp=True, verbose=True)

print(f"Estimated parameters:")
print(f"  a: {a.value:.2f}")
print(f"  b: {b.value:.2f}")
print(f"  Q_crit: {Q_crit.value:.2f}")
print(f"  R0: {R0.value:.2f}")
print(f"  R1: {R1.value:.2f}")
print(f"  C1: {C1.value:.2f}")
