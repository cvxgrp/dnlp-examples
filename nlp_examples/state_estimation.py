"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

3-bus state estimation example (Abur & Expósito textbook system).
Solves the WLS state estimation problem via DNLP.
"""

import numpy as np

import cvxpy as cp

# =============================================================================
# Section 1: Build 3×3 admittance matrix
# =============================================================================
z12 = 0.01 + 0.03j
z13 = 0.02 + 0.05j
z23 = 0.03 + 0.08j

y12, y13, y23 = 1.0 / z12, 1.0 / z13, 1.0 / z23

Y_bus = np.array([
    [y12 + y13,  -y12,      -y13],
    [-y12,       y12 + y23,  -y23],
    [-y13,       -y23,      y13 + y23],
])
G = np.real(Y_bus)
B = np.imag(Y_bus)

# =============================================================================
# Section 2: Group measurements by type with index arrays
# =============================================================================
# Voltage magnitude measurements
v_buses = [0, 1]
z_v = np.array([1.006, 0.968])
sigma_v = 0.004

# Power injection measurements (generator convention: negate load values)
inj_buses = [1]
z_pinj_meas, z_qinj_meas = 0.501, 0.286  # textbook values (load convention, positive)
z_pinj = np.array([-z_pinj_meas])
z_qinj = np.array([-z_qinj_meas])
sigma_inj = 0.01

# Line flow measurements (from → to)
flow_from = [0, 0]
flow_to = [1, 2]
z_pf = np.array([0.888, 1.173])
z_qf = np.array([0.568, 0.663])
sigma_flow = 0.008

# =============================================================================
# Section 3: DNLP formulation
# =============================================================================
n = 3
theta = cp.Variable((n, 1))
v = cp.Variable((n, 1))
C, S = cp.cos(theta - theta.T), cp.sin(theta - theta.T)
P = cp.multiply(v @ v.T, cp.multiply(G, C) + cp.multiply(B, S))
Q = cp.multiply(v @ v.T, cp.multiply(G, S) - cp.multiply(B, C))
p, q = cp.sum(P, axis=1), cp.sum(Q, axis=1)

# Vectorized line flows (no shunt admittances)
Pf = P[flow_from, flow_to] - cp.multiply(v[flow_from, 0] ** 2, G[flow_from, flow_to])
Qf = Q[flow_from, flow_to] + cp.multiply(v[flow_from, 0] ** 2, B[flow_from, flow_to])

# Weighted residual vector (8 measurements, 5 vectorized groups)
r = cp.hstack([
    (v[v_buses, 0] - z_v) / sigma_v,
    (p[inj_buses] - z_pinj) / sigma_inj,
    (q[inj_buses] - z_qinj) / sigma_inj,
    (Pf - z_pf) / sigma_flow,
    (Qf - z_qf) / sigma_flow,
])

prob = cp.Problem(cp.Minimize(cp.sum_squares(r)), [theta[0] == 0])

# =============================================================================
# Section 4: Flat start + solve
# =============================================================================
v.value = np.ones((n, 1))
theta.value = np.zeros((n, 1))
prob.solve(nlp=True, solver=cp.IPOPT, verbose=True)

v_est = v.value.flatten()
theta_est = theta.value.flatten()

# =============================================================================
# Section 5: Print results
# =============================================================================
print("\n" + "=" * 50)
print("DNLP STATE ESTIMATION RESULTS")
print("=" * 50)
print(f"Solver status: {prob.status}")
print(f"WLS objective J(x_hat) = {prob.value:.6f}")
print()
print(f"{'Bus':>5} {'|V| (pu)':>10} {'θ (deg)':>10}")
print("-" * 30)
for i in range(n):
    print(f"{i+1:>5} {v_est[i]:>10.4f} {np.degrees(theta_est[i]):>10.2f}")
