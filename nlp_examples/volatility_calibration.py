import numpy as np
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)

data = np.load('data/vol_calibration.npz')
implied_var = data['implied_var']
strikes = data['strikes']
exdate = str(data['exdate'])
F = float(data['F'])
T = (pd.to_datetime(exdate) - pd.to_datetime('2025-06-04')).days / 365.0
k = np.log(strikes / F)

N = len(k)
print("Number of data points:", N)
# define optimization problem
a_min, a_max = -1, 1
b_min, b_max = 0, 10
rho_min, rho_max = -1, 1
m_min, m_max = -2, 2
s_min, s_max = 0, 1.0

a = cp.Variable(bounds=[a_min, a_max])
b = cp.Variable(bounds=[b_min, b_max])
rho = cp.Variable(bounds=[rho_min, rho_max])
m = cp.Variable(bounds=[m_min, m_max])
s = cp.Variable(bounds=[s_min, s_max])

w = (1/T) * (a + b * (rho * (k - m) + cp.sqrt((k - m) ** 2 + s)))
objective = cp.Minimize(cp.sum_squares(w - implied_var))
problem = cp.Problem(objective)
problem.solve(nlp=True, verbose=True, best_of=10)

# plot implied volatility smile
k_plot = np.linspace(min(k), max(k), 100)
w_plot = (1/T) * (a.value + b.value * (rho.value * (k_plot - m.value) + np.sqrt((k_plot - m.value) ** 2 + s.value)))
plt.plot(k, np.sqrt(implied_var), 'rx', markersize=8)
plt.plot(k_plot, np.sqrt(w_plot), 'b--')
plt.xlabel('Log-Moneyness (k)', fontsize=12)
plt.ylabel('Implied Volatility', fontsize=12)
plt.savefig("figures/volatility_calibration.pdf")

print("Calibrated parameters:")
print(f"a = {a.value}")
print(f"b = {b.value}")
print(f"rho = {rho.value}")
print(f"m = {m.value}")
print(f"s = {s.value}")