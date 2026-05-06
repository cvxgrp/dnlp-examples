import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Generate toy data with outliers
# ---------------------------------------------------------------------------
np.random.seed(42)
m = 50
a_true, lambda_true, c_true = 5.0, 0.3, 1.0
sigma = 0.3
t = np.linspace(0, 10, m)
y = a_true * np.exp(-lambda_true * t) + c_true + sigma * np.random.randn(m)

# Corrupt ~7 points with large outliers
outlier_idx = np.random.choice(m, size=7, replace=False)
y[outlier_idx] += np.random.uniform(3, 4, size=7)

# ---------------------------------------------------------------------------
# Least-squares fit
# ---------------------------------------------------------------------------
a_ls, lmbda_ls, c_ls = cp.Variable(), cp.Variable(nonneg=True), cp.Variable()
residuals_ls = y - (a_ls * cp.exp(-lmbda_ls * t) + c_ls)
prob_ls = cp.Problem(cp.Minimize(cp.sum_squares(residuals_ls)))

a_ls.value, lmbda_ls.value, c_ls.value = 4.0, 0.5, 0.5
prob_ls.solve(nlp=True, solver=cp.IPOPT)

print("Least-squares fit")
print(f"  a={a_ls.value:.3f}, lambda={lmbda_ls.value:.3f}, c={c_ls.value:.3f}")

# ---------------------------------------------------------------------------
# Huber fit  
# ---------------------------------------------------------------------------
M = sigma
a_hub, lmbda_hub, c_hub = cp.Variable(), cp.Variable(nonneg=True), cp.Variable()
residuals_hub = y - (a_hub * cp.exp(-lmbda_hub * t) + c_hub)
prob_hub = cp.Problem(cp.Minimize(cp.sum(cp.huber(residuals_hub, M=M))))

a_hub.value, lmbda_hub.value, c_hub.value = 4.0, 0.5, 0.5
prob_hub.solve(nlp=True, solver=cp.IPOPT)

print("\nHuber fit")
print(f"  a={a_hub.value:.3f}, lambda={lmbda_hub.value:.3f}, c={c_hub.value:.3f}")

print(f"\nTrue parameters: a={a_true}, lambda={lambda_true}, c={c_true}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
t_fine = np.linspace(0, 10, 200)
y_true = a_true * np.exp(-lambda_true * t_fine) + c_true
y_ls = a_ls.value * np.exp(-lmbda_ls.value * t_fine) + c_ls.value
y_hub = a_hub.value * np.exp(-lmbda_hub.value * t_fine) + c_hub.value

fig, ax = plt.subplots(figsize=(8, 5))
inlier_mask = np.ones(m, dtype=bool)
inlier_mask[outlier_idx] = False
ax.scatter(t[inlier_mask], y[inlier_mask], color='gray', s=20, label='Measurements')
ax.scatter(t[outlier_idx], y[outlier_idx], color='red', s=30, marker='x', label='Outliers')
ax.plot(t_fine, y_true, 'k--', linewidth=1.5, label='True model')
ax.plot(t_fine, y_ls, 'r-', linewidth=2, label='Least-squares fit')
ax.plot(t_fine, y_hub, 'b-', linewidth=2, label='Huber fit')
ax.set_xlabel('$t$', fontsize=14)
ax.set_ylabel('$y(t)$', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig("figures/exponential_decay_huber.pdf", bbox_inches="tight", dpi=300)
print("\nSaved figures/exponential_decay_huber.pdf")
