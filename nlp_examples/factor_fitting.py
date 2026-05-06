import cvxpy as cp
import numpy as np
from scipy.linalg import sqrtm
from util_factor_fitting import expectation_maximization, objective, PCFA_via_corr


# -----------------------------------------------------------------------------------
#                      synthetic data generation
# -----------------------------------------------------------------------------------
np.random.seed(42)
dim = 200
rank = 10
n_samples = 250
F_true = np.random.randn(dim, rank)
d_true = np.random.rand(dim) + 0.5
true_cov = F_true @ F_true.T + np.diag(d_true)
# sample from the true distribution
X = np.random.multivariate_normal(mean=np.zeros(dim), cov=true_cov, size=n_samples)
S = np.cov(X, rowvar=False)
chol_S = np.linalg.cholesky(S, upper=False)
diag_indices = np.arange(rank)

# -----------------------------------------------------------------------------------
#              expectation-maximization (EM) algorithm for factor analysis
# -----------------------------------------------------------------------------------
F0 = np.random.randn(dim, rank)
d0 = np.random.rand(dim) + 0.5
F_est, d_est = expectation_maximization(S, rank, F0, d0, max_iter=1000)
obj_EM = objective(S, F_est @ F_est.T + np.diag(d_est))


# -----------------------------------------------------------------------------------
#               Apply DNLP. We initialize using PCA.
# -----------------------------------------------------------------------------------
e = cp.Variable((dim, 1), nonneg=True)
L = cp.Variable((rank, rank), sparsity=np.tril_indices(n=rank))
G = cp.Variable((dim, rank))
F0_DNLP, d0_DNLP = PCFA_via_corr(S, rank)
e.value = 1 / d0_DNLP
G.value = (F0_DNLP / d0_DNLP) @ np.linalg.inv(sqrtm(np.eye(rank) + F0_DNLP.T @ (F0_DNLP / d0_DNLP)))


constraints = [L @ L.T == np.eye(rank) - G.T @ (G / e)]
term1 = -cp.sum(cp.log(e)) - 2 * cp.sum(cp.log(cp.diag(L)))
term2 = np.diag(S) @ e - cp.sum_squares(chol_S.T @ G)
neg_log_likelihood = term1 + term2
prob = cp.Problem(cp.Minimize(neg_log_likelihood), constraints)
prob.solve(nlp=True, verbose=True, solver=cp.IPOPT)


recovered_Sigma = np.linalg.inv(np.diag(e.value.flatten()) - G.value @ G.value.T)
obj_DNLP = objective(S, recovered_Sigma)
print("obj from EM:     ", obj_EM)
print("obj from DNLP:   ", prob.value)

