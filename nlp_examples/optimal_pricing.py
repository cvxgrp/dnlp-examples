import cvxpy as cp
import numpy as np
from util_optimal_pricing import construct_elasticity

np.random.seed(0)  
n = 1000
r_nom = 1 + 4 * np.random.rand(n, 1)
kappa_nom = 0.85 * r_nom
E = construct_elasticity(n)    
pi_min, pi_max = np.log(0.9), np.log(1.1)

delta = cp.Variable((n, 1))
pi = cp.Variable((n, 1), bounds=[pi_min, pi_max])
profit = cp.sum(cp.multiply(r_nom, cp.exp(delta + pi)) - cp.multiply(kappa_nom, cp.exp(delta)))
constr = [delta == E @ pi]
problem = cp.Problem(cp.Maximize(profit), constr)
problem.solve(nlp=True, verbose=True)


profit_nominal = np.sum(r_nom - kappa_nom)
profit_optimized = profit.value
print(f"Nominal profit: {profit_nominal:.2f}")
print(f"Optimized profit: {profit_optimized:.2f}")
print(f"Relative improvement: {(profit_optimized - profit_nominal) / profit_nominal * 100:.2f}%")