import numpy as np
import numpy.linalg as LA


def expectation_maximization(S, rank, F, d, max_iter=200, eta=0):   
    s = np.diag(S)   
    for _ in range(1, max_iter + 1):
        # E-step
        G = np.linalg.inv((F.T * (1 / d.reshape(1, -1))) @ F + np.eye(rank))
        B = G @ F.T * (1 / d.reshape(1, -1))
        Cxz = S @ B.T
        Czz = B @ Cxz + G

        # M-step
        F = Cxz @ np.linalg.inv(Czz)
        d = np.maximum(s - 2*np.sum(Cxz*F, axis=1) + np.sum(F * (F @ Czz), axis=1), eta)
       
    return F, d

def objective(S, Sigma):
    term1 = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(Sigma))))
    term2 = np.trace(np.linalg.solve(Sigma, S))
    return term1 + term2

def PCFA_via_corr(sigma, k):
    vola = np.sqrt(np.diag(sigma)).reshape(-1, 1)
    R = (1 / vola) * sigma * (1 / vola).T
    lmbda, Q = LA.eigh(R)
    lmbda = lmbda[::-1][0:k]
    Q = Q[:, ::-1][:, 0:k]

    # low-rank approximation of correlation matrix
    F = (Q @ np.diag(np.sqrt(lmbda)))
    d = np.diag(R - F @ F.T)
   
    # scale so it becomes low rank approximation of covariance matrix
    F = vola * F 
    d = (np.squeeze(vola)**2) * d
    
    return F, d.reshape(-1, 1)