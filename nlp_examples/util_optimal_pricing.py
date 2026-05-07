import numpy as np
import scipy.sparse as sparse


def construct_elasticity(n: int):
    """Constructs a sparse elasticity matrix with random entries.
    
    Args:
        n (int): Size of the square matrix
        
    Returns:
        scipy.sparse.csr_matrix: Sparse elasticity matrix
    """
    
    blocks = n // 10

    E = sparse.lil_matrix((n, n))

    for i in range(blocks):
        r = slice(i*10, (i+1)*10)
        E[r, r] = np.random.uniform(-0.5, 0.5, (10, 10))

    E.setdiag(np.random.uniform(-3.0, -1.0, n))

    return E.tocsr()


def generate_data(n: int, seed: int=1):
    """Generate synthetic data for profit optimization problem.
    
    Args:
        n (int): Number of products
        seed (int, optional): Random seed. Defaults to 1.
        
    Returns:
        Tuple[ProfitData, ConstraintData]: Profit and constraint data
    """
    
    np.random.seed(seed)
    
    m = n // 5

    r_nom = 1 + 4 * np.random.rand(n)
    kappa_nom = 0.85 * r_nom
    elasticity = construct_elasticity(n)    
    pi_min, pi_max = np.log(0.9), np.log(1.1)
    delta_min, delta_max = np.log(0.8), np.log(1.2)
    C = np.random.randn(n, m)

    return pi_min, pi_max, delta_min, delta_max, r_nom, kappa_nom, elasticity, C