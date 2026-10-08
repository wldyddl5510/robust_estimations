"""Dense Algorithm 1, without an outer support-selection MIP."""
from time import perf_counter
import numpy as np
import gurobipy as gp
from utils import mom_initialization
from IP_algorithm import dense_estimation


def non_sparse_estimation(data, epsilon, lambda_upper, delta=.05, *,
                          tol=1e-5, seed=None, C=2, max_inner_iter=1000):
    start = perf_counter()
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError('data must be an (n,d) array')
    d = data.shape[1]
    blocks, center, _, _, tol = mom_initialization(
        data, d, epsilon, lambda_upper, delta, tol, seed, C=C, complexity=d)
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        estimate, info = dense_estimation(blocks, center, d, tol, env=env,
                                         max_inner_iter=max_inner_iter)
    gap = max(0., info['objective']-info['lower_bound'])
    info.update(converged=gap <= tol, gap=gap, tol=tol, C=C, K=len(blocks),
                direction_sparsity=d, block_complexity=d, runtime=perf_counter()-start)
    return estimate, info
