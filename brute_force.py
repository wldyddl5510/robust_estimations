"""Discretized-net baseline with the draft's outer support cutting planes."""

from functools import lru_cache
from itertools import combinations
from math import comb, isqrt
from time import perf_counter

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import eye, hstack, vstack

from utils import cutting_plane, mom_initialization


@lru_cache(maxsize=None)
def _lattice_count(dimension, budget):
    if dimension == 0:
        return 1
    if dimension == 1:
        return 2 * isqrt(budget)
    if budget < dimension:
        return 0
    return sum(
        2 * _lattice_count(dimension - 1, budget - value * value)
        for value in range(1, isqrt(budget - dimension + 1) + 1)
    )


def _lattice_points(dimension, budget):
    if dimension == 0:
        yield ()
        return
    if budget < dimension:
        return
    for value in range(1, isqrt(budget - dimension + 1) + 1):
        for tail in _lattice_points(dimension - 1, budget - value * value):
            yield (value,) + tail
            yield (-value,) + tail


def discretized_net(d, s, radius=0.25, max_points=200_000):
    """Return a deterministic radius-net of {v: ||v||_0 <= 2s, ||v||_2 <= 1}.

    Set k=min(2s,d) and grid spacing h=2*radius/sqrt(k). Rounding any v to
    this grid has Euclidean error at most radius and adds no nonzero coordinates.
    Enumerate grid points of norm <= 1+radius, then project onto the unit
    ball. Projection cannot increase distance to v, proving the covering.
    Include +/- coordinate vectors explicitly for the estimator's M bounds.
    max_points raises an error before allocating an oversized net.
    """
    if not isinstance(d, (int, np.integer)) or d < 1:
        raise ValueError("d must be a positive integer")
    if not isinstance(s, (int, np.integer)) or not 1 <= s <= d:
        raise ValueError("s must be an integer with 1 <= s <= d")
    if not 0 < radius < 1:
        raise ValueError("radius must satisfy 0 < radius < 1")

    k = min(2 * s, d)
    step = 2 * radius / np.sqrt(k)
    budget = int(np.floor(np.nextafter(k * (1 + radius)**2 / (4 * radius**2), np.inf)))
    size = 1 + 2 * d
    for dimension in range(1, k + 1):
        size += comb(d, dimension) * _lattice_count(dimension, budget)
        if max_points is not None and size > max_points:
            raise ValueError(
                f"Net exceeds max_points={max_points:,} (at least {size:,} points)."
            )

    net = np.zeros((size, d))
    net[1:d + 1] = np.eye(d)
    net[d + 1:2 * d + 1] = -np.eye(d)
    offset = 1 + 2 * d
    for dimension in range(1, k + 1):
        local = step * np.array(list(_lattice_points(dimension, budget)), dtype=float)
        local /= np.maximum(1, np.linalg.norm(local, axis=1, keepdims=True))
        for support in combinations(range(d), dimension):
            net[offset:offset + len(local), list(support)] = local
            offset += len(local)
    return net


def brute_force_estimation(
    data, s, epsilon, lambda_upper, delta=0.05, *, tol=None, seed=None, C=2,
    net_radius=0.25, max_net_points=200_000, max_iter=1000,
):
    """Return (mu_hat, info) using a full direction net and outer cutting planes.

    lambda_upper is a supplied covariance eigenvalue bound (initial choice:
    2*lambda_max(Sigma)). K is the smallest odd integer >=
    C*max(s*log(d/s), epsilon*n, log(1/delta)), with C=2 by default. Samples enter balanced,
    randomly permuted blocks; using the same seed reproduces the partition.
    tol defaults to sqrt(K*lambda_upper/n).

    Fixed-support LPs use SciPy/HiGHS; the outer support MILP uses Gurobi.

    The reported gap certifies the finite-net objective, within solver
    tolerances, not the continuous-direction objective. Check info['converged']
    before using a run as a terminated estimator. runtime includes net creation,
    block means, projections, model construction, and all optimization calls.
    """
    start = perf_counter()
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    block_means, center, current_support, best_mu, tol = mom_initialization(
        data, s, epsilon, lambda_upper, delta, tol, seed, C=C,
    )
    K, d = block_means.shape

    net = discretized_net(d, s, net_radius, max_net_points)
    medians = np.empty(len(net))
    batch_size = max(1, 1_000_000 // K)
    for first in range(0, len(net), batch_size):
        last = first + batch_size
        medians[first:last] = np.median(block_means @ net[first:last].T, axis=0)
    best_value = float(np.max(np.abs(medians - net @ best_mu)))
    # Coordinate directions imply |mu_j| <= |center_j| + F_net(mu).
    M = np.abs(center) + best_value + 1e-8 * max(1, np.max(np.abs(center)), best_value)
    # Sparse LP matrix: both signs of each net constraint, then the box bounds.
    A = hstack((
        vstack((net, -net, eye(d), -eye(d))),
        np.r_[-np.ones(2 * len(net)), np.zeros(2 * d)][:, None],
    ), format="csc")
    objective = np.r_[np.zeros(d), 1.0]

    def solve_support(support):
        lp = linprog(
            objective, A_ub=A,
            b_ub=np.r_[medians, -medians, M * support, M * support],
            bounds=[(None, None)] * d + [(0, None)], method="highs-ds",
            options={"primal_feasibility_tolerance": 1e-9,
                     "dual_feasibility_tolerance": 1e-9},
        )
        if not lp.success:
            raise RuntimeError(f"Fixed-support LP failed: {lp.message}")
        candidate = np.clip(lp.x[:d], -M, M) * support
        value = float(np.max(np.abs(medians - net @ candidate)))
        # HiGHS marginals for <= constraints are the negative multipliers.
        duals = lp.ineqlin.marginals[-2 * d:]
        gradient = M * (duals[:d] + duals[d:])
        return candidate, lp.fun, value, gradient

    best_mu, info = cutting_plane(
        s, current_support, best_mu, best_value, solve_support, tol, max_iter=max_iter,
    )
    info.update({
        "C": C, "K": K, "net_size": len(net), "runtime": perf_counter() - start,
    })
    return best_mu, info
