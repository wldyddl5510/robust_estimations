"""Discretized-net baseline with the draft's outer support cutting planes."""

from functools import lru_cache
from itertools import combinations
from math import comb, isqrt
from time import perf_counter

import gurobipy as gp
import numpy as np

from utils import cutting_plane, mom_initialization


class NetSizeError(ValueError):
    """The full covering net exceeds the allocation limit."""

    def __init__(self, size, max_points):
        self.size = size
        super().__init__(f"Net has {size:,} points; exceeds max_points={max_points:,}.")


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
    if not 0 < radius <= 1:
        raise ValueError("radius must satisfy 0 < radius <= 1")

    k = min(2 * s, d)
    step = 2 * radius / np.sqrt(k)
    budget = int(np.floor(np.nextafter(k * (1 + radius)**2 / (4 * radius**2), np.inf)))
    size = 1 + 2 * d
    for dimension in range(1, k + 1):
        size += comb(d, dimension) * _lattice_count(dimension, budget)
    if max_points is not None and size > max_points:
        raise NetSizeError(size, max_points)

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


def _support_extremes(net, medians, support):
    """Keep the median extrema for each distinct active-coordinate direction."""
    active = np.flatnonzero(support)
    groups = (np.unique(net[:, active], axis=0, return_inverse=True)[1]
              if len(active) else np.zeros(len(net), dtype=int))
    order = np.lexsort((medians, groups))
    starts = np.r_[0, np.flatnonzero(np.diff(groups[order])) + 1]
    ends = np.r_[starts[1:] - 1, len(net) - 1]
    indices = np.r_[order[starts], order[ends]]
    return net[indices], medians[indices]


def solve_fixed_support_lp(net, medians, support, M, *, env=None):
    """Solve the net LP, returning (feasible_mu, L, cut_gradient)."""
    if len(net) > 200_000:
        # The retained inequalities give the same value at this support.
        # Their dual cut remains a lower bound for the full-net objective.
        net, medians = _support_extremes(net, medians, support)

    with gp.Model(env=env) as model:
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.Method = 1  # Dual simplex.
        model.Params.FeasibilityTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        mu = model.addMVar(len(support), lb=-gp.GRB.INFINITY)
        r = model.addVar(lb=0)
        upper = model.addConstr(mu <= M * support)
        lower = model.addConstr(-mu <= M * support)
        model.addConstr(net @ mu - r <= medians)
        model.addConstr(-net @ mu - r <= -medians)
        model.setObjective(r, gp.GRB.MINIMIZE)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"Fixed-support LP failed: Gurobi status {model.Status}")
        candidate = np.clip(mu.X, -M * support, M * support)
        # Pi is the RHS derivative, i.e. the negative Lagrange multiplier.
        gradient = M * (upper.Pi + lower.Pi)
        return candidate, float(model.ObjVal), gradient


def brute_force_estimation(
    data, s, epsilon, lambda_upper, delta=0.05, *, tol=None, seed=None, C=2,
    net_radius=0.25, max_net_points=200_000, max_iter=1000,
):
    """Return (mu_hat, info) using a full direction net and outer cutting planes.

    lambda_upper is a supplied covariance eigenvalue bound (initial choice:
    2*lambda_max(Sigma)). K is the smallest odd integer >=
    C*max(s*log(d/s), epsilon*n, log(1/delta)), with C=2 by default. Samples enter balanced,
    randomly permuted blocks; using the same seed reproduces the partition.
    tol defaults to sqrt(K*lambda_upper/n)/4.

    Both the fixed-support LPs and outer support MILP use Gurobi.

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
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()

        def solve_support(support):
            candidate, L, gradient = solve_fixed_support_lp(net, medians, support, M, env=env)
            value = float(np.max(np.abs(medians - net @ candidate)))
            return candidate, L, value, gradient

        best_mu, info = cutting_plane(
            s, current_support, best_mu, best_value, solve_support, tol, max_iter=max_iter,
        )
    info.update({
        "C": C, "K": K, "net_radius": net_radius, "net_size": len(net),
        "runtime": perf_counter() - start,
    })
    return best_mu, info
