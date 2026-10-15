"""Sparse robust mean estimation over a finite direction set: a covering net or a random bank.

Each method minimizes F(mu) = max_j |target_j - <v_j, mu>| over ||mu||_0 <= s, where target_j is
a robust center (MoM median or trimmed mean) of the projections onto v_j, by solving a
fixed-support LP inside utils.cutting_plane. Every set contains the +-coordinate directions, so
improving estimates satisfy |mu_i| <= |center_i| + F(initial). Gaps certify the finite-set
objective, within solver precision, not the continuous supremum.
"""

from functools import lru_cache
from itertools import combinations
from math import comb, isqrt
from time import perf_counter

import gurobipy as gp
import numpy as np

from trimmed_IP_algorithm import trimmed_initialization, trimmed_mean
from utils import check_data, cutting_plane, gurobi_env, mom_blocks, new_model, top_s

MAX_NET_POINTS = 200_000


class NetSizeError(ValueError):
    """The full covering net exceeds the allocation limit."""

    def __init__(self, size, max_points):
        self.size = size
        super().__init__(f"Net has {size:,} points; exceeds max_points={max_points:,}.")


@lru_cache(maxsize=None)
def _lattice_count(dimension, budget):
    """Number of nonzero integer vectors of the given dimension with squared norm <= budget."""
    if dimension == 0:
        return 1
    if dimension == 1:
        return 2 * isqrt(budget)
    if budget < dimension:
        return 0
    return sum(2 * _lattice_count(dimension - 1, budget - value * value)
               for value in range(1, isqrt(budget - dimension + 1) + 1))


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


def discretized_net(d, s, radius=0.25, max_points=MAX_NET_POINTS):
    """Deterministic radius-net of {v: ||v||_0 <= 2s, ||v||_2 <= 1}, with +-e_i included.

    With k = min(2s, d) and grid spacing 2*radius/sqrt(k), rounding any v to the grid moves it
    by at most radius without adding nonzeros; grid points of norm <= 1 + radius are projected
    onto the unit ball, which cannot increase the distance to v. Raises NetSizeError before
    allocating more than max_points rows.
    """
    k = min(2 * s, d)
    step = 2 * radius / np.sqrt(k)
    budget = int(np.floor(np.nextafter(k * (1 + radius)**2 / (4 * radius**2), np.inf)))
    size = 1 + 2 * d + sum(comb(d, m) * _lattice_count(m, budget) for m in range(1, k + 1))
    if max_points is not None and size > max_points:
        raise NetSizeError(size, max_points)
    net = np.zeros((size, d))
    net[1:d + 1] = np.eye(d)
    net[d + 1:2 * d + 1] = -np.eye(d)
    offset = 1 + 2 * d
    for m in range(1, k + 1):
        local = step * np.array(list(_lattice_points(m, budget)), dtype=float)
        local /= np.maximum(1, np.linalg.norm(local, axis=1, keepdims=True))
        for support in combinations(range(d), m):
            net[offset:offset + len(local), list(support)] = local
            offset += len(local)
    return net


def sample_sparse_directions(d, s, tilde_J, *, seed=None):
    """d coordinate directions followed by tilde_J random sparse unit vectors.

    Each draw takes a uniform support of size min(2s, d) and a uniform unit vector on it; draws
    are not deduplicated. Their streams are independent of the MoM block partition, and a larger
    tilde_J extends the same bank.
    """
    if isinstance(tilde_J, (bool, np.bool_)) or not isinstance(tilde_J, (int, np.integer)) or tilde_J < 1:
        raise ValueError("tilde_J must be a positive integer")
    q = min(2 * s, d)
    vector_stream, support_stream = np.random.SeedSequence(seed, spawn_key=(1,)).spawn(2)
    vector_rng, support_rng = np.random.default_rng(vector_stream), np.random.default_rng(support_stream)
    directions = np.zeros((d + tilde_J, d))
    directions[np.arange(d), np.arange(d)] = 1.
    batch = max(1, 1_000_000 // q)  # bounds temporary memory independently of the bank size
    for first in range(0, tilde_J, batch):
        count = min(batch, tilde_J - first)
        vectors = vector_rng.standard_normal((count, q))
        vectors /= np.linalg.norm(vectors, axis=1)[:, None]
        rows = d + first + np.arange(count)
        if q == d:
            directions[rows] = vectors
        else:
            supports = np.array([support_rng.choice(d, size=q, replace=False) for _ in range(count)])
            directions[rows[:, None], supports] = vectors
    return directions


def _support_extremes(directions, targets, support):
    """Keep the extreme targets for each distinct active-coordinate direction."""
    active = np.flatnonzero(support)
    groups = (np.unique(directions[:, active], axis=0, return_inverse=True)[1]
              if len(active) else np.zeros(len(directions), dtype=int))
    order = np.lexsort((targets, groups))
    starts = np.r_[0, np.flatnonzero(np.diff(groups[order])) + 1]
    ends = np.r_[starts[1:] - 1, len(directions) - 1]
    indices = np.r_[order[starts], order[ends]]
    return directions[indices], targets[indices]


def solve_fixed_support_lp(directions, targets, support, M, *, env=None):
    """min_mu max_j |target_j - <v_j, mu>| over |mu| <= M*support; returns (feasible_mu, L, cut_gradient)."""
    if len(directions) > MAX_NET_POINTS:
        # The kept inequalities give the same value on this support, and their dual cut
        # remains a lower bound for the full set.
        directions, targets = _support_extremes(directions, targets, support)
    with new_model(env, Method=1, FeasibilityTol=1e-9, OptimalityTol=1e-9) as model:
        mu = model.addMVar(len(support), lb=-gp.GRB.INFINITY)
        r = model.addVar(lb=0)
        upper = model.addConstr(mu <= M * support)
        lower = model.addConstr(-mu <= M * support)
        model.addConstr(directions @ mu - r <= targets)
        model.addConstr(-directions @ mu - r <= -targets)
        model.setObjective(r, gp.GRB.MINIMIZE)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"Fixed-support LP failed: Gurobi status {model.Status}")
        # Pi is the RHS derivative, i.e. the negative Lagrange multiplier.
        return np.clip(mu.X, -M * support, M * support), float(model.ObjVal), M * (upper.Pi + lower.Pi)


def _projection_targets(observations, directions, statistic):
    """statistic(observations @ directions.T), computed in batches of about a million entries."""
    targets = np.empty(len(directions))
    step = max(1, 1_000_000 // len(observations))
    for first in range(0, len(directions), step):
        targets[first:first + step] = statistic(observations @ directions[first:first + step].T)
    return targets


def _fit(directions, targets, center, support, s, tol, max_iter):
    """Minimize max_j |targets_j - directions_j @ mu| over ||mu||_0 <= s from the thresholded center."""
    def objective(mu):
        return float(np.max(np.abs(targets - directions @ mu)))

    initial_mu = center * support
    initial = objective(initial_mu)
    M = np.abs(center) + initial + 1e-8 * max(1., np.max(np.abs(center)), initial)
    with gurobi_env() as env:
        def solve_support(z):
            mu, lower, gradient = solve_fixed_support_lp(directions, targets, z, M, env=env)
            return mu, lower, objective(mu), gradient

        return cutting_plane(s, support, initial_mu, initial, solve_support, tol, max_iter)


def brute_force_estimation(data, s, epsilon, *, delta=0.05, C=3, seed=None, tol=1e-5, max_iter=1000):
    """MoM objective over the full radius-1/4 covering net; returns (mu, info).

    Raises NetSizeError (a ValueError) when the net exceeds MAX_NET_POINTS rows.
    Runtime includes building the net and its projections.
    """
    start = perf_counter()
    blocks = mom_blocks(data, s, epsilon, delta=delta, C=C, seed=seed)
    center = np.median(blocks, axis=0)
    net = discretized_net(blocks.shape[1], s)
    targets = _projection_targets(blocks, net, lambda projections: np.median(projections, axis=0))
    mu, info = _fit(net, targets, center, top_s(center, s), s, tol, max_iter)
    return mu, {**info, "C": C, "K": len(blocks), "net_size": len(net), "runtime": perf_counter() - start}


def random_mom_estimation(data, s, epsilon, *, tilde_J, delta=0.05, C=3, seed=None, tol=1e-5, max_iter=1000):
    """MoM objective over d coordinate plus tilde_J random sparse directions; returns (mu, info).

    The bank and the block partition both come from seed (independent streams) and stay fixed.
    """
    start = perf_counter()
    blocks = mom_blocks(data, s, epsilon, delta=delta, C=C, seed=seed)
    K, d = blocks.shape
    center = np.median(blocks, axis=0)
    directions = sample_sparse_directions(d, s, tilde_J, seed=seed)
    targets = np.r_[center, _projection_targets(blocks, directions[d:], lambda p: np.median(p, axis=0))]
    mu, info = _fit(directions, targets, center, top_s(center, s), s, tol, max_iter)
    return mu, {**info, "C": C, "K": K, "tilde_J": tilde_J, "runtime": perf_counter() - start}


def random_trimmed_estimation(data, s, epsilon, *, tilde_J, delta=0.05, seed=None, tol=1e-5, max_iter=1000):
    """Trimmed-mean objective (Algorithm 2's k per tail, raw observations) over the same kind of bank."""
    start = perf_counter()
    data = check_data(data, s)
    n, d = data.shape
    center, support, k = trimmed_initialization(data, s, epsilon, delta)
    directions = sample_sparse_directions(d, s, tilde_J, seed=seed)
    targets = np.r_[center, _projection_targets(data, directions[d:], lambda p: trimmed_mean(p, k))]
    mu, info = _fit(directions, targets, center, support, s, tol, max_iter)
    return mu, {**info, "n": n, "k": k, "effective_n": n - 2 * k, "tilde_J": tilde_J,
                "runtime": perf_counter() - start}
