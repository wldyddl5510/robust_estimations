"""Section 6: sparse MoM/trimmed estimation over a fixed random direction bank.

Each bank contains tilde_J random sparse unit directions and all d coordinate
directions. The latter preserve the coordinatewise box-bound argument. Gaps
certify this finite-bank objective, within solver precision, not the continuous
supremum. No continuous-direction separation oracle is called.
"""

from time import perf_counter

import gurobipy as gp
import numpy as np

from brute_force import solve_fixed_support_lp
from trimmed_IP_algorithm import trimmed_initialization, trimmed_mean
from utils import cutting_plane, mom_initialization


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def sample_sparse_directions(d, s, tilde_J, *, seed=None):
    """Return d coordinate directions followed by tilde_J independent draws.

    For each draw, choose a uniform support of size q=min(2*s,d), then a
    uniform unit-sphere vector on that support. Draws are not deduplicated.
    Absolute residuals account for both signs without doubling the bank.
    seed is a nonnegative integer or None. A separate SeedSequence stream
    keeps direction draws independent of the existing MoM block stream.
    With a fixed seed, increasing tilde_J extends the same random bank.
    """
    _positive_integer(d, "d")
    _positive_integer(s, "s")
    _positive_integer(tilde_J, "tilde_J")
    if s > d:
        raise ValueError("s must satisfy 1 <= s <= d")
    q = min(2 * s, d)
    vector_stream, support_stream = np.random.SeedSequence(seed, spawn_key=(1,)).spawn(2)
    vector_rng = np.random.default_rng(vector_stream)
    support_rng = np.random.default_rng(support_stream)
    directions = np.zeros((d + tilde_J, d))
    directions[np.arange(d), np.arange(d)] = 1.
    # Bound temporary Gaussian/support arrays independently of the bank size.
    draw_batch = max(1, 1_000_000 // q)
    for first in range(0, tilde_J, draw_batch):
        count = min(draw_batch, tilde_J - first)
        vectors = vector_rng.standard_normal((count, q))
        norms = np.linalg.norm(vectors, axis=1)
        while np.any(norms == 0):
            zero = norms == 0
            vectors[zero] = vector_rng.standard_normal((int(zero.sum()), q))
            norms[zero] = np.linalg.norm(vectors[zero], axis=1)
        vectors /= norms[:, None]
        if q == d:
            directions[d + first:d + first + count] = vectors
        else:
            supports = np.array([support_rng.choice(d, size=q, replace=False) for _ in range(count)])
            directions[d + first + np.arange(count)[:, None], supports] = vectors
    return directions


def _validate_options(tilde_J, max_iter, batch_size):
    _positive_integer(tilde_J, "tilde_J")
    _positive_integer(max_iter, "max_iter")
    if batch_size is not None:
        _positive_integer(batch_size, "batch_size")


def _direction_targets(observations, directions, center, statistic, batch_size):
    """Precompute scalar robust centers without allocating the full projection matrix."""
    d = observations.shape[1]
    targets = np.empty(len(directions))
    targets[:d] = center
    if batch_size is None:
        batch_size = max(1, 1_000_000 // len(observations))
    for first in range(d, len(directions), batch_size):
        last = min(first + batch_size, len(directions))
        targets[first:last] = statistic(observations @ directions[first:last].T)
    return targets


def _solve_sampled_problem(directions, targets, center, support, initial_mu, s, tol, max_iter):
    def objective(mu):
        return float(np.max(np.abs(targets - directions @ mu)))

    initial_value = objective(initial_mu)
    # e_l is in the bank, so every improving mu satisfies
    # |mu_l| <= |center_l| + F_bank(initial_mu), even for a very small tilde_J.
    margin = 1e-8 * max(1., np.max(np.abs(center)), initial_value)
    M = np.abs(center) + initial_value + margin
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", 1)
        env.start()

        def solve_support(z):
            candidate, lower, gradient = solve_fixed_support_lp(directions, targets, z, M, env=env)
            return candidate, lower, objective(candidate), gradient

        return cutting_plane(
            s, support, initial_mu, initial_value, solve_support, tol, max_iter=max_iter,
        )


def _bank_info(d, s, tilde_J, seed, direction_seed):
    return {
        "tilde_J": int(tilde_J), "coordinate_directions": d,
        "num_directions": d + int(tilde_J), "direction_sparsity": min(2 * s, d),
        "objective_scope": "sampled_directions_plus_coordinates",
        "seed": None if seed is None else int(seed),
        "direction_seed": None if direction_seed is None else int(direction_seed),
    }


def random_mom_estimation(
    data, s, epsilon, lambda_upper, delta=0.05, *, tilde_J, tol=1e-5,
    seed=None, direction_seed=None, C=2, max_iter=1000, batch_size=None,
):
    """Return (mu_hat, info) for the sampled MoM objective with sparse support.

    tilde_J is the positive number of random draws; d coordinate directions
    are additional. The bank and the MoM partition stay fixed throughout a fit.
    Existing MoM block-count/initialization rules and seed compatibility are
    preserved. direction_seed defaults to seed; changing it leaves blocks alone.
    batch_size bounds the number of random projections processed at once;
    None targets at most one million projection entries per batch (one column
    minimum). The direction bank itself occupies O((tilde_J+d)*d) memory.
    tol defaults to 1e-5 (also when None), as in Algorithm 1.
    max_iter exhaustion returns info['converged']=False.
    """
    start = perf_counter()
    _validate_options(tilde_J, max_iter, batch_size)
    tol = 1e-5 if tol is None else tol
    blocks, center, support, initial_mu, tol = mom_initialization(
        data, s, epsilon, lambda_upper, delta, tol, seed, C=C,
    )
    K, d = blocks.shape
    direction_seed = seed if direction_seed is None else direction_seed
    directions = sample_sparse_directions(d, s, tilde_J, seed=direction_seed)
    targets = _direction_targets(
        blocks, directions, center, lambda projections: np.median(projections, axis=0), batch_size,
    )
    estimate, info = _solve_sampled_problem(
        directions, targets, center, support, initial_mu, s, tol, max_iter,
    )
    info.update(_bank_info(d, s, tilde_J, seed, direction_seed))
    info.update({"C": C, "K": K, "runtime": perf_counter() - start})
    return estimate, info


def random_trimmed_estimation(
    data, s, epsilon, delta=0.05, *, tilde_J, tol=1e-5, seed=None,
    direction_seed=None, max_iter=1000, batch_size=None,
):
    """Return (mu_hat, info) for the sampled trimmed-mean objective.

    Uses raw observations and the existing k=floor(epsilon*n) +
    max(ceil(log(2/delta)),ceil(epsilon*n/2)) rule, with k removals per tail.
    An invalid k is rejected. tilde_J, coordinate augmentation, seed behavior,
    batching and finite-objective certificates follow random_mom_estimation;
    there is no MoM partition. No theoretical minimum tilde_J is imposed.
    """
    start = perf_counter()
    _validate_options(tilde_J, max_iter, batch_size)
    center, support, initial_mu, k, tol = trimmed_initialization(data, s, epsilon, delta, tol)
    observations = np.asarray(data, dtype=float)
    n, d = observations.shape
    direction_seed = seed if direction_seed is None else direction_seed
    directions = sample_sparse_directions(d, s, tilde_J, seed=direction_seed)
    targets = _direction_targets(
        observations, directions, center, lambda projections: trimmed_mean(projections, k), batch_size,
    )
    estimate, info = _solve_sampled_problem(
        directions, targets, center, support, initial_mu, s, tol, max_iter,
    )
    info.update(_bank_info(d, s, tilde_J, seed, direction_seed))
    info.update({"n": n, "k": k, "effective_n": n - 2 * k, "runtime": perf_counter() - start})
    return estimate, info
