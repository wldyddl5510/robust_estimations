"""Projected Algorithm 1: dense fits on low-dimensional projections, then sparse LS aggregation.

partition_estimation cuts the coordinates into d/r random blocks; haar_estimation uses J
independent Haar-random rank-r projections. Only the aggregation imposes ||mu||_0 <= s,
minimizing sum_j ||P_j mu - estimate_j||^2 with utils.cutting_plane from zero, to a squared-loss
gap of max(tol, tol * objective). The reported gap concerns the aggregation only.
"""

from time import perf_counter

import gurobipy as gp
import numpy as np

from IP_algorithm import dense_estimation, deterministic_upper_bound
from utils import check_data, cutting_plane, gurobi_env, mom_blocks, new_model, top_s

STAT_KEYS = ("oracle_calls", "inner_iterations", "constraints", "heuristic_calls", "heuristic_cuts",
             "deferred_supports", "deterministic_certificates")


def _dense_fits(projected, s, epsilon, delta, C, seed, tol, env):
    """Dense Algorithm 1 on each projected dataset; returns ((J, r) estimates, summed counters, K)."""
    estimates, stats = [], dict.fromkeys(STAT_KEYS, 0)
    for X in projected:
        blocks = mom_blocks(X, s, epsilon, delta=delta, C=C, seed=seed)
        estimate, info = dense_estimation(blocks, s, tol, env)
        estimates.append(estimate)
        for key in STAT_KEYS:
            stats[key] += info[key]
    return np.array(estimates), stats, len(blocks)


def random_coordinate_projections(d, r, rng):
    """(d/r, r) sorted coordinate blocks from one random permutation; every coordinate is in one block."""
    if not isinstance(r, (int, np.integer)) or not 1 <= r <= d or d % r:
        raise ValueError(f"r must be an integer in [1, d] that divides d={d}")
    return np.sort(rng.permutation(d).reshape(d // r, r), axis=1)


def aggregate_coordinates(projections, estimates, d, s, *, tol=1e-5, max_iter=1000):
    """min sum_j ||mu[P_j] - estimate_j||^2 over ||mu||_0 <= s for coordinate-subset projections P_j.

    The loss separates over coordinates, so a support z has the exact closed form
    constant - gains @ z, which is also a valid support cut. Unobserved coordinates stay zero.
    """
    counts = np.bincount(projections.ravel(), minlength=d)
    sums = np.bincount(projections.ravel(), weights=estimates.ravel(), minlength=d)
    means = np.divide(sums, counts, out=np.zeros(d), where=counts > 0)
    gains = sums * means
    constant = float(np.sum(estimates**2))

    def solve_support(z):
        mu = np.where(z > 0.5, means, 0.0)
        return mu, float(constant - gains @ z), float(np.sum((mu[projections] - estimates)**2)), -gains

    return cutting_plane(s, np.zeros(d), np.zeros(d), constant, solve_support, tol, max_iter, rtol=tol)


def partition_estimation(data, s, epsilon, *, r, delta=0.05, C=2, seed=None, tol=1e-5, max_iter=1000):
    """Random partition into J = d/r coordinate blocks; returns (mu, info).

    Each dense fit replaces (d, s) by (r, min(r, s)) in the block rule, uses directions in
    V_min(2s,r), and shares the seed's block partition; the coordinate permutation uses an
    independent child stream. The aggregation is unboxed and exact on each support.
    """
    start = perf_counter()
    data = check_data(data, s)
    d = data.shape[1]
    seed_sequence = np.random.SeedSequence(seed)
    projections = random_coordinate_projections(d, r, np.random.default_rng(seed_sequence.spawn(1)[0]))
    with gurobi_env() as env:
        estimates, stats, K = _dense_fits((data[:, block] for block in projections), min(s, r),
                                          epsilon, delta, C, seed_sequence, tol, env)
    mu, info = aggregate_coordinates(projections, estimates, d, s, tol=tol, max_iter=max_iter)
    return mu, {**stats, **info, "C": C, "K": K, "J": len(projections), "r": r, "runtime": perf_counter() - start}


def random_haar_projections(d, r, J, rng):
    """(J, r, d) independent Haar projections with orthonormal rows (Gaussian thin QR, sign-fixed R).

    The row space is uniform on the Grassmannian; there is no sqrt(d/r) rescaling.
    """
    for name, value in (("r", r), ("J", J)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if r > d:
        raise ValueError("r must not exceed d")
    projections = np.empty((J, r, d))
    for j in range(J):
        Q, R = np.linalg.qr(rng.normal(size=(d, r)), mode="reduced")
        projections[j] = (Q * np.where(np.diag(R) < 0, -1.0, 1.0)).T
    return projections


def _boxed_least_squares(P, y, z, M, env):
    """argmin sum_j ||P_j x - y_j||^2 over |x| <= M*z, a convex QP with no extra regularization."""
    if not np.any(z):
        return np.zeros(len(M))
    scale = max(1.0, float(np.max(np.linalg.norm(y, axis=1))))
    bounds = M * z / scale
    with new_model(env, NonConvex=0, FeasibilityTol=1e-9, OptimalityTol=1e-9, Method=1) as model:
        x = model.addMVar(len(M), lb=-bounds, ub=bounds)
        residual = model.addMVar(y.size, lb=-gp.GRB.INFINITY)
        model.addConstr(residual == P.reshape(-1, len(M)) @ x - (y / scale).ravel())
        model.setObjective(residual @ residual, gp.GRB.MINIMIZE)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            model.Params.Presolve = 0
            model.Params.DualReductions = 0
            model.Params.NumericFocus = 3
            model.reset()
            model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"Haar aggregation QP failed: Gurobi status {model.Status}")
        return np.clip(x.X * scale, -M * z, M * z)


def haar_support_ls(P, y, z, M, *, tol=1e-6, env=None):
    """(mu, L, U, g) for sum_j ||P_j mu - y_j||^2 over |mu| <= M*z on the binary support z.

    mu solves the box QP and U is its loss. The tangent of the loss at mu, minimized over each
    support's box and rounded down, gives the cut L + g @ (z' - z); U - L must be within
    max(tol, tol * U).
    """
    mu = _boxed_least_squares(P, y, z, M, env)
    residual = P @ mu - y
    slope = 2 * np.einsum("jrd,jr->d", P, residual)
    gradient = -M * np.abs(slope)
    value, linear, bound_term = float(np.sum(residual**2)), float(slope @ mu), float(gradient @ z)
    roundoff = 64 * np.finfo(float).eps * max(1, abs(value), abs(linear), abs(bound_term))
    lower = value - linear + bound_term - roundoff
    if not np.isfinite([lower, value]).all() or value - lower > max(tol, tol * abs(value)):
        raise RuntimeError("Haar aggregation QP did not certify the requested gap")
    return mu, lower, value, gradient


def aggregate_haar(P, y, s, M, *, tol=1e-5, max_iter=1000, env=None):
    """min sum_j ||P_j mu - y_j||^2 over ||mu||_0 <= s and |mu| <= M, from zero."""
    d = P.shape[2]
    return cutting_plane(s, np.zeros(d), np.zeros(d), float(np.sum(y**2)),
                         lambda z: haar_support_ls(P, y, z, M, tol=tol / 4, env=env), tol, max_iter, rtol=tol)


def haar_estimation(data, s, epsilon, *, r, J, delta=0.05, C=2, seed=None, tol=1e-5, max_iter=1000):
    """J independent Haar projections of rank r; returns (mu, info).

    Each dense fit uses the full sphere V_r and the block rule for (r, r, delta/J); all projections
    share one block partition from a child stream independent of the projections. The aggregation
    domain is the box |mu_i| <= M_i = |coordinate MoM_i| + U_det (+ margin), where U_det is
    Algorithm 1's deterministic bound at the thresholded MoM of the original data: a data-driven
    estimator domain, not a certificate that the unboxed optimum lies inside. J need not relate to
    d/r, and rank-deficient stacked projections are allowed.
    """
    start = perf_counter()
    data = check_data(data, s)
    d = data.shape[1]
    projection_seed, block_seed = np.random.SeedSequence(seed).spawn(2)
    projections = random_haar_projections(d, r, J, np.random.default_rng(projection_seed))
    blocks = mom_blocks(data, s, epsilon, delta=delta, C=C, seed=seed)
    center = np.median(blocks, axis=0)
    M = np.abs(center) + deterministic_upper_bound(blocks, center * top_s(center, s), s)
    M += 1e-8 * max(1.0, float(M.max()))
    with gurobi_env() as env:
        estimates, stats, K = _dense_fits((data @ P.T for P in projections), r, epsilon, delta / J, C,
                                          block_seed, tol, env)
        mu, info = aggregate_haar(projections, estimates, s, M, tol=tol, max_iter=max_iter, env=env)
    return mu, {**stats, **info, "C": C, "K": K, "J": J, "r": r, "box_K": len(blocks),
                "runtime": perf_counter() - start}
