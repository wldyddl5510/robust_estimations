import gurobipy as gp
import numpy as np
from scipy.stats import multivariate_t


def t_dist_data_generation(n, loc, nu, d, scale=1):
    """Return (n, d) multivariate t samples with location loc and scale.

    loc is a scalar or a length-d vector; nu must be positive.
    scale is a scalar (times I_d), a length-d diagonal, or a (d, d) matrix.
    For nu > 2, covariance is nu / (nu - 2) times the scale matrix.
    """
    loc = np.asarray(loc, dtype=float)
    if loc.ndim != 0 and loc.shape != (d,):
        raise ValueError("loc must be a scalar or a length-d vector")
    loc = np.broadcast_to(loc, (d,))
    return multivariate_t.rvs(loc=loc, shape=scale, df=nu, size=n).reshape(n, d)


def sparse_t_dist_data_generation(n, loc, nu, d, s, scale=1):
    """Return (data, mu) with loc on s uniformly chosen mean coordinates.

    data has shape (n, d), mu has shape (d,), and nu > 1 ensures a finite mean.
    scale follows t_dist_data_generation; samples remain dense in general.
    """
    if not 0 <= s <= d:
        raise ValueError("s must satisfy 0 <= s <= d")
    if np.ndim(loc) != 0 or not np.isfinite(loc) or (s > 0 and loc == 0):
        raise ValueError("loc must be a finite scalar, nonzero when s > 0")
    if not nu > 1:
        raise ValueError("nu must be greater than 1 for a finite mean")

    mu = np.zeros(d)
    mu[np.random.choice(d, size=s, replace=False)] = loc
    return t_dist_data_generation(n, mu, nu, d, scale=scale), mu


def huber_contamination(data, epsilon, outlier_loc=10, outlier_std=1):
    """Return a copy of (n, d) data with independent Gaussian replacements.

    Each row is replaced with probability epsilon, giving Binomial(n, epsilon)
    replacements, not a fixed contamination budget. Set outlier parameters
    independently of data for the Huber model. outlier_loc and outlier_std
    are scalars or length-d vectors; outlier_std is a standard deviation.
    """
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must satisfy 0 <= epsilon <= 1")
    contaminated = np.array(data, dtype=float, copy=True)
    if contaminated.ndim != 2:
        raise ValueError("data must have shape (n, d)")

    n, d = contaminated.shape
    mask = np.random.random(n) < epsilon
    contaminated[mask] = np.random.normal(
        loc=outlier_loc, scale=outlier_std, size=(mask.sum(), d)
    )
    return contaminated


def adversarial_sparse_contamination(data, epsilon, s, strength=3):
    """Return a copy with at most floor(epsilon * n) adaptively replaced rows.

    Coordinate medians estimate the support. The attack weakens its top-s
    coordinates and reinforces the next s (or all remaining coordinates).
    The lowest projections onto this direction are moved to the projection
    median + strength * MAD, preserving every orthogonal component.
    Zero MAD falls back to the projection std, then 1 for constant projections.
    This is a heuristic, not a worst-case attack against a particular estimator.
    """
    if not 0 <= epsilon < 0.5:
        raise ValueError("epsilon must satisfy 0 <= epsilon < 0.5")
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError("strength must be positive and finite")
    contaminated = np.array(data, dtype=float, copy=True)
    if contaminated.ndim != 2:
        raise ValueError("data must have shape (n, d)")
    n, d = contaminated.shape
    if not 1 <= s <= d:
        raise ValueError("s must satisfy 1 <= s <= d")
    k = int(epsilon * n)
    if k == 0:
        return contaminated

    center = np.median(contaminated, axis=0)
    order = np.argsort(-np.abs(center))
    signs = np.where(center >= 0, 1, -1)
    direction = np.zeros(d)
    direction[order[:s]] = -signs[order[:s]]
    direction[order[s:2 * s]] = signs[order[s:2 * s]]
    direction /= np.linalg.norm(direction)

    projections = (contaminated - center) @ direction
    midpoint = np.median(projections)
    spread = np.median(np.abs(projections - midpoint))
    if spread == 0:
        spread = np.std(projections) or 1.0
    target = midpoint + strength * spread
    indices = np.argpartition(projections, k - 1)[:k]
    contaminated[indices] += (target - projections[indices])[:, None] * direction
    return contaminated


def mom_initialization(data, s, epsilon, lambda_upper, delta=0.05, tol=None, seed=None, *, C=1):
    """Return block means, coordinate medians, initial support/mean, and tolerance.

    C is the block-count multiplier (default 1). Blocks are balanced and seeded.
    Default tolerance is sqrt(K*lambda_upper/n)/100.
    lambda_upper bounds the covariance eigenvalue, not the t scale matrix.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or min(data.shape) == 0 or not np.isfinite(data).all():
        raise ValueError("data must be a finite, nonempty (n, d) array")
    n, d = data.shape
    if not isinstance(s, (int, np.integer)) or not 1 <= s <= d:
        raise ValueError("s must be an integer with 1 <= s <= d")
    if not 0 <= epsilon < 0.5 or not 0 < delta < 1:
        raise ValueError("require 0 <= epsilon < 0.5 and 0 < delta < 1")
    if not np.isfinite(lambda_upper) or lambda_upper <= 0:
        raise ValueError("lambda_upper must be positive and finite")
    if not np.isfinite(C) or C <= 0:
        raise ValueError("C must be positive and finite")
    K = int(np.ceil(C * max(s * np.log(d / s), epsilon * n, np.log(1 / delta))))
    K += K % 2 == 0
    if K > n:
        raise ValueError(f"Required K={K} exceeds n={n}; change the experiment settings.")
    statistical_tol = np.sqrt(K * lambda_upper / n)
    tol = statistical_tol / 100 if tol is None else tol
    if not 0 < tol <= statistical_tol:
        raise ValueError("tol must satisfy 0 < tol <= sqrt(K*lambda_upper/n)")

    blocks = np.array_split(np.random.default_rng(seed).permutation(n), K)
    block_means = np.array([data[block].mean(axis=0) for block in blocks])
    center = np.median(block_means, axis=0)
    support = np.zeros(d)
    support[np.argsort(-np.abs(center))[:s]] = 1
    return block_means, center, support, center * support, float(tol)


def cutting_plane(
    s, initial_support, initial_mu, initial_upper, solve_support, tol, max_iter=1000,
):
    """Return (best_mu, info) using the draft's outer support cutting planes.

    The objective must be nonnegative. solve_support(z) returns (mu, L, U, g):
    mu is feasible on support z, U bounds its objective from above, and
    L + g @ (z_new - z) is a valid lower bound for every feasible z_new.
    For the draft's box constraints, g = -M * (lambda + rho).
    Keeping L and U separate supports both exact LPs and approximate oracles.

    initial_mu must be feasible on the binary initial_support, with objective
    at most initial_upper. The caller measures end-to-end runtime, including
    its own preprocessing. info['objective'] is the best certified upper bound.
    """
    current_support = np.array(initial_support, dtype=float, copy=True)
    best_mu = np.array(initial_mu, dtype=float, copy=True)
    d = best_mu.size
    if best_mu.ndim != 1 or current_support.shape != best_mu.shape:
        raise ValueError("initial_mu and initial_support must be matching vectors")
    if not isinstance(s, (int, np.integer)) or not 1 <= s <= d:
        raise ValueError("s must be an integer with 1 <= s <= d")
    if not np.isin(current_support, [0, 1]).all() or current_support.sum() > s:
        raise ValueError("initial_support must be binary with at most s entries")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    best_upper = float(initial_upper)
    lower_bound = 0.0
    converged = False

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", 1)
        env.start()
        with gp.Model(env=env) as master:
            master.Params.MIPGap = 0
            master.Params.MIPGapAbs = 0
            master.Params.FeasibilityTol = 1e-9
            master.Params.IntFeasTol = 1e-9
            master.Params.OptimalityTol = 1e-9
            z = master.addMVar(d, vtype=gp.GRB.BINARY)
            eta = master.addVar(lb=0)
            master.addConstr(z.sum() <= s)
            master.setObjective(eta)

            for iteration in range(1, max_iter + 1):
                candidate, L, U, gradient = solve_support(current_support.copy())
                if U < best_upper:
                    best_mu = np.array(candidate, dtype=float, copy=True)
                    best_upper = float(U)
                master.addConstr(eta >= L + np.asarray(gradient) @ (z - current_support))
                master.optimize()
                if master.Status != gp.GRB.OPTIMAL:
                    raise RuntimeError(f"Support MILP failed: Gurobi status {master.Status}")
                lower_bound = max(lower_bound, master.ObjBound)
                if best_upper - lower_bound <= tol:
                    converged = True
                    break
                current_support = np.rint(z.X)

    info = {
        "converged": converged, "objective": best_upper, "lower_bound": lower_bound,
        "gap": max(0.0, best_upper - lower_bound), "tol": tol, "iterations": iteration,
    }
    return best_mu, info
