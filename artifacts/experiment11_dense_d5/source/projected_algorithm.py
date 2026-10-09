"""Random coordinate partition, dense Algorithm 1, then sparse aggregation."""

from time import perf_counter

import gurobipy as gp
import numpy as np

from IP_algorithm import dense_estimation
from utils import cutting_plane, mom_initialization


def random_coordinate_projections(d, r, rng):
    """Return a (d/r, r) array from one random permutation cut into equal blocks.

    Row j lists P_j's sorted coordinates; every coordinate lies in exactly one row.
    """
    if not isinstance(r, (int, np.integer)) or not 1 <= r <= d:
        raise ValueError("r must be an integer with 1 <= r <= d")
    if d % r:
        raise ValueError(f"r must divide d={d} to give d/r equal blocks")
    return np.sort(rng.permutation(d).reshape(d // r, r), axis=1)


def least_squares_statistics(projections, estimates, d, *, weights=None):
    """Return coordinate counts, sums, and constant for (weighted) squared loss."""
    weights = np.ones(len(estimates)) if weights is None else np.asarray(weights)
    repeated = np.repeat(weights, projections.shape[1])
    counts = np.bincount(projections.ravel(), weights=repeated, minlength=d)
    sums = np.bincount(projections.ravel(), weights=(repeated * estimates.ravel()), minlength=d)
    constant = float(weights @ np.sum(estimates**2, axis=1))
    return counts, sums, constant


def fixed_support_least_squares(counts, sums, constant, support):
    """Return (mu, L, U, g); the affine support value is constant - q @ z."""
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    gains = sums * means
    mu = np.where(np.asarray(support) > 0.5, means, 0.0)
    value = float(constant - gains @ support)
    return mu, value, max(value, 0.0), -gains


def fixed_support_max(projections, estimates, support, *, tol=1e-6, rtol=0.0, env=None):
    """Return (mu, L, U, g) for squared minimax on a binary support.

    The QCP dual gives simplex weights w. Weighted least squares then gives
    a globally valid binary-support cut A(w) - q(w) @ z, with g = -q(w).
    L evaluates this cut; U evaluates the candidate's actual squared max loss.
    """
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")
    if not np.isfinite(rtol) or rtol < 0:
        raise ValueError("rtol must be nonnegative and finite")
    support = np.asarray(support, dtype=float)
    d = len(support)
    covered = np.bincount(projections.ravel(), minlength=d) > 0
    active = (support > 0.5) & covered
    norms = np.sum(estimates**2, axis=1)
    if not active.any():
        weights = np.zeros(len(estimates))
        weights[np.argmax(norms)] = 1
        counts, sums, constant = least_squares_statistics(projections, estimates, d, weights=weights)
        mu, lower, upper, gradient = fixed_support_least_squares(counts, sums, constant, support)
        return mu, lower, upper, gradient

    with gp.Model(env=env) as model:
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.QCPDual = 1
        model.Params.NonConvex = 0
        model.Params.FeasibilityTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        model.Params.BarQCPConvTol = 1e-8
        mu = model.addMVar(d, lb=np.where(active, -gp.GRB.INFINITY, 0),
                          ub=np.where(active, gp.GRB.INFINITY, 0))
        # A free epigraph variable makes the dual weights sum to one.
        t = model.addVar(lb=-gp.GRB.INFINITY)
        constraints = []
        for coordinates, estimate in zip(projections, estimates):
            residual = model.addMVar(len(coordinates), lb=-gp.GRB.INFINITY)
            model.addConstr(residual == mu[coordinates] - estimate)
            constraints.append(model.addConstr(residual @ residual <= t))
        model.setObjective(t, gp.GRB.MINIMIZE)
        for attempt in range(2):
            model.optimize()
            # The certificate holds for any solution with simplex weights, so a
            # SUBOPTIMAL barrier stop is accepted whenever its gap is within tol.
            if model.SolCount > 0:
                candidate = np.where(active, mu.X, 0.0)
                try:
                    weights = np.maximum(-np.array([float(c.QCPi) for c in constraints]), 0)
                except gp.GurobiError:  # Duals may be unavailable off OPTIMAL.
                    weights = np.zeros(len(constraints))
                weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(weights), 1 / len(weights))
                counts, sums, constant = least_squares_statistics(
                    projections, estimates, d, weights=weights,
                )
                weighted_mu, _, _, gradient = fixed_support_least_squares(counts, sums, constant, support)
                # Residual evaluation avoids cancellation near a zero optimum.
                lower = float(weights @ np.sum((weighted_mu[projections] - estimates)**2, axis=1))
                upper = float(np.max(np.sum((candidate[projections] - estimates)**2, axis=1)))
                roundoff = 64 * np.finfo(float).eps * max(1, abs(lower), abs(upper))
                if (np.isfinite(candidate).all() and np.isfinite(gradient).all()
                        and np.isfinite([lower, upper]).all()
                        and -roundoff <= upper - lower <= max(tol, rtol * abs(upper))):
                    return candidate, min(lower, upper), upper, gradient
            elif model.Status not in (gp.GRB.SUBOPTIMAL, gp.GRB.NUMERIC):
                raise RuntimeError(f"Max aggregation QCP failed: Gurobi status {model.Status}")
            if attempt == 0:
                model.Params.NumericFocus = 3
                # Retry tighter: a looser tolerance cannot certify a small gap.
                model.Params.BarQCPConvTol = 1e-10
                model.reset()
        raise RuntimeError("Max aggregation QCP did not certify the requested primal-dual gap")


def refine_max_estimate(projections, estimates, support, initial_mu, *, tol=1e-5, env=None):
    """Minimize summed loss on the same support under the incumbent max cap.

    Convex QP outer approximations avoid barrier degeneracy at a minimax
    solution. A feasible incumbent and the QP lower bound certify the secondary
    gap to tol * max(1, sum_loss). Max-cap feasibility allows
    1e-9 * max(1, cap), matching the normalized solver precision.
    """
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")
    support = np.asarray(support)
    initial_mu = np.asarray(initial_mu, dtype=float)
    counts, sums, constant = least_squares_statistics(projections, estimates, len(support))
    least_squares, *_ = fixed_support_least_squares(counts, sums, constant, support)
    cap = float(np.max(np.sum((initial_mu[projections] - estimates)**2, axis=1)))
    if np.max(np.sum((least_squares[projections] - estimates)**2, axis=1)) <= cap:
        return least_squares

    scale = max(1.0, np.sqrt(cap))
    targets, bound = estimates / scale, cap / scale**2
    best = initial_mu / scale
    best_value = float(np.sum((best[projections] - targets)**2))
    feasibility_slack = 1e-9 * max(1, bound)
    active = (support > 0.5) & (counts > 0)
    lower_bounds = np.where(active, -np.inf, 0.)
    upper_bounds = np.where(active, np.inf, 0.)
    for coordinates, target in zip(projections, targets):
        selected = active[coordinates]
        radius = np.sqrt(max(0.0, bound - float(np.sum(target[~selected]**2))))
        indices = coordinates[selected]
        lower_bounds[indices] = np.maximum(lower_bounds[indices], target[selected] - radius)
        upper_bounds[indices] = np.minimum(upper_bounds[indices], target[selected] + radius)
    # Preserve the known feasible point when interval arithmetic rounds inward.
    lower_bounds, upper_bounds = np.minimum(lower_bounds, best), np.maximum(upper_bounds, best)
    with gp.Model(env=env) as model:
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.Method = 1
        model.Params.FeasibilityTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        mu = model.addMVar(len(support), lb=lower_bounds, ub=upper_bounds)
        residuals = model.addMVar(targets.size, lb=-gp.GRB.INFINITY)
        model.addConstr(residuals == mu[projections.ravel()] - targets.ravel())
        model.setObjective(residuals @ residuals, gp.GRB.MINIMIZE)
        point = best.copy()
        for iteration in range(1000):
            residual = point[projections] - targets
            losses = np.sum(residual**2, axis=1)
            for j in np.flatnonzero((losses > bound) | (iteration == 0)):
                indices = projections[j][active[projections[j]]]
                gradient = 2 * residual[j][active[projections[j]]]
                norm = np.linalg.norm(gradient)
                if norm > 0:
                    model.addConstr((gradient / norm) @ (mu[indices] - point[indices]) <= (bound - losses[j]) / norm)
            model.optimize()
            if model.Status != gp.GRB.OPTIMAL:
                raise RuntimeError(f"Max LS refinement QP failed: Gurobi status {model.Status}")
            point = np.where(active, mu.X, 0.0)
            # Interpolate with the feasible incumbent to preserve the true max
            # cap, rather than accepting the QP's linearized constraints alone.
            residual = best[projections] - targets
            direction = point[projections] - best[projections]
            linear = np.sum(residual * direction, axis=1)
            quadratic = np.sum(direction**2, axis=1)
            slack = np.maximum(0, bound + feasibility_slack - np.sum(residual**2, axis=1))
            root = np.sqrt(linear**2 + quadratic * slack)
            steps = np.full(len(estimates), np.inf)
            positive = linear > 0
            np.divide(slack, root + linear, out=steps, where=positive)
            np.divide(root - linear, quadratic, out=steps, where=(~positive) & (quadratic > 0))
            candidate = best + min(1.0, float(steps.min())) * (point - best)
            value = float(np.sum((candidate[projections] - targets)**2))
            if value < best_value:
                best, best_value = candidate, value
            if best_value - model.ObjVal <= tol * max(1 / scale**2, best_value):
                return best * scale
        raise RuntimeError("Max aggregation LS refinement reached its iteration limit")


def aggregate_estimates(
    projections, estimates, d, s, *, objective="least_squares", tol=1e-5,
    rtol=1e-5, max_iter=1000, env=None,
):
    """Aggregate projected means with the shared support cutting plane.

    Both objectives use squared residuals. Start at the zero vector; uncovered
    coordinates remain zero. The returned bounds/gap concern aggregation only.
    """
    projections, estimates = np.asarray(projections), np.asarray(estimates, dtype=float)
    if objective not in ("least_squares", "max"):
        raise ValueError("objective must be 'least_squares' or 'max'")
    if (projections.ndim != 2 or min(projections.shape) == 0
            or projections.shape != estimates.shape or not np.isfinite(estimates).all()
            or not np.issubdtype(projections.dtype, np.integer)
            or np.any(projections < 0) or np.any(projections >= d)):
        raise ValueError("require matching finite (J, r) estimates and coordinate indices in [0, d)")
    counts, sums, constant = least_squares_statistics(projections, estimates, d)
    if objective == "least_squares":
        def solve_support(z):
            mu, lower, _, gradient = fixed_support_least_squares(counts, sums, constant, z)
            upper = float(np.sum((mu[projections] - estimates)**2))
            return mu, lower, upper, gradient
        initial_upper = constant
    else:
        def solve_support(z):
            return fixed_support_max(projections, estimates, z, tol=tol / 4, rtol=rtol / 4, env=env)
        initial_upper = float(np.max(np.sum(estimates**2, axis=1)))
    mu, info = cutting_plane(
        s, np.zeros(d), np.zeros(d), initial_upper, solve_support, tol,
        max_iter=max_iter, rtol=rtol,
    )
    info["refined"] = False
    if objective == "max" and info["converged"]:
        info["objective_before_refinement"] = info["objective"]
        info["refinement_max_slack"] = 1e-9 * max(1, info["objective"])
        mu = refine_max_estimate(projections, estimates, info["support"], mu, env=env)
        info["refined"] = True
        info["objective"] = float(np.max(np.sum((mu[projections] - estimates)**2, axis=1)))
        info["gap"] = max(0.0, info["objective"] - info["lower_bound"])
        info["gap_tolerance"] = max(tol, rtol * abs(info["objective"]))
        info["converged"] = info["gap"] <= info["gap_tolerance"]
    return mu, info


def projected_estimation(
    data, s, epsilon, lambda_upper, delta=0.05, *, r, tol=1e-5, seed=None, C=2,
    objective="least_squares", aggregation_tol=1e-5, aggregation_rtol=1e-5,
    max_iter=1000, max_inner_iter=1000, oracle_strategy="hybrid",
    oracle_formulation="indicator", perspective=False,
):
    """Return (mu_hat, info) from a random partition into J = d/r blocks of r coordinates.

    r must divide d. The dense stage replaces (d, s) by (r, min(r, s)) in the K
    rule and uses V_{min(2s,r)}. Projections share the seed's block partition;
    the coordinate permutation comes from an independent child stream.
    Aggregation minimizes the sum or maximum squared
    residual under ||mu||_0 <= s using utils.cutting_plane, with no thresholding.
    tol controls the dense stage and defaults to 1e-5 (also when None).
    Aggregation stops at a squared-loss gap of
    max(aggregation_tol, aggregation_rtol * abs(objective)); max aggregation
    then minimizes the sum on the selected support under its previous max cap,
    up to the reported refinement_max_slack for numerical feasibility.
    info['tol'] and info['gap'] describe aggregation, while projection_tol
    records the dense-stage tolerance. An outer limit returns converged=False;
    uncertified inner solves raise. Runtime includes all stages.
    """
    start = perf_counter()
    if objective not in ("least_squares", "max"):
        raise ValueError("objective must be 'least_squares' or 'max'")
    if not np.isfinite(aggregation_tol) or aggregation_tol <= 0:
        raise ValueError("aggregation_tol must be positive and finite")
    if not np.isfinite(aggregation_rtol) or aggregation_rtol < 0:
        raise ValueError("aggregation_rtol must be nonnegative and finite")
    for name, value in (("max_iter", max_iter), ("max_inner_iter", max_inner_iter)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must have shape (n, d)")
    d = data.shape[1]
    if not isinstance(s, (int, np.integer)) or not 1 <= s <= d:
        raise ValueError("s must be an integer with 1 <= s <= d")
    tol = 1e-5 if tol is None else tol
    seed_sequence = np.random.SeedSequence(seed)
    projections = random_coordinate_projections(d, r, np.random.default_rng(seed_sequence.spawn(1)[0]))
    J = len(projections)
    projected_s = min(s, r)
    estimates = np.empty((J, r))
    stats = {"oracle_calls": 0, "inner_iterations": 0, "constraints": 0,
             "heuristic_calls": 0, "heuristic_cuts": 0, "deferred_supports": 0,
             "deterministic_certificates": 0}

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        for j, coordinates in enumerate(projections):
            block_means, center, _, _, projected_tol = mom_initialization(
                data[:, coordinates], projected_s, epsilon, lambda_upper, delta, tol,
                seed_sequence, C=C,
            )
            estimates[j], info = dense_estimation(
                block_means, center, projected_s, projected_tol,
                max_inner_iter=max_inner_iter, env=env,
                oracle_strategy=oracle_strategy, oracle_formulation=oracle_formulation,
                perspective=perspective,
            )
            for key in stats:
                stats[key] += info[key]
        estimate, aggregation_info = aggregate_estimates(
            projections, estimates, d, s, objective=objective, tol=aggregation_tol,
            rtol=aggregation_rtol, max_iter=max_iter, env=env,
        )
    stats.update(aggregation_info)
    stats.update({
        "C": C, "K": len(block_means), "J": J, "r": r,
        "projection_tol": projected_tol, "aggregation_tol": aggregation_tol,
        "aggregation_rtol": aggregation_rtol,
        "aggregation": "cutting_plane", "objective_type": objective,
        "uncovered": d - len(np.unique(projections)), "runtime": perf_counter() - start,
        "oracle_strategy": oracle_strategy, "oracle_formulation": oracle_formulation,
        "perspective": perspective,
    })
    return estimate, stats
