"""Sparse aggregation of means from general linear projections."""

import gurobipy as gp
import numpy as np

from utils import cutting_plane


def _inputs(projections, estimates, M, objective, tol, rtol):
    P = np.asarray(projections, dtype=float)
    y = np.asarray(estimates, dtype=float)
    M = np.asarray(M, dtype=float)
    if (P.ndim != 3 or min(P.shape) == 0 or y.shape != P.shape[:2]
            or M.shape != (P.shape[2],) or not np.isfinite(P).all()
            or not np.isfinite(y).all() or not np.isfinite(M).all()
            or np.any(M <= 0)):
        raise ValueError("require finite P (J, r, d), estimates (J, r), and positive M (d,)")
    if objective not in ("least_squares", "max"):
        raise ValueError("objective must be 'least_squares' or 'max'")
    if not np.isfinite(tol) or tol <= 0 or not np.isfinite(rtol) or rtol < 0:
        raise ValueError("tol must be positive and rtol nonnegative, both finite")
    return P, y, M


def _support(support, d):
    z = np.asarray(support, dtype=float)
    if z.shape != (d,) or not np.isin(z, [0, 1]).all():
        raise ValueError("support must be a binary vector of length d")
    return z


def _residuals(P, y, mu):
    return P @ mu - y


def _losses(P, y, mu):
    return np.sum(_residuals(P, y, mu)**2, axis=1)


def _configure(model):
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.NonConvex = 0
    model.Params.FeasibilityTol = 1e-9
    model.Params.OptimalityTol = 1e-9


def _weighted_least_squares(P, y, z, M, weights, env):
    """Solve a convex box QP, retaining zero curvature in unobserved directions."""
    if not np.any(z):
        return np.zeros(len(M))
    scale = max(1.0, float(np.max(np.linalg.norm(y, axis=1))))
    bounds = M * z / scale
    A = (np.sqrt(weights)[:, None, None] * P).reshape(-1, len(M))
    target = (np.sqrt(weights)[:, None] * y / scale).ravel()
    with gp.Model(env=env) as model:
        _configure(model)
        model.Params.Method = 1
        x = model.addMVar(len(M), lb=-bounds, ub=bounds)
        residual = model.addMVar(target.size, lb=-gp.GRB.INFINITY)
        model.addConstr(residual == A @ x - target)
        model.setObjective(residual @ residual, gp.GRB.MINIMIZE)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            # Tiny max-dual weights can confuse presolve despite a finite box.
            model.Params.Presolve = 0
            model.Params.DualReductions = 0
            model.Params.NumericFocus = 3
            model.reset()
            model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"Haar aggregation QP failed: Gurobi status {model.Status}")
        return np.clip(x.X * scale, -M * z, M * z)


def _certificate(P, y, z, M, point, weights):
    """A weighted-loss tangent minimized over each support box is a valid cut."""
    residual = _residuals(P, y, point)
    value = float(weights @ np.sum(residual**2, axis=1))
    slope = 2 * np.einsum("j,jrd,jr->d", weights, P, residual)
    gradient = -M * np.abs(slope)
    linear = float(slope @ point)
    bound_term = float(gradient @ z)
    # Round downward instead of treating an approximate primal optimum as L.
    roundoff = 64 * np.finfo(float).eps * max(1, abs(value), abs(linear), abs(bound_term))
    lower = value - linear + bound_term - roundoff
    if not np.isfinite(lower) or not np.isfinite(gradient).all():
        raise RuntimeError("Nonfinite Haar aggregation certificate")
    return float(lower), gradient


def fixed_support_aggregation(
    projections, estimates, support, M, *, objective="least_squares",
    tol=1e-6, rtol=1e-6, env=None,
):
    """Return (mu, L, U, g) on a binary support with -M*z <= mu <= M*z.

    L + g @ (z_new-z) lower-bounds the boxed objective on any other support.
    For max loss, nonnegative QCP duals normalized to sum one produce a
    weighted squared loss below the max; a box QP tightens its tangent cut.
    Jr < d is allowed. No ridge or additional rank assumption is imposed.
    """
    P, y, M = _inputs(projections, estimates, M, objective, tol, rtol)
    z = _support(support, len(M))
    if objective == "least_squares" or not np.any(z) or len(P) == 1:
        weights = np.ones(len(P))
        if objective == "max":
            weights = np.zeros(len(P))
            weights[np.argmax(np.sum(y**2, axis=1))] = 1
        mu = _weighted_least_squares(P, y, z, M, weights, env)
        lower, gradient = _certificate(P, y, z, M, mu, weights)
        losses = _losses(P, y, mu)
        upper = float(losses.sum() if objective == "least_squares" else losses.max())
        if not np.isfinite(upper) or upper - lower > max(tol, rtol * abs(upper)):
            raise RuntimeError("Haar aggregation QP did not certify the requested gap")
        return mu, lower, upper, gradient

    scale = max(1.0, float(np.max(np.linalg.norm(y, axis=1))))
    bounds = M * z / scale
    with gp.Model(env=env) as model:
        _configure(model)
        model.Params.QCPDual = 1
        model.Params.BarQCPConvTol = 1e-8
        x = model.addMVar(len(M), lb=-bounds, ub=bounds)
        t = model.addVar(lb=-gp.GRB.INFINITY)
        constraints = []
        for projection, target in zip(P, y / scale):
            residual = model.addMVar(len(target), lb=-gp.GRB.INFINITY)
            model.addConstr(residual == projection @ x - target)
            constraints.append(model.addConstr(residual @ residual <= t))
        model.setObjective(t, gp.GRB.MINIMIZE)
        for attempt in range(4):
            model.optimize()
            if model.SolCount:
                candidate = np.clip(x.X * scale, -M * z, M * z)
                try:
                    weights = np.maximum(-np.array([float(c.QCPi) for c in constraints]), 0)
                except gp.GurobiError:
                    weights = np.zeros(len(P))
                if np.isfinite(weights).all() and weights.sum() > 0:
                    weights /= weights.sum()
                else:
                    weights = np.full(len(P), 1 / len(P))
                point = _weighted_least_squares(P, y, z, M, weights, env)
                lower, gradient = _certificate(P, y, z, M, point, weights)
                upper = float(_losses(P, y, candidate).max())
                if (np.isfinite(candidate).all() and np.isfinite(upper)
                        and 0 <= upper - lower <= max(tol, rtol * abs(upper))):
                    return candidate, lower, upper, gradient
            elif model.Status not in (gp.GRB.SUBOPTIMAL, gp.GRB.NUMERIC):
                raise RuntimeError(f"Haar max aggregation QCP failed: Gurobi status {model.Status}")
            model.Params.NumericFocus = 3
            model.Params.BarQCPConvTol = 1e-10 if attempt == 0 else 1e-12
            if attempt == 2:
                model.Params.Presolve = 0
            model.reset()
    raise RuntimeError("Haar max aggregation QCP did not certify the requested gap")


def _refine_max(P, y, z, M, initial_mu, tol, rtol, env):
    """Minimize LS on the selected box/support under the incumbent max cap.

    Linear outer approximations avoid a degenerate barrier solve at the cap.
    Every returned candidate is checked against the actual quadratic losses.
    """
    cap = float(_losses(P, y, initial_mu).max())
    least_squares = _weighted_least_squares(P, y, z, M, np.ones(len(P)), env)
    if _losses(P, y, least_squares).max() <= cap:
        return least_squares
    scale = max(1.0, np.sqrt(cap))
    targets, bound = y / scale, cap / scale**2
    best = initial_mu / scale
    best_value = float(_losses(P, targets, best).sum())
    slack = 1e-9 * max(1.0, bound)
    bounds = M * z / scale
    with gp.Model(env=env) as model:
        _configure(model)
        model.Params.Method = 1
        x = model.addMVar(len(M), lb=-bounds, ub=bounds)
        residual = model.addMVar(targets.size, lb=-gp.GRB.INFINITY)
        model.addConstr(residual == P.reshape(-1, len(M)) @ x - targets.ravel())
        model.setObjective(residual @ residual, gp.GRB.MINIMIZE)
        point = best.copy()
        for iteration in range(1000):
            residual = _residuals(P, targets, point)
            losses = np.sum(residual**2, axis=1)
            for j in np.flatnonzero((losses > bound) | (iteration == 0)):
                gradient = 2 * P[j].T @ residual[j]
                norm = np.linalg.norm(gradient * z)
                if norm > 0:
                    model.addConstr((gradient / norm) @ (x - point) <= (bound - losses[j]) / norm)
            model.optimize()
            if model.Status != gp.GRB.OPTIMAL:
                # Degenerate cap tangents can defeat dual simplex even with
                # a feasible incumbent. Retry the same QP with primal simplex.
                model.Params.Method = 0
                model.Params.Presolve = 0
                model.Params.DualReductions = 0
                model.Params.NumericFocus = 3
                model.reset()
                model.optimize()
            if model.Status != gp.GRB.OPTIMAL:
                raise RuntimeError(f"Haar max refinement QP failed: Gurobi status {model.Status}")
            point = np.clip(x.X, -bounds, bounds)
            # Keep the original cap-feasible anchor: a previous best can be
            # exactly on the numerical slack boundary and permit only step 0.
            anchor = initial_mu / scale
            residual = _residuals(P, targets, anchor)
            direction = P @ (point - anchor)
            linear = np.sum(residual * direction, axis=1)
            quadratic = np.sum(direction**2, axis=1)
            available = np.maximum(0, bound + slack - np.sum(residual**2, axis=1))
            root = np.sqrt(linear**2 + quadratic * available)
            steps = np.full(len(P), np.inf)
            positive = linear > 0
            np.divide(available, root + linear, out=steps, where=positive)
            np.divide(root - linear, quadratic, out=steps, where=(~positive) & (quadratic > 0))
            candidate = anchor + min(1.0, float(steps.min())) * (point - anchor)
            value = float(_losses(P, targets, candidate).sum())
            if value < best_value:
                best, best_value = candidate, value
            if best_value - model.ObjVal <= max(tol / scale**2, rtol * abs(best_value)):
                answer = np.clip(best * scale, -M * z, M * z)
                roundoff = 64 * np.finfo(float).eps * max(1, cap)
                if _losses(P, y, answer).max() <= cap + 1e-9 * max(1, cap) + roundoff:
                    return answer
                raise RuntimeError("Haar max refinement violated its max cap")
    raise RuntimeError("Haar max aggregation refinement reached its iteration limit")


def aggregate_haar_estimates(
    projections, estimates, s, M, *, objective="least_squares", tol=1e-5,
    rtol=1e-5, max_iter=1000, env=None,
):
    """Sparse LS/max aggregation within the supplied data-driven coefficient box.

    This box defines the estimator domain, without claiming equivalence to an
    unboxed problem. Global sparsity uses utils.cutting_plane, starting at zero.
    Max aggregation subsequently minimizes LS on its selected support while
    retaining the incumbent max cap up to 1e-9 relative numerical slack.
    """
    P, y, M = _inputs(projections, estimates, M, objective, tol, rtol)
    d = len(M)
    losses = np.sum(y**2, axis=1)
    initial_upper = float(losses.sum() if objective == "least_squares" else losses.max())

    def solve_support(z):
        return fixed_support_aggregation(
            P, y, z, M, objective=objective, tol=tol / 4, rtol=rtol / 4, env=env,
        )

    mu, info = cutting_plane(
        s, np.zeros(d), np.zeros(d), initial_upper, solve_support, tol,
        max_iter=max_iter, rtol=rtol,
    )
    info["refined"] = False
    if objective == "max" and info["converged"]:
        info["objective_before_refinement"] = info["objective"]
        info["refinement_max_slack"] = 1e-9 * max(1, info["objective"])
        z = np.asarray(info["support"], dtype=float)
        mu = _refine_max(P, y, z, M, mu, tol, rtol, env)
        info["refined"] = True
        info["objective"] = float(_losses(P, y, mu).max())
        info["gap"] = max(0.0, info["objective"] - info["lower_bound"])
        info["gap_tolerance"] = max(tol, rtol * abs(info["objective"]))
        info["converged"] = info["gap"] <= info["gap_tolerance"]
    return mu, info
