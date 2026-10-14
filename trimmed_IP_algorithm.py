"""Algorithm 2: sparse trimmed-mean estimation via (21), (22), and support cuts.

Unlike Algorithm 1, every oracle uses the raw observations with k removals per tail.
Bounds and gaps refer to the optimization, within floating-point precision.
"""

from functools import partial
from time import perf_counter

import gurobipy as gp
import numpy as np

from IP_algorithm import (
    ConstraintGeneration, _finish_separation, _heuristic_search, _row_bounds, _sparse_direction,
)
from utils import check_data, cutting_plane, gurobi_env, new_model, top_s


def _validate_k(n, k):
    if not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    # The draft's rule: never silently clamp the trimming count.
    assert 2 * k < n, f"require k < n/2; got k={k}, n={n}"


def trimmed_mean(values, k, axis=0):
    """Average after removing exactly k observations from each tail."""
    values = np.asarray(values, dtype=float)
    n = values.shape[axis]
    _validate_k(n, k)
    return np.take(np.sort(values, axis=axis), np.arange(k, n - k), axis=axis).mean(axis=axis)


def trimmed_initialization(data, s, epsilon, delta=0.05):
    """Return (coordinate trimmed mean, its top-s indicator, k).

    k = floor(epsilon*n) + max(ceil(log(2/delta)), ceil(epsilon*n/2)), as in the draft.
    """
    if not 0 <= epsilon < 0.5 or not 0 < delta < 1:
        raise ValueError("require 0 <= epsilon < 0.5 and 0 < delta < 1")
    n = len(data)
    k = int(np.floor(epsilon * n)) + max(int(np.ceil(np.log(2.) - np.log(delta))), int(np.ceil(epsilon * n / 2)))
    center = trimmed_mean(data, k)
    return center, top_s(center, s), k


def _oracle_data(data, mu, s, k):
    """Residuals, direction sparsity p = min(2s, d), and q = n - 2k."""
    data, mu = np.asarray(data, dtype=float), np.asarray(mu, dtype=float)
    if mu.shape != (data.shape[1],):
        raise ValueError("mu must be a length-d vector")
    _validate_k(len(data), k)
    return data - mu, min(2 * s, data.shape[1]), len(data) - 2 * k


def deterministic_upper_bound(data, mu, s, k):
    """Bound F(mu) by the trimmed mean of the rows' top-min(2s,d) residual norms."""
    residuals, p, _ = _oracle_data(data, mu, s, k)
    return float(trimmed_mean(_row_bounds(residuals, p), k))


def _direction_witness(residuals, direction, k):
    """(trimmed-mean projection, (S, B), direction), with the sign making the value >= 0."""
    projections = residuals @ direction
    lower = float(trimmed_mean(projections, k))
    if lower < 0:
        direction, projections, lower = -direction, -projections, -lower
    S = tuple(map(int, np.flatnonzero(direction))) or (0,)
    # B holds the n-k largest projections, not just the central n-2k.
    B = tuple(sorted(map(int, np.argsort(projections, kind="stable")[k:])))
    return lower, (S, B), direction


def heuristic_separation(data, mu, s, k, *, decision_threshold=None, warm_start=None, env=None):
    """Return a feasible (lower_F, (S, B), direction) witness for (21); never an upper certificate."""
    residuals, p, q = _oracle_data(data, mu, s, k)
    row_bounds = _row_bounds(residuals, p)
    center = trimmed_mean(residuals, k)

    def fit_pair(S, B):
        """Unit direction on S maximizing the mean of the q smallest projections over B (hinge SOCP)."""
        with new_model(env, NonConvex=0, BarQCPConvTol=1e-8) as model:
            v = model.addMVar(len(S), lb=-1, ub=1)
            # For this fixed B, the q-th smallest row bound caps theta.
            bound = float(np.partition(row_bounds[list(B)], q - 1)[q - 1])
            theta = model.addVar(lb=0, ub=bound)
            u = model.addMVar(len(B), lb=0, ub=bound + row_bounds[list(B)])
            model.addConstr(v @ v <= 1)
            model.addConstr(u >= theta - residuals[np.ix_(B, S)] @ v)
            model.setObjective(theta - u.sum() / q, gp.GRB.MAXIMIZE)
            model.optimize()
            if model.Status != gp.GRB.OPTIMAL or not model.SolCount:
                return None
            direction = np.zeros(residuals.shape[1])
            direction[list(S)] = v.X
            return direction

    return _heuristic_search(residuals, p, center, partial(_direction_witness, residuals, k=k), fit_pair,
                             lambda selected: (np.mean(selected, axis=0), center), decision_threshold, warm_start)


def separation_oracle(data, mu, s, k, tol, *, decision_threshold=None, env=None, warm_start=None):
    """Solve (21), returning (lower_F, upper_F, (S, B)), with the same stopping rules as Algorithm 1.

    Theta is bounded by the (n-k)-th smallest residual row norm; the objective by the trimmed
    mean of the row norms.
    """
    residuals, p, q = _oracle_data(data, mu, s, k)
    n, d = residuals.shape
    witness = partial(_direction_witness, residuals, k=k)
    row_bounds = _row_bounds(residuals, p)
    theta_ub = float(np.partition(row_bounds, n - k - 1)[n - k - 1])
    upper_cutoff = float(trimmed_mean(row_bounds, k))
    with new_model(env, MIPGap=0, MIPGapAbs=tol / 2, FeasibilityTol=1e-9, IntFeasTol=1e-9,
                   OptimalityTol=1e-9, BarQCPConvTol=1e-9, NonConvex=0) as model:
        v = model.addMVar(d, lb=-1, ub=1)
        zeta = model.addMVar(d, vtype=gp.GRB.BINARY) if p < d else None
        b = model.addMVar(n, vtype=gp.GRB.BINARY)
        theta = model.addVar(lb=0, ub=theta_ub)
        u = model.addMVar(n, lb=0, ub=theta_ub + row_bounds)
        model.addConstr(v @ v <= 1)
        if zeta is not None:
            model.addConstr(v <= zeta)
            model.addConstr(-v <= zeta)
            model.addConstr(zeta.sum() <= p)
        model.addConstr(b.sum() == n - k)
        for i in range(n):
            model.addConstr((b[i] == 1) >> (u[i] >= theta - residuals[i] @ v))
            model.addConstr((b[i] == 0) >> (u[i] == 0))
        objective = theta - u.sum() / q
        model.addConstr(objective <= upper_cutoff)
        model.setObjective(objective, gp.GRB.MAXIMIZE)
        if warm_start is not None:
            _, (start_S, start_B), direction = witness(_sparse_direction(warm_start, p))
            projections = residuals @ direction
            start_theta = max(0., min(float(np.sort(projections)[n - k - 1]), theta_ub))
            v.Start = direction
            b.Start = np.isin(np.arange(n), start_B).astype(float)
            theta.Start = start_theta
            start_u = np.zeros(n)
            start_u[list(start_B)] = np.maximum(0., start_theta - projections[list(start_B)])
            u.Start = start_u
            if zeta is not None:
                zeta.Start = np.isin(np.arange(d), start_S).astype(float)
        return _finish_separation(model, v, zeta, decision_threshold, tol, witness, ((0,), tuple(range(n - k))),
                                  cap=upper_cutoff)


def _support_cut(data, support, M, pairs, vectors, q):
    """Turn arbitrary finite cone vectors into a valid support lower cut."""
    scale = max(1., sum(np.linalg.norm(w) for w in vectors)) * (1 + 1e-12)
    slope = np.zeros(data.shape[1])
    intercept = 0.
    for (S, B), w in zip(pairs, vectors):
        w = w / scale
        slope[list(S)] += w
        intercept += float(np.sort(data[np.ix_(B, S)] @ w)[:q].mean())
    gradient = -M * np.abs(slope)
    lower = float(intercept + gradient @ support)
    return (lower, gradient) if lower >= 0 else (0., np.zeros_like(slope))


def _explicit_dual_cut(data, support, M, pairs, q, primal, tol, env):
    """Recover a cut when the quadratic-cone dual attributes are too inaccurate.

    Here the cone vectors are primal variables. Re-evaluating their capped-simplex support
    functions and normalizing their norms gives a feasible dual certificate.
    """
    with new_model(env, NonConvex=0, FeasibilityTol=1e-9, OptimalityTol=1e-9) as model:
        radii = model.addMVar(len(pairs), lb=0, ub=1)
        model.addConstr(radii.sum() <= 1)
        vectors, objective = [], 0.
        for a, (S, B) in enumerate(pairs):
            w = model.addMVar(len(S), lb=-1, ub=1)
            vectors.append(w)
            theta = model.addVar(lb=-gp.GRB.INFINITY)
            u = model.addMVar(len(B), lb=0)
            model.addConstr(w @ w <= radii[a] * radii[a])
            model.addConstr(u >= theta - data[np.ix_(B, S)] @ w)
            objective += theta - u.sum() / q
        absolute = model.addMVar(data.shape[1], lb=0)
        for j in range(data.shape[1]):
            total = gp.quicksum(vectors[a][tuple(S).index(j)].item() for a, (S, _) in enumerate(pairs) if j in S)
            model.addConstr(absolute[j] >= total)
            model.addConstr(absolute[j] >= -total)
        model.setObjective(objective - (M * support) @ absolute, gp.GRB.MAXIMIZE)
        for attempt, barrier_tol in enumerate((1e-9, 1e-12)):
            if attempt:
                model.Params.NumericFocus = 3
                model.reset()
            model.Params.BarQCPConvTol = barrier_tol
            model.optimize()
            if model.Status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL) or not model.SolCount:
                continue
            values = [w.X for w in vectors]
            if not all(np.isfinite(w).all() for w in values):
                continue
            lower, gradient = _support_cut(data, support, M, pairs, values, q)
            if primal - lower <= tol:
                return lower, gradient
    raise RuntimeError("Restricted SOCP did not certify the requested dual gap")


def _restricted_primal_value(data, candidate, pairs, weights, q):
    """Repair the weights to the capped simplex and evaluate the cones directly."""
    primal = 0.
    for (S, B), raw in zip(pairs, weights):
        a = np.maximum(0., raw)
        if not np.isfinite(a).all():
            return np.inf
        if q == len(B) or a.sum() == 0:
            a = np.full(len(B), 1 / len(B))
        else:
            a /= a.sum()
            over = a > 1 / q
            if np.any(over):
                blend = float(np.max((a[over] - 1 / q) / (a[over] - 1 / len(B))))
                a = (1 - blend) * a + blend / len(B)
        primal = max(primal, float(np.linalg.norm(data[np.ix_(B, S)].T @ a - candidate[list(S)])))
    return primal


def solve_restricted_socp(data, support, M, pairs, k, *, env=None, tol=1e-7):
    """Solve (22), returning (feasible_mu, lower_cut_value, gradient).

    The cut is rebuilt from the cone vectors, as a feasible dual, instead of anchoring an
    approximate box-dual slope at the primal value. The capped simplex's linear minimum is the
    mean of its q smallest coefficients.
    """
    q, d = len(data) - 2 * k, data.shape[1]
    with new_model(env, QCPDual=1, NonConvex=0, FeasibilityTol=1e-9, OptimalityTol=1e-9,
                   BarQCPConvTol=1e-9) as model:
        mu = model.addMVar(d, lb=-gp.GRB.INFINITY)
        r = model.addVar(lb=0)
        model.addConstr(mu <= M * support)
        model.addConstr(-mu <= M * support)
        equalities, weights = [], []
        for S, B in pairs:
            alpha = model.addMVar(len(B), lb=0, ub=1 / q)
            weights.append(alpha)
            residual = model.addMVar(len(S), lb=-gp.GRB.INFINITY)
            model.addConstr(alpha.sum() == 1)
            equalities.append(model.addConstr(residual == data[np.ix_(B, S)].T @ alpha - mu[list(S)]))
            model.addConstr(residual @ residual <= r * r)
        model.setObjective(r)
        best_candidate, best_primal = None, np.inf
        for attempt, barrier_tol in enumerate((1e-9, 1e-12, 1e-14)):
            if attempt:
                model.Params.NumericFocus = 3
                model.reset()
            model.Params.BarQCPConvTol = barrier_tol
            model.optimize()
            if model.Status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
                raise RuntimeError(f"Trimmed restricted SOCP failed: Gurobi status {model.Status}")
            if not model.SolCount:
                continue
            candidate = np.clip(mu.X, -M * support, M * support)
            if not np.isfinite(candidate).all():
                continue
            primal = _restricted_primal_value(data, candidate, pairs, [a.X for a in weights], q)
            if primal < best_primal:
                best_candidate, best_primal = candidate, primal
            try:
                vectors = [np.asarray(eq.Pi) for eq in equalities]
            except (AttributeError, gp.GurobiError):
                continue
            if not all(np.isfinite(w).all() for w in vectors):
                continue
            # Interior mean coordinates have zero aggregate dual slope; repair that stationarity
            # numerically before normalizing. Any resulting vectors still give a valid cut.
            interior = np.flatnonzero(np.abs(candidate) < M * support - 1e-8 * np.maximum(1., M))
            for j in interior:
                entries = [(a, tuple(S).index(j)) for a, (S, _) in enumerate(pairs) if j in S]
                if entries:
                    correction = sum(vectors[a][b] for a, b in entries) / len(entries)
                    for a, b in entries:
                        vectors[a][b] -= correction
            lower, gradient = _support_cut(data, support, M, pairs, vectors, q)
            # SUBOPTIMAL is accepted only through these repaired primal/dual certificates.
            if best_primal - lower <= tol:
                return best_candidate, lower, gradient
        if best_candidate is None:
            raise RuntimeError("Restricted SOCP did not return a finite primal solution")
        lower, gradient = _explicit_dual_cut(data, support, M, pairs, q, best_primal, tol, env)
        return best_candidate, lower, gradient


class TrimmedConstraintGeneration(ConstraintGeneration):
    """Algorithm 1's inner loop with the trimmed-mean oracles (21) and (22) on raw samples."""

    def __init__(self, data, s, k, tol, env, center, initial_mu):
        self.k = k
        super().__init__(data, s, tol, env, center, initial_mu)

    def upper_bound(self, mu):
        return deterministic_upper_bound(self.data, mu, self.s, self.k)

    def heuristic(self, mu, threshold, warm_start):
        return heuristic_separation(self.data, mu, self.s, self.k, decision_threshold=threshold,
                                    warm_start=warm_start, env=self.env)

    def oracle(self, mu, threshold, warm_start):
        return separation_oracle(self.data, mu, self.s, self.k, self.sep_tol,
                                 decision_threshold=threshold, warm_start=warm_start, env=self.env)

    def restricted(self, support):
        return solve_restricted_socp(self.data, support, self.M, self.pairs, self.k, env=self.env,
                                     tol=self.inner_tol / 4)


def trimmed_ip_estimation(data, s, epsilon, *, delta=0.05, tol=1e-5, max_iter=1000):
    """Return (mu, info) using Algorithm 2 on the raw observations.

    No blocks, block multiplier, or covariance bound. Separation is hybrid as in Algorithm 1;
    inner failures raise, while the outer iteration limit returns info['converged'] = False.
    """
    start = perf_counter()
    data = check_data(data, s)
    center, support, k = trimmed_initialization(data, s, epsilon, delta)
    initial_mu = center * support
    with gurobi_env() as env:
        solver = TrimmedConstraintGeneration(data, s, k, tol, env, center, initial_mu)
        mu, info = cutting_plane(s, support, initial_mu, solver.upper, solver, tol, max_iter)
    return mu, {**info, **solver.info(), "n": len(data), "k": k, "effective_n": len(data) - 2 * k,
                "runtime": perf_counter() - start}
