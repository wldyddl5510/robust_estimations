"""Algorithm 1: SOCP constraint generation with an integer separation oracle.

F(mu) = max_{v in V} median_k <block_mean_k - mu, v>, with V the unit vectors having at most
min(2s, d) nonzeros. Bounds and gaps hold within solver tolerances.
"""

from functools import partial
from time import perf_counter

import gurobipy as gp
import numpy as np

from utils import check_sparsity, complexity, cutting_plane, gurobi_env, mom_blocks, new_model, odd_ceil, top_s

MAX_INNER_ITER = 1000


def _oracle_data(block_means, mu, s):
    """Residuals, direction sparsity min(2s, d), and the median's top count h = (K+1)/2."""
    block_means, mu = np.asarray(block_means, dtype=float), np.asarray(mu, dtype=float)
    K, d = block_means.shape
    if K % 2 != 1 or mu.shape != (d,):
        raise ValueError("require odd K and a length-d mu")
    return block_means - mu, min(2 * s, d), (K + 1) // 2


def _row_bounds(residuals, k):
    """Euclidean norm of each row's k largest-magnitude entries."""
    largest = np.partition(np.abs(residuals), residuals.shape[1] - k, axis=1)[:, -k:]
    return np.linalg.norm(largest, axis=1)


def deterministic_upper_bound(block_means, mu, s):
    """Bound F(mu) by the median of the rows' largest-2s coordinate norms."""
    residuals, k, _ = _oracle_data(block_means, mu, s)
    return float(np.median(_row_bounds(residuals, k)))


def _sparse_direction(vector, k):
    """Keep the k largest-magnitude entries and normalize (zero stays zero)."""
    direction = np.array(vector, dtype=float, copy=True)
    if k < len(direction):
        direction[np.argpartition(np.abs(direction), len(direction) - k)[:len(direction) - k]] = 0.0
    norm = np.linalg.norm(direction)
    if norm:
        direction /= norm
    return direction


def _direction_witness(residuals, direction, h):
    """(median projection, (S, B), direction), with the sign chosen to make the median >= 0."""
    projections = residuals @ direction
    lower = float(np.median(projections))
    if lower < 0:
        direction, projections, lower = -direction, -projections, -lower
    S = tuple(map(int, np.flatnonzero(direction))) or (0,)
    B = tuple(sorted(map(int, np.argsort(projections, kind="stable")[-h:])))
    return lower, (S, B), direction


def _heuristic_search(residuals, k, center, witness, fit_pair, propose, threshold, warm_start):
    """Shared heuristic separation; returns the best feasible (lower_F, (S, B), direction).

    Evaluates cheap sparse candidates (warm start, best coordinate, sparse center and mean, a few
    rows), then improves the two best distinct (S, B) pairs by at most two alternations of
    fit_pair(S, B) (a convex fixed-pair solve) and the proposals built from the selected rows.
    Returns as soon as a witness exceeds the threshold.
    """
    n, d = residuals.shape
    candidates = [] if warm_start is None else [_sparse_direction(warm_start, k)]
    coordinate = np.zeros(d)
    coordinate[np.argmax(np.abs(center))] = 1.0
    candidates += [coordinate, _sparse_direction(center, k), _sparse_direction(np.mean(residuals, axis=0), k)]
    # A few large and evenly spread rows keep the start cheap even for many rows.
    row_bounds = _row_bounds(residuals, k)
    rows = np.unique(np.r_[np.argsort(row_bounds)[-min(n, 6):], np.linspace(0, n - 1, min(n, 6), dtype=int)])
    candidates += [_sparse_direction(residuals[i], k) for i in rows]
    witnesses = sorted(map(witness, candidates), key=lambda w: w[0], reverse=True)
    best = witnesses[0]

    def enough(w):
        return threshold is not None and w[0] > threshold + 1e-8 * max(1.0, abs(threshold))

    if enough(best):
        return best
    starts = []
    for w in witnesses:
        if w[1] not in {start[1] for start in starts}:
            starts.append(w)
        if len(starts) == 2:
            break
    tried = set()
    for current in starts:
        for _ in range(2):
            if current[1] in tried:
                break
            tried.add(current[1])
            try:
                direction = fit_pair(*current[1])
            except gp.GurobiError:
                direction = None  # harmless: exact separation remains the fallback
            if direction is None:
                break
            current = witness(direction / max(1.0, np.linalg.norm(direction)))
            best = max(best, current, key=lambda w: w[0])
            if enough(best):
                return best
            # Proposals from the selected rows are kept only if they improve.
            for vector in propose(residuals[list(current[1][1])]):
                proposal = witness(_sparse_direction(vector, k))
                current = max(current, proposal, key=lambda w: w[0])
                best = max(best, proposal, key=lambda w: w[0])
            if enough(best):
                return best
    return best


def _finish_separation(model, v, zeta, threshold, tol, witness, empty_pair, cap=np.inf):
    """Solve a separation MIP and return (lower_F, upper_F, (S, B)) from its incumbent and bound.

    With a threshold the solve stops once the incumbent exceeds it (a violated cut) or the bound
    falls below it (a certificate); otherwise the absolute gap must be at most tol.
    """
    stopped = False
    if threshold is None:
        model.optimize()
    else:
        margin = 1e-5 * max(1.0, abs(threshold))

        def decide(model, where):
            nonlocal stopped
            if where != gp.GRB.Callback.MIP:
                return
            incumbent = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
            bound = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
            if ((abs(incumbent) < gp.GRB.INFINITY / 2 and incumbent > threshold + margin)
                    or (abs(bound) < gp.GRB.INFINITY / 2 and bound < threshold - margin)):
                stopped = True
                model.terminate()

        model.optimize(decide)
    if model.Status != gp.GRB.OPTIMAL and not (stopped and model.Status == gp.GRB.INTERRUPTED):
        raise RuntimeError(f"Separation oracle failed: Gurobi status {model.Status}")
    if model.SolCount == 0:
        upper = min(cap, max(0.0, float(model.ObjBound)))
        if threshold is None or upper > threshold:
            raise RuntimeError("Separation oracle stopped without a certificate or incumbent")
        return 0.0, upper, empty_pair
    d = len(v.X)
    S = np.flatnonzero(zeta.X > 0.5) if zeta is not None else np.arange(d)
    direction = np.zeros(d)
    direction[S] = v.X[S]
    lower, pair, _ = witness(direction / max(1.0, np.linalg.norm(direction)))
    upper = max(lower, 0.0, min(cap, float(model.ObjBound)))
    if model.Status == gp.GRB.INTERRUPTED:
        if not (lower > threshold or upper <= threshold):
            raise RuntimeError("Separation oracle stopped without a valid cut or certificate")
    elif upper - lower > tol:
        raise RuntimeError("Separation oracle did not certify the requested absolute gap")
    return lower, upper, pair


def heuristic_separation(block_means, mu, s, *, decision_threshold=None, warm_start=None, env=None):
    """Return a feasible (lower_F, (S, B), direction) witness for (10); never an upper certificate."""
    residuals, k, h = _oracle_data(block_means, mu, s)

    def fit_pair(S, B):
        """Unit direction on S maximizing the smallest projection over the blocks in B."""
        with new_model(env, NonConvex=0, BarQCPConvTol=1e-7) as model:
            v = model.addMVar(len(S), lb=-1, ub=1)
            level = model.addVar(lb=0)
            model.addConstr(v @ v <= 1)
            model.addConstr(residuals[np.ix_(B, S)] @ v >= level)
            model.setObjective(level, gp.GRB.MAXIMIZE)
            model.optimize()
            if model.Status != gp.GRB.OPTIMAL or not model.SolCount:
                return None
            direction = np.zeros(residuals.shape[1])
            direction[list(S)] = v.X
            return direction

    return _heuristic_search(residuals, k, np.median(residuals, axis=0),
                             partial(_direction_witness, residuals, h=h), fit_pair,
                             lambda selected: (np.mean(selected, axis=0), np.median(selected, axis=0)),
                             decision_threshold, warm_start)


def separation_oracle(block_means, mu, s, tol, *, decision_threshold=None, env=None, warm_start=None):
    """Solve (10), returning (lower_F, upper_F, (S, B)).

    The incumbent gives lower_F and Gurobi's global bound gives upper_F. Without a threshold,
    tol is the absolute gap. With one, stop once the incumbent exceeds it (a violated cut) or
    the bound falls below it (a certificate).
    """
    residuals, k, h = _oracle_data(block_means, mu, s)
    K, d = residuals.shape
    witness = partial(_direction_witness, residuals, h=h)
    upper_cutoff = float(np.median(_row_bounds(residuals, k)))
    with new_model(env, MIPGap=0, MIPGapAbs=tol / 2, FeasibilityTol=1e-9, IntFeasTol=1e-9,
                   OptimalityTol=1e-9, BarQCPConvTol=1e-8, NonConvex=0) as model:
        v = model.addMVar(d, lb=-1, ub=1)
        zeta = model.addMVar(d, vtype=gp.GRB.BINARY) if k < d else None  # dense: no support choice
        b = model.addMVar(K, vtype=gp.GRB.BINARY)
        level = model.addVar(lb=0, ub=upper_cutoff)
        model.addConstr(v @ v <= 1)
        if zeta is not None:
            model.addConstr(v <= zeta)
            model.addConstr(-v <= zeta)
            model.addConstr(zeta.sum() <= k)
        model.addConstr(b.sum() == h)
        for i in range(K):
            model.addConstr((b[i] == 1) >> (residuals[i] @ v >= level))
        model.setObjective(level, gp.GRB.MAXIMIZE)
        if warm_start is not None:
            start_lower, (start_S, start_B), direction = witness(_sparse_direction(warm_start, k))
            v.Start = direction
            b.Start = np.isin(np.arange(K), start_B).astype(float)
            level.Start = min(start_lower, upper_cutoff)
            if zeta is not None:
                zeta.Start = np.isin(np.arange(d), start_S).astype(float)
        return _finish_separation(model, v, zeta, decision_threshold, tol, witness, ((0,), tuple(range(h))))


def solve_restricted_socp(block_means, support, M, pairs, *, env=None):
    """Solve (17) over the pairs, returning (feasible_mu, L, cut_gradient)."""
    with new_model(env, QCPDual=1, NonConvex=0, FeasibilityTol=1e-9, OptimalityTol=1e-9,
                   BarQCPConvTol=1e-8) as model:
        mu = model.addMVar(len(support), lb=-gp.GRB.INFINITY)
        r = model.addVar(lb=0)
        upper = model.addConstr(mu <= M * support)
        lower = model.addConstr(-mu <= M * support)
        for S, B in pairs:
            alpha = model.addMVar(len(B), lb=0)
            residual = model.addMVar(len(S), lb=-gp.GRB.INFINITY)
            model.addConstr(alpha.sum() == 1)
            model.addConstr(residual == block_means[np.ix_(B, S)].T @ alpha - mu[list(S)])
            model.addConstr(residual @ residual <= r * r)  # convex because r >= 0
        model.setObjective(r, gp.GRB.MINIMIZE)
        model.optimize()
        if model.Status == gp.GRB.SUBOPTIMAL:
            model.Params.NumericFocus = 3
            model.Params.BarQCPConvTol = 1e-6
            model.Params.FeasibilityTol = 1e-8
            model.Params.OptimalityTol = 1e-8
            model.reset()
            model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"Restricted SOCP failed: Gurobi status {model.Status}")
        # Pi of a <= constraint in a minimization is nonpositive: g = -M * (lambda + rho).
        return np.clip(mu.X, -M * support, M * support), float(model.ObjVal), M * (upper.Pi + lower.Pi)


class ConstraintGeneration:
    """Inner loop of Algorithm 1, used as the solve_support callback of utils.cutting_plane.

    Alternates the restricted problem and separation on one support at a time, keeping the
    generated (S, B) pairs, solver counters, and the last heuristic direction across supports.
    Cheap heuristic cuts come first. A support's first visit returns a valid cut after at most
    three restricted solves; a repeated support is solved until upper_F - L <= tol/4.
    Algorithm 2 overrides the four statistic-specific methods below.
    """

    def __init__(self, data, s, tol, env, center, initial_mu):
        self.data, self.s, self.env = data, s, env
        self.inner_tol, self.sep_tol = tol / 4, tol / 8
        self.stats = {"oracle_calls": 0, "heuristic_calls": 1, "heuristic_cuts": 0,
                      "inner_iterations": 0, "deferred_supports": 0, "deterministic_certificates": 0}
        self.direction, self.visited = None, set()
        self.upper = self.upper_bound(initial_mu)
        _, pair, _ = self.heuristic(initial_mu, -1, None)
        self.pairs, self.seen = [pair], {pair}
        # Coordinate directions give |mu_j - center_j| <= F(mu) <= upper for every improving mu.
        self.M = np.abs(center) + self.upper + 1e-8 * max(1, np.max(np.abs(center)), self.upper)

    def upper_bound(self, mu):
        return deterministic_upper_bound(self.data, mu, self.s)

    def heuristic(self, mu, threshold, warm_start):
        return heuristic_separation(self.data, mu, self.s, decision_threshold=threshold,
                                    warm_start=warm_start, env=self.env)

    def oracle(self, mu, threshold, warm_start):
        return separation_oracle(self.data, mu, self.s, self.sep_tol, decision_threshold=threshold,
                                 warm_start=warm_start, env=self.env)

    def restricted(self, support):
        return solve_restricted_socp(self.data, support, self.M, self.pairs, env=self.env)

    def info(self):
        return {**self.stats, "constraints": len(self.pairs)}

    def _add(self, pair):
        self.pairs.append(pair)
        self.seen.add(pair)

    def __call__(self, support, certify=None):
        """Return (mu, L, U, gradient) for a binary support."""
        key = tuple(np.flatnonzero(support > 0.5))
        certify = key in self.visited if certify is None else certify
        self.visited.add(key)
        for iteration in range(MAX_INNER_ITER):
            mu, L, gradient = self.restricted(support)
            self.stats["inner_iterations"] += 1
            U = self.upper_bound(mu)
            if U - L <= self.inner_tol:
                self.stats["deterministic_certificates"] += 1
                return mu, L, U, gradient
            threshold = L + self.inner_tol
            lower, pair, self.direction = self.heuristic(mu, threshold, self.direction)
            self.stats["heuristic_calls"] += 1
            violated = lower > threshold and pair not in self.seen
            if violated:
                self._add(pair)
                self.stats["heuristic_cuts"] += 1
            if not certify and (not violated or iteration >= 2):
                self.stats["deferred_supports"] += 1
                return mu, L, U, gradient
            if violated:
                continue
            lower, U, pair = self.oracle(mu, threshold, self.direction)
            self.stats["oracle_calls"] += 1
            if U - L <= self.inner_tol:
                return mu, L, U, gradient
            if lower <= L:
                raise RuntimeError("Separation oracle did not find a violated constraint")
            if pair in self.seen:
                raise RuntimeError("An existing (S,B) constraint is still violated; check solver tolerances")
            self._add(pair)
        raise RuntimeError("Inner constraint generation reached its iteration limit before certification")


def dense_estimation(block_means, s, tol, env):
    """Algorithm 1 on the full support (no support cutting plane), starting at the coordinate median.

    Directions stay in V_min(2s,d); info['objective'] is within tol/4 of info['lower_bound'].
    """
    center = np.median(block_means, axis=0)
    solver = ConstraintGeneration(block_means, s, tol, env, center, center)
    mu, L, U, _ = solver(np.ones(len(center)), certify=True)
    return mu, {**solver.info(), "objective": U, "lower_bound": L}


def ip_estimation_from_block_means(block_means, s, *, tol=1e-5, max_iter=1000):
    """Algorithm 1 on the supplied odd collection of block means (used as given); returns (mu, info).

    Starts from the hard-thresholded coordinate median. Bounds refer to the continuous-direction
    objective; info['objective'] is a certified upper bound. Inner failures raise, while the
    outer iteration limit returns info['converged'] = False.
    """
    start = perf_counter()
    block_means = np.asarray(block_means, dtype=float)
    if block_means.ndim != 2 or len(block_means) % 2 != 1 or not np.isfinite(block_means).all():
        raise ValueError("block_means must be a finite (K, d) array with odd K")
    check_sparsity(s, block_means.shape[1])
    center = np.median(block_means, axis=0)
    support = top_s(center, s)
    initial_mu = center * support
    with gurobi_env() as env:
        solver = ConstraintGeneration(block_means, s, tol, env, center, initial_mu)
        mu, info = cutting_plane(s, support, initial_mu, solver.upper, solver, tol, max_iter)
    return mu, {**info, **solver.info(), "K": len(block_means), "runtime": perf_counter() - start}


def ip_estimation(data, s, epsilon, *, delta=0.05, C=2, seed=None, tol=1e-5, max_iter=1000, subsample=False):
    """Algorithm 1 on MoM block means; returns (mu, info).

    subsample=True is block subsampling: only K_sub = min(K, smallest odd integer >=
    C*max(s*log(e*d/s), log(4/delta))) blocks of the same partition, chosen uniformly without
    replacement (independent seed stream) and kept in block order, enter the optimization.
    With K_sub = K this is exactly Algorithm 1. Runtime includes block construction.
    """
    start = perf_counter()
    blocks = mom_blocks(data, s, epsilon, delta=delta, C=C, seed=seed)
    K, d = blocks.shape
    extra = {}
    if subsample:
        K_sub = min(K, odd_ceil(C * max(complexity(s, d), np.log(4.) - np.log(delta))))
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(2,)))
        selected = np.sort(rng.permutation(K)[:K_sub])
        blocks = blocks[selected]
        extra = {"K_sub": K_sub, "selected_block_indices": selected.tolist()}
    mu, info = ip_estimation_from_block_means(blocks, s, tol=tol, max_iter=max_iter)
    return mu, {**info, **extra, "C": C, "K": K, "runtime": perf_counter() - start}
