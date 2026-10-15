"""Algorithm 1: SOCP constraint generation with an integer separation oracle.

F(mu) = max_{v in V} median_k <block_mean_k - mu, v>, with V the unit vectors having at most
min(2s, d) nonzeros. Bounds and gaps hold within solver tolerances.
"""

from functools import partial
from fractions import Fraction
from time import perf_counter

import gurobipy as gp
import numpy as np

from utils import check_sparsity, complexity, cutting_plane, gurobi_env, mom_blocks, new_model, odd_ceil, top_s

MAX_INNER_ITER = 1000
SDP_TIME_LIMIT = 0.25  # suggested native budget (seconds) per SDP screen; pass sdp_time_limit=SDP_TIME_LIMIT to enable
RTOL = 1e-3  # default tol = RTOL * deterministic upper bound at the initial estimate


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
    """Bound F(mu) by the median of the rows' largest-min(2s, d) coordinate norms."""
    residuals, k, _ = _oracle_data(block_means, mu, s)
    return float(np.median(_row_bounds(residuals, k)))


def default_tol(tol, rtol, initial_upper):
    """tol if given, else rtol * initial_upper (rtol when that bound is 0), a scale-equivariant tolerance."""
    if tol is not None:
        return tol
    if not 0 < rtol < np.inf:
        raise ValueError("rtol must be positive and finite")
    return rtol * initial_upper if initial_upper > 0 else rtol


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


def _finish_separation(model, v, zeta, threshold, tol, witness, empty_pair, cap=np.inf, screened=None):
    """Solve a separation MIP and return (lower_F, upper_F, (S, B)) from its incumbent and bound.

    With a threshold the solve stops once the incumbent exceeds it (a violated cut) or the bound
    reaches it (a certificate); otherwise the absolute gap must be at most tol. screened, if not
    None, bounds every solution the screening removed, so F <= max(bound, screened) and an
    infeasible screened model certifies F <= screened.
    """
    stopped = False
    if threshold is None:
        model.optimize()
    else:
        # Half the MIP gap, but above solver precision, keeps the recomputed witness above the threshold.
        margin = max(tol / 2, 1e-9 * max(1.0, abs(threshold)))

        def decide(model, where):
            nonlocal stopped
            if where != gp.GRB.Callback.MIP:
                return
            incumbent = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
            bound = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
            if ((abs(incumbent) < gp.GRB.INFINITY / 2 and incumbent > threshold + margin)
                    or (abs(bound) < gp.GRB.INFINITY / 2 and bound <= threshold)):
                stopped = True
                model.terminate()

        model.optimize(decide)
    floor = 0.0 if screened is None else screened
    if screened is not None and model.Status in (gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD):
        if threshold is None or floor > threshold:
            raise RuntimeError("Screened separation model is infeasible above the threshold")
        return 0.0, floor, empty_pair
    if model.Status != gp.GRB.OPTIMAL and not (stopped and model.Status == gp.GRB.INTERRUPTED):
        raise RuntimeError(f"Separation oracle failed: Gurobi status {model.Status}")
    if model.SolCount == 0:
        upper = max(floor, min(cap, max(0.0, float(model.ObjBound))))
        if threshold is None or upper > threshold:
            raise RuntimeError("Separation oracle stopped without a certificate or incumbent")
        return 0.0, upper, empty_pair
    d = len(v.X)
    S = np.flatnonzero(zeta.X > 0.5) if zeta is not None else np.arange(d)
    direction = np.zeros(d)
    direction[S] = v.X[S]
    lower, pair, _ = witness(direction / max(1.0, np.linalg.norm(direction)))
    upper = max(lower, floor, min(cap, float(model.ObjBound)))
    # A decided threshold query needs no gap: a screened model may finish OPTIMAL with
    # upper = screened <= threshold before the callback stops it.
    decided = threshold is not None and (lower > threshold or upper <= threshold)
    if model.Status == gp.GRB.INTERRUPTED and not decided:
        raise RuntimeError("Separation oracle stopped without a valid cut or certificate")
    if not decided and upper - lower > tol:
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


def _pair_minima(rows, k, iterations=30):
    """For all row pairs (i < j), golden-section values of min_a ||a rows_i + (1-a) rows_j||_(k).

    Each value is attained at some a in [0, 1], so it upper-bounds the exact minimum.
    """
    first, second = np.triu_indices(len(rows), 1)
    a, b = rows[first], rows[second]

    def norm(t):
        return _row_bounds(t[:, None] * a + (1 - t[:, None]) * b, k)

    ratio = (np.sqrt(5) - 1) / 2
    low, high = np.zeros(len(first)), np.ones(len(first))
    x1, x2 = high - ratio, low + ratio
    f1, f2 = norm(x1), norm(x2)
    for _ in range(iterations):
        left = f1 <= f2  # the minimum lies in [low, x2]
        high, low = np.where(left, x2, high), np.where(left, low, x1)
        x1, x2 = np.where(left, high - ratio * (high - low), x2), np.where(left, x1, low + ratio * (high - low))
        f_new = norm(np.where(left, x1, x2))
        f1, f2 = np.where(left, f_new, f2), np.where(left, f1, f_new)
    return first, second, np.minimum(f1, f2)


def separation_oracle(block_means, mu, s, tol, *, decision_threshold=None, env=None, warm_start=None):
    """Solve (10), returning (lower_F, upper_F, (S, B)).

    The incumbent gives lower_F and Gurobi's global bound gives upper_F. Without a threshold,
    tol is the absolute gap. With one, stop once the incumbent exceeds it (a violated cut) or
    the bound reaches it (a certificate). Rows enter through big-M constraints with
    M_i = ||r_i||_(q) + max level. A threshold G also screens: a selected row i needs
    ||r_i||_(q) >= G, and two selected rows i, j need min_a ||a r_i + (1-a) r_j||_(q) >= G, since
    a q-sparse unit v with r_i @ v, r_j @ v >= G has every convex combination above G.
    """
    residuals, q, h = _oracle_data(block_means, mu, s)
    K, d = residuals.shape
    witness = partial(_direction_witness, residuals, h=h)
    row_bounds = _row_bounds(residuals, q)
    upper_cutoff = float(np.median(row_bounds))
    with new_model(env, MIPGap=0, MIPGapAbs=tol / 2, FeasibilityTol=1e-9, IntFeasTol=1e-9,
                   OptimalityTol=1e-9, BarQCPConvTol=1e-8, NonConvex=0) as model:
        v = model.addMVar(d, lb=-1, ub=1)
        zeta = model.addMVar(d, vtype=gp.GRB.BINARY) if q < d else None  # dense: no support choice
        b = model.addMVar(K, vtype=gp.GRB.BINARY)
        level = model.addVar(lb=0, ub=upper_cutoff)
        model.addConstr(v @ v <= 1)
        if zeta is not None:
            model.addConstr(v <= zeta)
            model.addConstr(-v <= zeta)
            model.addConstr(zeta.sum() <= q)
        model.addConstr(b.sum() == h)
        # |r_i @ v| <= ||r_i||_(q) for q-sparse unit v, so an unselected row never binds.
        model.addConstr(residuals @ v >= level - (row_bounds + upper_cutoff) * (1 - b))
        screened = None
        weak, conflicts = np.zeros(K, dtype=bool), []
        if decision_threshold is not None:
            screened = 0.0
            weak = row_bounds < decision_threshold
            if weak.any():
                b[np.flatnonzero(weak)].UB = 0
                screened = float(row_bounds[weak].max())
            strong = np.flatnonzero(~weak)
            first, second, minima = _pair_minima(residuals[strong], q)
            conflict = minima < decision_threshold
            conflicts = list(zip(strong[first[conflict]], strong[second[conflict]]))
            for i, j in conflicts:
                model.addConstr(b[i] + b[j] <= 1)
            if conflict.any():
                screened = max(screened, float(minima[conflict].max()))
        model.setObjective(level, gp.GRB.MAXIMIZE)
        if warm_start is not None:
            start_lower, (start_S, start_B), direction = witness(_sparse_direction(warm_start, q))
            chosen = np.isin(np.arange(K), start_B)
            # A start the screening removed is worth at most screened <= threshold, and Gurobi 13
            # can fail with a spurious "Q matrix is not PSD" while repairing it, so skip it.
            if not chosen[weak].any() and not any(chosen[i] and chosen[j] for i, j in conflicts):
                v.Start = direction
                b.Start = chosen.astype(float)
                level.Start = min(start_lower, upper_cutoff)
                if zeta is not None:
                    zeta.Start = np.isin(np.arange(d), start_S).astype(float)
        return _finish_separation(model, v, zeta, decision_threshold, tol, witness, ((0,), tuple(range(h))),
                                  screened=screened)


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
    three restricted solves; a repeated support is solved until upper_F - L <= tol/4, unless
    L >= best_upper - tol. Then the support cannot improve the incumbent by more than tol and its
    cut ends the outer search if the master returns to it, so certification is skipped.
    best_upper tracks the smallest returned U, which is the incumbent of utils.cutting_plane
    when this object is its solve_support with the same tol.
    Before exact separation, a bounded-effort SDP can certify F(mu) < L + inner_tol.
    An optional geometric screen runs before the SDP: plane cuts, then sparse-support box
    certification. In that mode the heuristic cut threshold is L + sep_tol; the certificate
    threshold remains L + inner_tol. Algorithm 2 disables both MoM screens.
    """

    def __init__(self, data, s, tol, env, center, initial_mu, *, sdp_time_limit=0.,
                 geometric_time_limit=0., geometric_max_nodes=5000):
        if not np.isfinite(sdp_time_limit) or sdp_time_limit < 0:
            raise ValueError("sdp_time_limit must be finite and nonnegative (0 disables the SDP)")
        if not np.isfinite(geometric_time_limit) or geometric_time_limit < 0:
            raise ValueError("geometric_time_limit must be finite and nonnegative")
        if not isinstance(geometric_max_nodes, (int, np.integer)) or geometric_max_nodes < 0:
            raise ValueError("geometric_max_nodes must be a nonnegative integer")
        self.data, self.s, self.env = data, s, env
        self.sdp_time_limit, self._sdp_certifier = sdp_time_limit, None
        self.geometric_time_limit = geometric_time_limit
        self.geometric_max_nodes = geometric_max_nodes
        self.tol, self.inner_tol, self.sep_tol = tol, tol / 4, tol / 8
        self.stats = {"oracle_calls": 0, "oracle_cuts": 0, "oracle_certificates": 0, "heuristic_calls": 1,
                      "heuristic_cuts": 0, "inner_iterations": 0, "deferred_supports": 0,
                      "deterministic_certificates": 0, "revisit_prunes": 0,
                      "sdp_calls": 0, "sdp_certificates": 0, "sdp_fallbacks": 0,
                      "geometric_calls": 0, "geometric_cuts": 0, "geometric_certificates": 0,
                      "geometric_fallbacks": 0, "geometric_nodes": 0, "geometric_supports": 0,
                      "restricted_time": 0.0, "heuristic_time": 0.0, "oracle_time": 0.0,
                      "sdp_certificate_time": 0.0, "geometric_time": 0.0}
        self.direction, self.visited = None, set()
        self.upper = self.best_upper = self.upper_bound(initial_mu)
        _, pair, _ = self._timed("heuristic", initial_mu, -1, None)
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

    def geometric(self, mu, cut_threshold, cert_threshold, warm_start):
        """Bounded plane search and verified geometric certification, for Algorithm 1 only."""
        from geometric_separation import GeometricResult, geometric_separation
        start = perf_counter()
        residuals, q, _ = _oracle_data(self.data, mu, self.s)
        # The geometric kernel certifies the represented residuals. Account for the
        # rounding in data-mu as well: |v_j| <= 1 bounds each row error by its top-q
        # absolute coordinate errors. This uniform bound also bounds median error.
        mu_exact = [Fraction(float(value)) for value in mu]
        error = max(sum(sorted((abs(Fraction(float(value)) - center - Fraction(float(residual)))
                                for value, center, residual in zip(row, mu_exact, residual_row)),
                               reverse=True)[:q], Fraction(0))
                    for row, residual_row in zip(self.data, residuals))

        def outward(value, upward):
            rounded = float(value)
            needs_rounding = Fraction(rounded) < value if upward else Fraction(rounded) > value
            if needs_rounding:
                rounded = float(np.nextafter(rounded, np.inf if upward else -np.inf))
            return rounded

        cut = outward(Fraction(float(cut_threshold)) + error, True)
        cert = outward(Fraction(float(cert_threshold)) - error, False)
        if not 0 < cut < cert:
            return GeometricResult("UNKNOWN", reason="residual_roundoff_exhausted_threshold_gap")
        result = geometric_separation(residuals, q, cut, cert,
                                      time_limit=max(0., self.geometric_time_limit - (perf_counter() - start)),
                                      max_nodes=self.geometric_max_nodes,
                                      max_nodes_per_support=self.geometric_max_nodes,
                                      warm_start=warm_start)
        result.stats["residual_roundoff_bound"] = outward(error, True)
        if result.status == "CUT":
            result.lower = outward(Fraction(float(result.lower)) - error, False)
        return result

    def sdp_certificate(self, mu, threshold):
        """Only a corrected SDP count upper bound < h certifies F(mu) < threshold."""
        try:
            if self._sdp_certifier is None:
                from sdp_separation import SDPSeparationCertifier
                self._sdp_certifier = SDPSeparationCertifier(self.data, self.s, time_limit=self.sdp_time_limit)
            count_upper = self._sdp_certifier.upper_bound(mu, threshold)
        except ImportError:
            return False  # SDP dependencies are optional for the exact Gurobi path.
        return bool(np.isfinite(count_upper) and count_upper < (len(self.data) + 1) // 2)

    def restricted(self, support):
        return solve_restricted_socp(self.data, support, self.M, self.pairs, env=self.env)

    def info(self):
        return {**self.stats, "constraints": len(self.pairs)}

    def _add(self, pair):
        self.pairs.append(pair)
        self.seen.add(pair)

    def _timed(self, phase, *args):
        start = perf_counter()
        result = getattr(self, phase)(*args)
        self.stats[f"{phase}_time"] += perf_counter() - start
        return result

    def _result(self, mu, L, U, gradient):
        self.best_upper = min(self.best_upper, U)
        return mu, L, U, gradient

    def __call__(self, support, certify=None):
        """Return (mu, L, U, gradient) for a binary support."""
        key = tuple(np.flatnonzero(support > 0.5))
        revisit = key in self.visited
        certify = revisit if certify is None else certify
        self.visited.add(key)
        for iteration in range(MAX_INNER_ITER):
            mu, L, gradient = self._timed("restricted", support)
            self.stats["inner_iterations"] += 1
            U = self.upper_bound(mu)
            if U - L <= self.inner_tol:
                self.stats["deterministic_certificates"] += 1
                return self._result(mu, L, U, gradient)
            if revisit and L >= self.best_upper - self.tol:
                self.stats["revisit_prunes"] += 1
                return self._result(mu, L, U, gradient)
            threshold = L + self.inner_tol
            cut_threshold = L + self.sep_tol if self.geometric_time_limit > 0 else threshold
            lower, pair, self.direction = self._timed("heuristic", mu, cut_threshold, self.direction)
            self.stats["heuristic_calls"] += 1
            violated = lower > cut_threshold and pair not in self.seen
            if violated:
                self._add(pair)
                self.stats["heuristic_cuts"] += 1
            if not certify and (not violated or iteration >= 2):
                self.stats["deferred_supports"] += 1
                return self._result(mu, L, U, gradient)
            if violated:
                continue
            if self.geometric_time_limit > 0:
                result = self._timed("geometric", mu, cut_threshold, threshold, self.direction)
                self.stats["geometric_calls"] += 1
                self.stats["geometric_nodes"] += result.stats.get("nodes", 0)
                self.stats["geometric_supports"] += result.stats.get("supports", 0)
                if result.status == "CERTIFIED":
                    self.stats["geometric_certificates"] += 1
                    return self._result(mu, L, min(U, threshold), gradient)
                if result.status == "CUT" and result.direction is not None:
                    residuals, q, h = _oracle_data(self.data, mu, self.s)
                    direction = np.asarray(result.direction, dtype=float)
                    if (direction.shape == (self.data.shape[1],) and np.isfinite(direction).all()
                            and np.count_nonzero(direction) <= q and np.linalg.norm(direction) > 0):
                        direction = direction / max(1., np.linalg.norm(direction))
                        lower, pair, direction = _direction_witness(residuals, direction, h)
                        if lower > cut_threshold and pair not in self.seen:
                            # Recheck the actual selected pair after any normalization/sign
                            # change, using the original data and mu without rounded subtraction.
                            values = {j: Fraction(float(direction[j])) for j in pair[0]}
                            ball_feasible = sum((v * v for v in values.values()), Fraction(0)) <= 1
                            strict_pair = ball_feasible and all(
                                sum(((Fraction(float(self.data[k, j])) - Fraction(float(mu[j]))) * v
                                     for j, v in values.items()), Fraction(0)) > Fraction(float(cut_threshold))
                                for k in pair[1])
                            if strict_pair:
                                self.direction = direction
                                self._add(pair)
                                self.stats["geometric_cuts"] += 1
                                continue
                self.stats["geometric_fallbacks"] += 1
            if self.sdp_time_limit > 0:
                certified = self._timed("sdp_certificate", mu, threshold)
                self.stats["sdp_calls"] += 1
                if certified:
                    self.stats["sdp_certificates"] += 1
                    # The SDP objective is a block count. The F upper bound is the threshold.
                    return self._result(mu, L, min(U, threshold), gradient)
                self.stats["sdp_fallbacks"] += 1
            lower, U, pair = self._timed("oracle", mu, threshold, self.direction)
            self.stats["oracle_calls"] += 1
            if U - L <= self.inner_tol:
                self.stats["oracle_certificates"] += 1
                return self._result(mu, L, U, gradient)
            if lower <= L:
                raise RuntimeError("Separation oracle did not find a violated constraint")
            if pair in self.seen:
                raise RuntimeError("An existing (S,B) constraint is still violated; check solver tolerances")
            self.stats["oracle_cuts"] += 1
            self._add(pair)
        raise RuntimeError("Inner constraint generation reached its iteration limit before certification")


def dense_estimation(block_means, s, tol, env, *, sdp_time_limit=0.,
                     geometric_time_limit=0., geometric_max_nodes=5000):
    """Algorithm 1 on the full support (no support cutting plane), starting at the coordinate median.

    Directions stay in V_min(2s,d); info['objective'] is within tol/4 of info['lower_bound'].
    """
    center = np.median(block_means, axis=0)
    solver = ConstraintGeneration(block_means, s, tol, env, center, center, sdp_time_limit=sdp_time_limit,
                                  geometric_time_limit=geometric_time_limit,
                                  geometric_max_nodes=geometric_max_nodes)
    mu, L, U, _ = solver(np.ones(len(center)), certify=True)
    return mu, {**solver.info(), "objective": U, "lower_bound": L}


def ip_estimation_from_block_means(block_means, s, *, tol=None, rtol=RTOL, max_iter=1000,
                                   sdp_time_limit=0., geometric_time_limit=0.,
                                   geometric_max_nodes=5000):
    """Algorithm 1 on the supplied odd collection of block means (used as given); returns (mu, info).

    Starts from the hard-thresholded coordinate median. Bounds refer to the continuous-direction
    objective; info['objective'] is a certified upper bound. The absolute tol defaults to rtol times
    the deterministic upper bound at the start (see default_tol). Inner failures raise, while the
    outer iteration limit returns info['converged'] = False. The SDP screen is optional: sdp_time_limit > 0
    enables it with that native budget per screen (seconds); the default 0 disables it. Model construction is included in sdp_certificate_time.
    geometric_time_limit > 0 enables the optional plane/BnB screen before the SDP and MIP;
    geometric_max_nodes is a global node budget per call (0 allows plane search only).
    """
    start = perf_counter()
    block_means = np.asarray(block_means, dtype=float)
    if block_means.ndim != 2 or len(block_means) % 2 != 1 or not np.isfinite(block_means).all():
        raise ValueError("block_means must be a finite (K, d) array with odd K")
    check_sparsity(s, block_means.shape[1])
    center = np.median(block_means, axis=0)
    support = top_s(center, s)
    initial_mu = center * support
    tol = default_tol(tol, rtol, deterministic_upper_bound(block_means, initial_mu, s))
    with gurobi_env() as env:
        solver = ConstraintGeneration(block_means, s, tol, env, center, initial_mu, sdp_time_limit=sdp_time_limit,
                                      geometric_time_limit=geometric_time_limit,
                                      geometric_max_nodes=geometric_max_nodes)
        mu, info = cutting_plane(s, support, initial_mu, solver.upper, solver, tol, max_iter)
    return mu, {**info, **solver.info(), "K": len(block_means), "runtime": perf_counter() - start}


def ip_estimation(data, s, epsilon, *, delta=0.05, C=3, seed=None, tol=None, rtol=RTOL, max_iter=1000,
                  subsample=False, sdp_time_limit=0., geometric_time_limit=0.,
                  geometric_max_nodes=5000):
    """Algorithm 1 on MoM block means; returns (mu, info).

    subsample=True is block subsampling: only K_sub = min(K, smallest odd integer >=
    C*max(s*log(e*d/s), log(4/delta))) blocks of the same partition, chosen uniformly without
    replacement (independent seed stream) and kept in block order, enter the optimization.
    With K_sub = K this is exactly Algorithm 1. Runtime includes block construction.
    sdp_time_limit > 0 enables the optional SDP screen with that native budget in seconds (default 0: off). tol and rtol
    are as in ip_estimation_from_block_means, computed on the blocks that enter the optimization.
    geometric_time_limit and geometric_max_nodes configure its optional geometric screen.
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
    mu, info = ip_estimation_from_block_means(blocks, s, tol=tol, rtol=rtol, max_iter=max_iter,
                                              sdp_time_limit=sdp_time_limit,
                                              geometric_time_limit=geometric_time_limit,
                                              geometric_max_nodes=geometric_max_nodes)
    return mu, {**info, **extra, "C": C, "K": K, "runtime": perf_counter() - start}
