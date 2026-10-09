"""Algorithm 2: sparse trimmed-mean estimation via (21), (22), and support cuts.

Unlike Algorithm 1, every oracle uses raw observations, with k removals per
tail. Bounds and gaps refer to optimization, within floating-point precision.
"""

from time import perf_counter

import gurobipy as gp
import numpy as np

from IP_algorithm import _oracle_options, _oracle_stats, _row_bounds, _sparse_direction
from utils import cutting_plane


def _validate_k(n, k):
    if not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    # Preserve the draft's rule: never silently clamp the trimming count.
    assert 2 * k < n, f"require k < n/2; got k={k}, n={n}"


def trimmed_mean(values, k, axis=0):
    """Average after removing exactly k observations from each tail."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 0 or not np.isfinite(values).all():
        raise ValueError("values must be a finite array with a sample axis")
    n = values.shape[axis]
    _validate_k(n, k)
    ordered = np.sort(values, axis=axis)
    return np.take(ordered, np.arange(k, n - k), axis=axis).mean(axis=axis)


def _data(data, s):
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or min(data.shape) == 0 or not np.isfinite(data).all():
        raise ValueError("data must be a finite, nonempty (n, d) array")
    if not isinstance(s, (int, np.integer)) or not 1 <= s <= data.shape[1]:
        raise ValueError("s must be an integer with 1 <= s <= d")
    return data


def trimmed_initialization(data, s, epsilon, delta=0.05, tol=1e-5):
    """Return (coordinate center, support, sparse center, k, tolerance)."""
    data = _data(data, s)
    if not 0 <= epsilon < 0.5 or not 0 < delta < 1:
        raise ValueError("require 0 <= epsilon < 0.5 and 0 < delta < 1")
    tol = 1e-5 if tol is None else tol
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")
    n, d = data.shape
    k = int(np.floor(epsilon * n)) + max(
        int(np.ceil(np.log(2.) - np.log(delta))), int(np.ceil(epsilon * n / 2)),
    )
    _validate_k(n, k)
    center = trimmed_mean(data, k)
    support = np.zeros(d)
    support[np.argsort(-np.abs(center), kind="stable")[:s]] = 1
    return center, support, center * support, k, float(tol)


def _oracle_data(data, mu, s, k):
    data = _data(data, s)
    mu = np.asarray(mu, dtype=float)
    if mu.shape != (data.shape[1],) or not np.isfinite(mu).all():
        raise ValueError("mu must be a finite length-d vector")
    _validate_k(len(data), k)
    return data - mu, min(2 * s, data.shape[1]), len(data) - 2 * k


def deterministic_upper_bound(data, mu, s, k):
    """Bound F(mu) by the trimmed mean of top-min(2s,d) residual norms."""
    residuals, p, _ = _oracle_data(data, mu, s, k)
    return float(trimmed_mean(_row_bounds(residuals, p), k))


def _direction_witness(residuals, direction, k):
    projections = residuals @ direction
    lower = float(trimmed_mean(projections, k))
    if lower < 0:
        direction = -direction
        projections = -projections
        lower = -lower
    S = tuple(map(int, np.flatnonzero(direction))) or (0,)
    # B contains n-k observations, not just the central n-2k observations.
    B = tuple(sorted(map(int, np.argsort(projections, kind="stable")[k:])))
    return lower, (S, B), direction


def heuristic_separation(
    data, mu, s, k, *, decision_threshold=None, warm_start=None, env=None,
):
    """Return a feasible lower-bound witness, never an upper certificate.

    Try sparse directions, then at most four fixed-(S,B) hinge SOCPs. Refresh B
    from the top n-k projections and propose supports from the selected samples.
    """
    residuals, p, q = _oracle_data(data, mu, s, k)
    n, d = residuals.shape
    if decision_threshold is not None and not np.isfinite(decision_threshold):
        raise ValueError("decision_threshold must be finite")
    candidates = []
    if warm_start is not None:
        warm_start = np.asarray(warm_start, dtype=float)
        if warm_start.shape != (d,) or not np.isfinite(warm_start).all():
            raise ValueError("warm_start must be a finite length-d direction")
        candidates.append(_sparse_direction(warm_start, p))
    center = trimmed_mean(residuals, k)
    coordinate = np.zeros(d)
    coordinate[np.argmax(np.abs(center))] = 1
    candidates.extend((coordinate, _sparse_direction(center, p),
                       _sparse_direction(np.mean(residuals, axis=0), p)))
    row_bounds = _row_bounds(residuals, p)
    row_ids = np.unique(np.r_[np.argsort(row_bounds)[-min(n, 6):],
                             np.linspace(0, n - 1, min(n, 6), dtype=int)])
    candidates.extend(_sparse_direction(residuals[i], p) for i in row_ids)
    witnesses = [_direction_witness(residuals, direction, k) for direction in candidates]
    witnesses.sort(key=lambda result: result[0], reverse=True)
    best = witnesses[0]

    def enough(result):
        return (decision_threshold is not None
                and result[0] > decision_threshold + 1e-8 * max(1., abs(decision_threshold)))

    if enough(best):
        return best
    seeds = []
    for witness in witnesses:
        if witness[1] not in {seed[1] for seed in seeds}:
            seeds.append(witness)
        if len(seeds) == 2:
            break
    tried = set()
    for seed in seeds:
        current = seed
        for _ in range(2):
            S, B = current[1]
            if (S, B) in tried:
                break
            tried.add((S, B))
            try:
                with gp.Model(env=env) as model:
                    model.Params.OutputFlag = 0
                    model.Params.Threads = 1
                    model.Params.NonConvex = 0
                    model.Params.BarQCPConvTol = 1e-8
                    v = model.addMVar(len(S), lb=-1, ub=1)
                    # For this fixed B, choose its q-th projection as theta.
                    bound = float(np.partition(row_bounds[list(B)], q - 1)[q - 1])
                    theta = model.addVar(lb=0, ub=bound)
                    u = model.addMVar(len(B), lb=0, ub=bound + row_bounds[list(B)])
                    model.addConstr(v @ v <= 1)
                    model.addConstr(u >= theta - residuals[np.ix_(B, S)] @ v)
                    model.setObjective(theta - u.sum() / q, gp.GRB.MAXIMIZE)
                    model.optimize()
                    if model.Status != gp.GRB.OPTIMAL or not model.SolCount:
                        break
                    direction = np.zeros(d)
                    direction[list(S)] = v.X
            except gp.GurobiError:
                # A local failure cannot certify convergence; exact separation
                # remains the fallback in constraint_generation.
                break
            direction /= max(1., np.linalg.norm(direction))
            current = _direction_witness(residuals, direction, k)
            if current[0] > best[0]:
                best = current
            if enough(best):
                return best
            selected = residuals[list(current[1][1])]
            for vector in (np.mean(selected, axis=0), center):
                proposal = _direction_witness(residuals, _sparse_direction(vector, p), k)
                if proposal[0] > current[0]:
                    current = proposal
                if proposal[0] > best[0]:
                    best = proposal
            if enough(best):
                return best
    return best


def separation_oracle(
    data, mu, s, k, tol, *, decision_threshold=None, env=None,
    formulation="indicator", perspective=False, warm_start=None,
):
    """Solve (21), returning (lower_F, upper_F, (S,B)).

    The feasible direction supplies lower_F; the global MIP bound supplies upper_F.
    Theta is bounded by the (n-k)-th ascending residual row norm, NOT by U_det.
    Both indicator and big-M formulations retain an optimal canonical threshold.
    """
    residuals, p, q = _oracle_data(data, mu, s, k)
    n, d = residuals.shape
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")
    if decision_threshold is not None and not np.isfinite(decision_threshold):
        raise ValueError("decision_threshold must be finite")
    _oracle_options("exact", formulation)
    if warm_start is not None:
        warm_start = np.asarray(warm_start, dtype=float)
        if warm_start.shape != (d,) or not np.isfinite(warm_start).all():
            raise ValueError("warm_start must be a finite length-d direction")
        warm_start = _sparse_direction(warm_start, p)
    row_bounds = _row_bounds(residuals, p)
    theta_ub = float(np.partition(row_bounds, n - k - 1)[n - k - 1])
    big_m = theta_ub + row_bounds
    upper_cutoff = float(trimmed_mean(row_bounds, k))
    with gp.Model(env=env) as model:
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.MIPGap = 0
        model.Params.MIPGapAbs = tol / 2
        model.Params.FeasibilityTol = 1e-9
        model.Params.IntFeasTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        model.Params.BarQCPConvTol = 1e-9
        model.Params.NonConvex = 0
        v = model.addMVar(d, lb=-1, ub=1)
        zeta = model.addMVar(d, vtype=gp.GRB.BINARY) if p < d else None
        b = model.addMVar(n, vtype=gp.GRB.BINARY)
        theta = model.addVar(lb=0, ub=theta_ub)
        u = model.addMVar(n, lb=0, ub=big_m)
        model.addConstr(v @ v <= 1)
        if zeta is not None:
            model.addConstr(v <= zeta)
            model.addConstr(-v <= zeta)
            model.addConstr(zeta.sum() <= p)
            if perspective:
                energy = model.addMVar(d, lb=0, ub=1)
                for j in range(d):
                    model.addConstr(v[j] * v[j] <= zeta[j] * energy[j])
                model.addConstr(energy.sum() <= 1)
        model.addConstr(b.sum() == n - k)
        for i in range(n):
            if formulation == "bigm":
                model.addConstr(u[i] >= theta - residuals[i] @ v - big_m[i] * (1 - b[i]))
                model.addConstr(u[i] <= big_m[i] * b[i])
            else:
                model.addConstr((b[i] == 1) >> (u[i] >= theta - residuals[i] @ v))
                model.addConstr((b[i] == 0) >> (u[i] == 0))
        objective = theta - u.sum() / q
        model.addConstr(objective <= upper_cutoff)
        model.setObjective(objective, gp.GRB.MAXIMIZE)
        if warm_start is not None:
            _, (start_S, start_B), direction = _direction_witness(residuals, warm_start, k)
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
                if perspective:
                    energy.Start = direction ** 2

        stopped = False

        def stop_when_decided(model, where):
            nonlocal stopped
            if where != gp.GRB.Callback.MIP:
                return
            incumbent = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
            bound = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
            margin = 1e-5 * max(1., abs(decision_threshold))
            if (abs(incumbent) < gp.GRB.INFINITY / 2
                    and incumbent > decision_threshold + margin):
                stopped = True
                model.terminate()
            elif (abs(bound) < gp.GRB.INFINITY / 2
                  and bound < decision_threshold - margin):
                stopped = True
                model.terminate()

        if decision_threshold is None:
            model.optimize()
        else:
            model.optimize(stop_when_decided)
        if model.Status != gp.GRB.OPTIMAL and not (stopped and model.Status == gp.GRB.INTERRUPTED):
            raise RuntimeError(f"Trimmed separation oracle failed: Gurobi status {model.Status}")
        if model.SolCount == 0:
            upper = min(upper_cutoff, max(0., float(model.ObjBound)))
            if decision_threshold is None or upper > decision_threshold:
                raise RuntimeError("Separation oracle stopped without a certificate or incumbent")
            return 0., upper, ((0,), tuple(range(n - k)))
        S = np.flatnonzero(zeta.X > 0.5) if zeta is not None else np.arange(d)
        direction = np.zeros(d)
        direction[S] = v.X[S]
        direction /= max(1., np.linalg.norm(direction))
        lower, pair, _ = _direction_witness(residuals, direction, k)
        upper = max(lower, 0., min(upper_cutoff, float(model.ObjBound)))
        if model.Status == gp.GRB.INTERRUPTED:
            if not (lower > decision_threshold or upper <= decision_threshold):
                raise RuntimeError("Separation oracle stopped without a valid cut or certificate")
        elif upper - lower > tol:
            raise RuntimeError("Separation oracle did not certify the requested absolute gap")
    return lower, upper, pair


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
    """Recover a cut when quadratic-cone dual attributes are too inaccurate.

    Here the cone vectors are primal variables. Re-evaluating their capped-simplex
    support functions and normalizing their norms gives a feasible dual certificate.
    """
    with gp.Model(env=env) as model:
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.NonConvex = 0
        model.Params.FeasibilityTol = 1e-9
        model.Params.OptimalityTol = 1e-9
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
            total = gp.quicksum(vectors[a][tuple(S).index(j)].item()
                               for a, (S, _) in enumerate(pairs) if j in S)
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
    """Repair weights to the capped simplex and directly evaluate the cones."""
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
        primal = max(primal, float(np.linalg.norm(
            data[np.ix_(B, S)].T @ a - candidate[list(S)],
        )))
    return primal


def solve_restricted_socp(data, support, M, pairs, k, *, env=None, tol=1e-7):
    """Solve (22), returning (feasible_mu, lower_cut_value, gradient).

    Reconstruct a feasible dual cut from cone vectors instead of anchoring an
    approximate box-dual slope at the primal objective. The capped simplex's
    linear minimum is the mean of its q smallest coefficients.
    """
    data = np.asarray(data, dtype=float)
    support, M = np.asarray(support, dtype=float), np.asarray(M, dtype=float)
    _validate_k(len(data), k)
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")
    q, d = len(data) - 2 * k, data.shape[1]
    with gp.Model(env=env) as model:
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.QCPDual = 1
        model.Params.NonConvex = 0
        model.Params.FeasibilityTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        model.Params.BarQCPConvTol = 1e-9
        mu = model.addMVar(d, lb=-gp.GRB.INFINITY)
        r = model.addVar(lb=0)
        model.addConstr(mu <= M * support)
        model.addConstr(-mu <= M * support)
        equalities, weights = [], []
        for S, B in pairs:
            if len(B) != len(data) - k or len(set(B)) != len(B):
                raise ValueError("each B must contain n-k distinct sample indices")
            alpha = model.addMVar(len(B), lb=0, ub=1 / q)
            weights.append(alpha)
            residual = model.addMVar(len(S), lb=-gp.GRB.INFINITY)
            model.addConstr(alpha.sum() == 1)
            equalities.append(model.addConstr(
                residual == data[np.ix_(B, S)].T @ alpha - mu[list(S)],
            ))
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
            # Interior mean coordinates have zero aggregate dual slope. Repair
            # that stationarity numerically before normalizing the cone vectors.
            # Any resulting vectors still define a valid cut below.
            interior = np.flatnonzero(
                np.abs(candidate) < M * support - 1e-8 * np.maximum(1., M),
            )
            for j in interior:
                entries = [(a, tuple(S).index(j)) for a, (S, _) in enumerate(pairs) if j in S]
                if entries:
                    correction = sum(vectors[a][b] for a, b in entries) / len(entries)
                    for a, b in entries:
                        vectors[a][b] -= correction
            lower, gradient = _support_cut(data, support, M, pairs, vectors, q)
            # Accept SUBOPTIMAL only from repaired primal/dual certificates.
            if best_primal - lower <= tol:
                return best_candidate, lower, gradient
        if best_candidate is None:
            raise RuntimeError("Restricted SOCP did not return a finite primal solution")
        lower, gradient = _explicit_dual_cut(data, support, M, pairs, q, best_primal, tol, env)
        return best_candidate, lower, gradient


def initial_constraint(
    data, center, initial_mu, s, k, sep_tol, *, env=None,
    oracle_strategy="hybrid", oracle_formulation="indicator", perspective=False,
):
    """Initialize U_best, a valid pair, and a box containing improving means."""
    if oracle_strategy == "hybrid":
        upper = deterministic_upper_bound(data, initial_mu, s, k)
        _, pair, _ = heuristic_separation(data, initial_mu, s, k, decision_threshold=-1, env=env)
    else:
        _, upper, pair = separation_oracle(
            data, initial_mu, s, k, sep_tol, env=env,
            formulation=oracle_formulation, perspective=perspective,
        )
    M = np.abs(center) + upper + 1e-8 * max(1., np.max(np.abs(center)), upper)
    return upper, pair, M


def constraint_generation(
    data, support, M, pairs, s, k, sep_tol, inner_tol, stats, *,
    max_inner_iter=1000, env=None, oracle_strategy="hybrid",
    oracle_formulation="indicator", perspective=False, certify=True, oracle_state=None,
):
    """Alternate (22) and (21); preserve pairs across outer support searches."""
    seen = set(pairs)
    oracle_state = {} if oracle_state is None else oracle_state
    for inner_iteration in range(max_inner_iter):
        candidate, L, gradient = solve_restricted_socp(
            data, support, M, pairs, k, env=env, tol=inner_tol / 4,
        )
        stats["inner_iterations"] += 1
        if oracle_strategy == "hybrid":
            U = deterministic_upper_bound(data, candidate, s, k)
            if U - L <= inner_tol:
                stats["deterministic_certificates"] += 1
                return candidate, L, U, gradient
            lower, pair, direction = heuristic_separation(
                data, candidate, s, k, decision_threshold=L + inner_tol,
                warm_start=oracle_state.get("direction"), env=env,
            )
            stats["heuristic_calls"] += 1
            oracle_state["direction"] = direction
            violated = lower > L + inner_tol and pair not in seen
            if violated:
                pairs.append(pair)
                seen.add(pair)
                stats["heuristic_cuts"] += 1
            if not certify and (not violated or inner_iteration >= 2):
                stats["deferred_supports"] += 1
                return candidate, L, U, gradient
            if violated:
                continue
        lower, U, pair = separation_oracle(
            data, candidate, s, k, sep_tol, decision_threshold=L + inner_tol, env=env,
            formulation=oracle_formulation, perspective=perspective,
            warm_start=oracle_state.get("direction"),
        )
        stats["oracle_calls"] += 1
        if U - L <= inner_tol:
            return candidate, L, U, gradient
        if lower <= L:
            raise RuntimeError("Separation oracle did not find a violated constraint")
        if pair in seen:
            raise RuntimeError("An existing (S,B) constraint is still violated; check solver tolerances")
        pairs.append(pair)
        seen.add(pair)
    raise RuntimeError("Inner constraint generation reached max_inner_iter before certification")


def trimmed_ip_estimation(
    data, s, epsilon, delta=0.05, *, tol=1e-5, max_iter=1000, max_inner_iter=1000,
    oracle_strategy="hybrid", oracle_formulation="indicator", perspective=False,
):
    """Return (mu_hat, info) using Algorithm 2 on raw observations.

    Use the draft's k exactly, asserting 2*k < n. No block randomization, block
    multiplier, or covariance bound is needed. tol=None also means 1e-5.
    Hybrid mode defers certification on first support visits; repeated supports
    use exact fallback. Heuristic values never replace upper certificates.
    Inner failures raise; an outer iteration limit returns converged=False.
    """
    start = perf_counter()
    _oracle_options(oracle_strategy, oracle_formulation)
    for name, value in (("max_iter", max_iter), ("max_inner_iter", max_inner_iter)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    data = _data(data, s)
    center, support, initial_mu, k, tol = trimmed_initialization(data, s, epsilon, delta, tol)
    inner_tol, sep_tol = tol / 4, tol / 8
    stats = _oracle_stats(oracle_strategy)
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", 1)
        env.start()
        initial_upper, first_pair, M = initial_constraint(
            data, center, initial_mu, s, k, sep_tol, env=env,
            oracle_strategy=oracle_strategy, oracle_formulation=oracle_formulation,
            perspective=perspective,
        )
        pairs, visited_supports, oracle_state = [first_pair], set(), {}

        def solve_support(current_support):
            key = tuple(np.flatnonzero(current_support > 0.5))
            certify = oracle_strategy == "exact" or key in visited_supports
            visited_supports.add(key)
            return constraint_generation(
                data, current_support, M, pairs, s, k, sep_tol, inner_tol, stats,
                max_inner_iter=max_inner_iter, env=env, oracle_strategy=oracle_strategy,
                oracle_formulation=oracle_formulation, perspective=perspective,
                certify=certify, oracle_state=oracle_state,
            )

        estimate, info = cutting_plane(
            s, support, initial_mu, initial_upper, solve_support, tol, max_iter=max_iter,
        )
    info.update(stats)
    info.update({
        "n": len(data), "k": k, "effective_n": len(data) - 2 * k,
        "constraints": len(pairs), "inner_tol": inner_tol, "sep_tol": sep_tol,
        "runtime": perf_counter() - start, "oracle_strategy": oracle_strategy,
        "oracle_formulation": oracle_formulation, "perspective": perspective,
    })
    return estimate, info
