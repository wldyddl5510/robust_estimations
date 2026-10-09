"""Algorithm 1: SOCP constraint generation with an integer separation oracle."""

from time import perf_counter

import gurobipy as gp
import numpy as np

from utils import cutting_plane, mom_initialization


def _oracle_data(block_means, mu, s):
    block_means = np.asarray(block_means, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if block_means.ndim != 2 or min(block_means.shape) == 0:
        raise ValueError("block_means must be a nonempty (K, d) array")
    K, d = block_means.shape
    if K % 2 != 1 or mu.shape != (d,):
        raise ValueError("require odd K and a length-d mu")
    if not np.isfinite(block_means).all() or not np.isfinite(mu).all():
        raise ValueError("block_means and mu must be finite")
    if not isinstance(s, (int, np.integer)) or not 1 <= s <= d:
        raise ValueError("s must be an integer with 1 <= s <= d")
    return block_means - mu, min(2 * s, d), (K + 1) // 2


def _row_bounds(residuals, k):
    largest = np.partition(np.abs(residuals), residuals.shape[1] - k, axis=1)[:, -k:]
    return np.linalg.norm(largest, axis=1)


def deterministic_upper_bound(block_means, mu, s):
    """Bound (10) by the median of the rows' largest-2s coordinate norms."""
    residuals, k, _ = _oracle_data(block_means, mu, s)
    return float(np.median(_row_bounds(residuals, k)))


def _sparse_direction(vector, k):
    """Project a candidate onto the feasible sparse unit ball."""
    direction = np.array(vector, dtype=float, copy=True)
    if k < len(direction):
        keep = np.argpartition(np.abs(direction), len(direction) - k)[-k:]
        mask = np.ones(len(direction), dtype=bool)
        mask[keep] = False
        direction[mask] = 0.0
    norm = np.linalg.norm(direction)
    if norm:
        direction /= norm
    return direction


def _direction_witness(residuals, direction, h):
    projections = residuals @ direction
    lower = float(np.median(projections))
    if lower < 0:
        direction = -direction
        projections = -projections
        lower = -lower
    S = tuple(map(int, np.flatnonzero(direction))) or (0,)
    B = tuple(sorted(map(int, np.argsort(projections, kind="stable")[-h:])))
    return lower, (S, B), direction


def heuristic_separation(
    block_means, mu, s, *, decision_threshold=None, warm_start=None, env=None,
):
    """Return a feasible (lower_F, (S,B), direction) witness, never a certificate.

    Try deterministic sparse directions first, then at most four convex
    fixed-(S,B) solves, alternating with top-half block selection and cheap
    support proposals. A threshold violation stops the search immediately.
    """
    residuals, k, h = _oracle_data(block_means, mu, s)
    K, d = residuals.shape
    if decision_threshold is not None and not np.isfinite(decision_threshold):
        raise ValueError("decision_threshold must be finite")
    candidates = []
    if warm_start is not None:
        warm_start = np.asarray(warm_start, dtype=float)
        if warm_start.shape != (d,) or not np.isfinite(warm_start).all():
            raise ValueError("warm_start must be a finite length-d direction")
        candidates.append(_sparse_direction(warm_start, k))
    coordinate_medians = np.median(residuals, axis=0)
    coordinate = np.zeros(d)
    coordinate[np.argmax(np.abs(coordinate_medians))] = 1.0
    candidates.extend((coordinate, _sparse_direction(coordinate_medians, k),
                       _sparse_direction(np.mean(residuals, axis=0), k)))
    # Bound initialization cost even when there are many blocks.
    row_bounds = _row_bounds(residuals, k)
    row_ids = np.unique(np.r_[np.argsort(row_bounds)[-min(K, 6):],
                             np.linspace(0, K - 1, min(K, 6), dtype=int)])
    candidates.extend(_sparse_direction(residuals[i], k) for i in row_ids)
    witnesses = [_direction_witness(residuals, direction, h) for direction in candidates]
    witnesses.sort(key=lambda result: result[0], reverse=True)
    best = witnesses[0]

    def enough(result):
        return (decision_threshold is not None
                and result[0] > decision_threshold + 1e-8 * max(1.0, abs(decision_threshold)))

    if enough(best):
        return best
    tried = set()
    # Two distinct initial pairs allow escaping a poor local block selection.
    seeds = []
    for witness in witnesses:
        if witness[1] not in {seed[1] for seed in seeds}:
            seeds.append(witness)
        if len(seeds) == 2:
            break
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
                    model.Params.BarQCPConvTol = 1e-7
                    v = model.addMVar(len(S), lb=-1, ub=1)
                    level = model.addVar(lb=0)
                    model.addConstr(v @ v <= 1)
                    model.addConstr(residuals[np.ix_(B, S)] @ v >= level)
                    model.setObjective(level, gp.GRB.MAXIMIZE)
                    model.optimize()
                    if model.Status != gp.GRB.OPTIMAL or not model.SolCount:
                        break
                    direction = np.zeros(d)
                    direction[list(S)] = v.X
            except gp.GurobiError:
                # This routine never provides an upper bound; exact separation
                # remains the caller's fallback if the local solve is unstable.
                break
            norm = np.linalg.norm(direction)
            direction /= max(1.0, norm)
            current = _direction_witness(residuals, direction, h)
            if current[0] > best[0]:
                best = current
            if enough(best):
                return best
            # New supports are proposed by the selected blocks and accepted
            # only if their directly evaluated feasible witness improves.
            selected = residuals[list(current[1][1])]
            for vector in (np.mean(selected, axis=0), np.median(selected, axis=0)):
                proposal = _direction_witness(residuals, _sparse_direction(vector, k), h)
                if proposal[0] > current[0]:
                    current = proposal
                if proposal[0] > best[0]:
                    best = proposal
            if enough(best):
                return best
    return best


def separation_oracle(
    block_means, mu, s, tol, *, decision_threshold=None, env=None,
    formulation="indicator", perspective=False, warm_start=None,
):
    """Solve (10), returning (lower_F, upper_F, (S, B)).

    The incumbent direction gives lower_F; Gurobi's global maximization bound
    gives upper_F. Symmetry in v makes maximizing the signed median equivalent
    to maximizing its absolute value. Without a decision_threshold, tol is an
    absolute objective gap. With one, stop as soon as either the incumbent
    exceeds it (a violated cut) or the global bound falls below it (certified).
    ``bigm`` uses explicit per-block bounds; ``indicator`` is an ablation.
    Optional perspective cones strengthen the support relaxation when 2s < d.
    """
    residuals, k, h = _oracle_data(block_means, mu, s)
    K, d = residuals.shape
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")
    if decision_threshold is not None and not np.isfinite(decision_threshold):
        raise ValueError("decision_threshold must be finite")
    if formulation not in ("bigm", "indicator"):
        raise ValueError("formulation must be 'bigm' or 'indicator'")
    if warm_start is not None:
        warm_start = np.asarray(warm_start, dtype=float)
        if warm_start.shape != (d,) or not np.isfinite(warm_start).all():
            raise ValueError("warm_start must be a finite length-d direction")
        warm_start = _sparse_direction(warm_start, k)
    row_bounds = _row_bounds(residuals, k)
    upper_cutoff = float(np.median(row_bounds))
    with gp.Model(env=env) as model:
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.MIPGap = 0
        model.Params.MIPGapAbs = tol / 2
        model.Params.FeasibilityTol = 1e-9
        model.Params.IntFeasTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        model.Params.BarQCPConvTol = 1e-8
        model.Params.NonConvex = 0
        v = model.addMVar(d, lb=-1, ub=1)
        # Dense directions need no support selection variables or constraints.
        zeta = model.addMVar(d, vtype=gp.GRB.BINARY) if k < d else None
        b = model.addMVar(K, vtype=gp.GRB.BINARY)
        level = model.addVar(lb=0, ub=upper_cutoff)
        model.addConstr(v @ v <= 1)
        if zeta is not None:
            model.addConstr(v <= zeta)
            model.addConstr(-v <= zeta)
            model.addConstr(zeta.sum() <= k)
            if perspective:
                energy = model.addMVar(d, lb=0, ub=1)
                for j in range(d):
                    model.addConstr(v[j] * v[j] <= zeta[j] * energy[j])
                model.addConstr(energy.sum() <= 1)
        model.addConstr(b.sum() == h)
        for i in range(K):
            if formulation == "bigm":
                # Inactive rows have projection >= -row_bounds[i], and level
                # <= upper_cutoff, so this M is valid separately for each row.
                big_m = upper_cutoff + row_bounds[i]
                model.addConstr(residuals[i] @ v >= level - big_m * (1 - b[i]))
            else:
                model.addConstr((b[i] == 1) >> (residuals[i] @ v >= level))
        model.setObjective(level, gp.GRB.MAXIMIZE)
        if warm_start is not None:
            start_lower, (start_S, start_B), warm_start = _direction_witness(residuals, warm_start, h)
            v.Start = warm_start
            b.Start = np.isin(np.arange(K), start_B).astype(float)
            level.Start = min(start_lower, upper_cutoff)
            if zeta is not None:
                zeta.Start = np.isin(np.arange(d), start_S).astype(float)
                if perspective:
                    energy.Start = warm_start ** 2

        stopped = False

        def stop_when_decided(model, where):
            nonlocal stopped
            if where != gp.GRB.Callback.MIP:
                return
            incumbent = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
            bound = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
            margin = 1e-5 * max(1.0, abs(decision_threshold))
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
            raise RuntimeError(f"Separation oracle failed: Gurobi status {model.Status}")
        if model.SolCount == 0:
            upper = max(0.0, float(model.ObjBound))
            if decision_threshold is None or upper > decision_threshold:
                raise RuntimeError("Separation oracle stopped without a certificate or incumbent")
            return 0.0, upper, ((0,), tuple(range(h)))
        S = np.flatnonzero(zeta.X > 0.5) if zeta is not None else np.arange(d)
        direction = np.zeros(d)
        direction[S] = v.X[S]
        direction /= max(1.0, np.linalg.norm(direction))
        lower, pair, _ = _direction_witness(residuals, direction, h)
        upper = max(lower, 0.0, float(model.ObjBound))
        if model.Status == gp.GRB.INTERRUPTED:
            if not (lower > decision_threshold or upper <= decision_threshold):
                raise RuntimeError("Separation oracle stopped without a valid cut or certificate")
        elif upper - lower > tol:
            raise RuntimeError("Separation oracle did not certify the requested absolute gap")
    return lower, upper, pair


def solve_restricted_socp(block_means, support, M, pairs, *, env=None):
    """Solve (17) over pairs, returning (feasible_mu, L, cut_gradient)."""
    d = len(support)
    with gp.Model(env=env) as model:
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.QCPDual = 1
        model.Params.NonConvex = 0
        model.Params.FeasibilityTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        model.Params.BarQCPConvTol = 1e-8
        mu = model.addMVar(d, lb=-gp.GRB.INFINITY)
        r = model.addVar(lb=0)
        upper = model.addConstr(mu <= M * support)
        lower = model.addConstr(-mu <= M * support)
        for S, B in pairs:
            alpha = model.addMVar(len(B), lb=0)
            residual = model.addMVar(len(S), lb=-gp.GRB.INFINITY)
            model.addConstr(alpha.sum() == 1)
            model.addConstr(residual == block_means[np.ix_(B, S)].T @ alpha - mu[list(S)])
            # r >= 0 makes this a convex second-order cone.
            model.addConstr(residual @ residual <= r * r)
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
        candidate = np.clip(mu.X, -M * support, M * support)
        # Gurobi Pi for a <= constraint in a minimization problem is nonpositive.
        gradient = M * (upper.Pi + lower.Pi)
        return candidate, float(model.ObjVal), gradient


def initial_constraint(
    block_means, center, initial_mu, s, sep_tol, *, env=None,
    oracle_strategy="hybrid", oracle_formulation="indicator", perspective=False,
):
    """Initialize a certified upper bound, a valid (S,B) pair, and box bounds.

    Coordinate directions give |mu_j - center_j| <= F(mu) <= F(initial_mu),
    so M_j = |center_j| + upper_F bounds every improving estimate.
    Hybrid mode uses the deterministic bound and a cheap feasible direction;
    exact mode solves (10) to sep_tol.
    """
    if oracle_strategy == "hybrid":
        upper = deterministic_upper_bound(block_means, initial_mu, s)
        _, pair, _ = heuristic_separation(
            block_means, initial_mu, s, decision_threshold=-1, env=env,
        )
    else:
        _, upper, pair = separation_oracle(
            block_means, initial_mu, s, sep_tol, env=env,
            formulation=oracle_formulation, perspective=perspective,
        )
    M = np.abs(center) + upper + 1e-8 * max(1, np.max(np.abs(center)), upper)
    return upper, pair, M


def constraint_generation(
    block_means, support, M, pairs, s, sep_tol, inner_tol, stats, *,
    max_inner_iter=1000, env=None, oracle_strategy="hybrid",
    oracle_formulation="indicator", perspective=False, certify=True, oracle_state=None,
):
    """Alternate (17) and (10) on a fixed support, returning (mu, L, U, gradient).

    Hybrid separation tries feasible directions before an exact MIP. With
    certify=False, return a valid restricted cut after at most three solves
    (or a heuristic stall), with a deterministic objective upper bound. A
    repeated support must use certify=True to prevent indefinite stalling.
    New (S,B) pairs are appended to pairs in place; oracle_calls counts MIPs.
    """
    seen = set(pairs)
    oracle_state = {} if oracle_state is None else oracle_state
    for inner_iteration in range(max_inner_iter):
        candidate, L, gradient = solve_restricted_socp(block_means, support, M, pairs, env=env)
        stats["inner_iterations"] += 1
        if oracle_strategy == "hybrid":
            U = deterministic_upper_bound(block_means, candidate, s)
            if U - L <= inner_tol:
                stats["deterministic_certificates"] += 1
                return candidate, L, U, gradient
            lower, pair, direction = heuristic_separation(
                block_means, candidate, s, decision_threshold=L + inner_tol,
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
            block_means, candidate, s, sep_tol, decision_threshold=L + inner_tol, env=env,
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


def _oracle_options(oracle_strategy, oracle_formulation):
    if oracle_strategy not in ("hybrid", "exact"):
        raise ValueError("oracle_strategy must be 'hybrid' or 'exact'")
    if oracle_formulation not in ("bigm", "indicator"):
        raise ValueError("oracle_formulation must be 'bigm' or 'indicator'")


def _oracle_stats(oracle_strategy):
    return {
        "oracle_calls": int(oracle_strategy == "exact"),
        "heuristic_calls": int(oracle_strategy == "hybrid"),
        "heuristic_cuts": 0, "inner_iterations": 0,
        "deferred_supports": 0, "deterministic_certificates": 0,
    }


def dense_estimation(
    block_means, center, s, tol, *, max_inner_iter=1000, env=None,
    oracle_strategy="hybrid", oracle_formulation="indicator", perspective=False,
):
    """Algorithm 1 without the support cutting plane, returning (mu, info).

    Every coordinate stays active and the search starts at the dense
    coordinate-wise MoM center. Directions are restricted to V_{min(2s,d)}.
    info['objective'] is an upper bound within inner_tol=tol/4 of info['lower_bound'].
    """
    inner_tol, sep_tol = tol / 4, tol / 8
    _oracle_options(oracle_strategy, oracle_formulation)
    stats = _oracle_stats(oracle_strategy)
    _, first_pair, M = initial_constraint(
        block_means, center, center, s, sep_tol, env=env,
        oracle_strategy=oracle_strategy, oracle_formulation=oracle_formulation,
        perspective=perspective,
    )
    pairs = [first_pair]
    estimate, L, U, _ = constraint_generation(
        block_means, np.ones(len(center)), M, pairs, s, sep_tol, inner_tol, stats,
        max_inner_iter=max_inner_iter, env=env,
        oracle_strategy=oracle_strategy, oracle_formulation=oracle_formulation,
        perspective=perspective,
    )
    stats.update({"objective": U, "lower_bound": L, "constraints": len(pairs),
                  "oracle_strategy": oracle_strategy, "oracle_formulation": oracle_formulation,
                  "perspective": perspective})
    return estimate, stats


def ip_estimation_from_block_means(
    block_means, s, *, tol=1e-5, max_iter=1000, max_inner_iter=1000,
    oracle_strategy="hybrid", oracle_formulation="indicator", perspective=False,
):
    """Run Algorithm 1 on an existing odd collection of block means.

    Rows are used as supplied, including repeated rows. Initialization, bounds,
    and every separation call use only these rows. No repartitioning occurs.
    Runtime covers initialization and optimization, excluding block construction.
    """
    start = perf_counter()
    _oracle_options(oracle_strategy, oracle_formulation)
    for name, value in (("max_iter", max_iter), ("max_inner_iter", max_inner_iter)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    block_means = np.asarray(block_means, dtype=float)
    if (block_means.ndim != 2 or min(block_means.shape) == 0
            or not np.isfinite(block_means).all()):
        raise ValueError("block_means must be a finite, nonempty (K, d) array")
    K, d = block_means.shape
    if K % 2 != 1:
        raise ValueError("the number of block means must be odd")
    if isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer)) or not 1 <= s <= d:
        raise ValueError("s must be an integer satisfying 1 <= s <= d")
    tol = 1e-5 if tol is None else tol
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")
    center = np.median(block_means, axis=0)
    support = np.zeros(d)
    support[np.argsort(-np.abs(center))[:s]] = 1
    initial_mu = center * support
    inner_tol, sep_tol = tol / 4, tol / 8
    stats = _oracle_stats(oracle_strategy)

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        initial_upper, first_pair, M = initial_constraint(
            block_means, center, initial_mu, s, sep_tol, env=env,
            oracle_strategy=oracle_strategy, oracle_formulation=oracle_formulation,
            perspective=perspective,
        )
        pairs = [first_pair]
        visited_supports = set()
        oracle_state = {}

        def solve_support(current_support):
            key = tuple(np.flatnonzero(current_support > 0.5))
            certify = oracle_strategy == "exact" or key in visited_supports
            visited_supports.add(key)
            return constraint_generation(
                block_means, current_support, M, pairs, s, sep_tol, inner_tol, stats,
                max_inner_iter=max_inner_iter, env=env,
                oracle_strategy=oracle_strategy, oracle_formulation=oracle_formulation,
                perspective=perspective, certify=certify, oracle_state=oracle_state,
            )

        estimate, info = cutting_plane(
            s, support, initial_mu, initial_upper, solve_support, tol, max_iter=max_iter,
        )

    info.update(stats)
    info.update({
        "K": len(block_means), "constraints": len(pairs),
        "inner_tol": inner_tol, "sep_tol": sep_tol, "runtime": perf_counter() - start,
        "oracle_strategy": oracle_strategy, "oracle_formulation": oracle_formulation,
        "perspective": perspective,
    })
    return estimate, info


def ip_estimation(
    data, s, epsilon, lambda_upper, delta=0.05, *, tol=1e-5, seed=None, C=2,
    max_iter=1000, max_inner_iter=1000, oracle_strategy="hybrid",
    oracle_formulation="indicator", perspective=False,
):
    """Return (mu_hat, info) using Algorithm 1 and the shared outer cutting plane.

    Initialization shares the block rule with brute_force_estimation: C=2 by default.
    tol defaults to 1e-5 (also when None). inner_tol=tol/4 and sep_tol=tol/8.
    Hybrid mode uses partial restricted solves on a support's first visit;
    later visits certify upper_F - L <= inner_tol. Generated (S,B) pairs
    persist across outer iterations. Bounds concern the continuous-direction
    objective, within solver tolerances; info['objective'] is an upper bound.
    Inner/oracle failures raise rather than returning an uncertified result.
    An outer iteration limit instead returns info['converged']=False.
    """
    start = perf_counter()
    _oracle_options(oracle_strategy, oracle_formulation)
    for name, value in (("max_iter", max_iter), ("max_inner_iter", max_inner_iter)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    tol = 1e-5 if tol is None else tol
    block_means, _, _, _, tol = mom_initialization(
        data, s, epsilon, lambda_upper, delta, tol, seed, C=C,
    )
    estimate, info = ip_estimation_from_block_means(
        block_means, s, tol=tol, max_iter=max_iter, max_inner_iter=max_inner_iter,
        oracle_strategy=oracle_strategy, oracle_formulation=oracle_formulation,
        perspective=perspective,
    )
    info.update({"C": C, "runtime": perf_counter() - start})
    return estimate, info
