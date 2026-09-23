"""Algorithm 1: SOCP constraint generation with an integer separation oracle."""

from time import perf_counter

import gurobipy as gp
import numpy as np

from utils import cutting_plane, mom_initialization


def separation_oracle(block_means, mu, s, tol, *, env=None):
    """Solve (10), returning (lower_F, upper_F, (S, B)).

    The incumbent direction gives lower_F; Gurobi's global maximization bound
    gives upper_F. Symmetry in v makes maximizing the signed median equivalent
    to maximizing its absolute value. tol is an absolute objective gap.
    """
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
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")

    residuals = block_means - mu
    k, h = min(2 * s, d), (K + 1) // 2
    # Each projection is bounded by the norm of the row's largest k entries.
    row_bounds = np.linalg.norm(np.sort(np.abs(residuals), axis=1)[:, -k:], axis=1)
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
        zeta = model.addMVar(d, vtype=gp.GRB.BINARY)
        b = model.addMVar(K, vtype=gp.GRB.BINARY)
        t = model.addVar(lb=0, ub=float(np.median(row_bounds)))
        model.addConstr(v @ v <= 1)
        model.addConstr(v <= zeta)
        model.addConstr(-v <= zeta)
        model.addConstr(zeta.sum() <= k)
        model.addConstr(b.sum() == h)
        for i in range(K):
            model.addConstr((b[i] == 1) >> (residuals[i] @ v >= t))
        model.setObjective(t, gp.GRB.MAXIMIZE)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL or model.SolCount == 0:
            raise RuntimeError(f"Separation oracle failed: Gurobi status {model.Status}")

        S = np.flatnonzero(zeta.X > 0.5)
        direction = np.zeros(d)
        direction[S] = v.X[S]
        direction /= max(1.0, np.linalg.norm(direction))
        projections = residuals @ direction
        if np.median(projections) < 0:
            projections = -projections
        lower = float(np.median(projections))
        upper = max(lower, 0.0, float(model.ObjBound))
        if upper - lower > tol:
            raise RuntimeError("Separation oracle did not certify the requested absolute gap")
        # Any nonempty support works when the incumbent direction is zero.
        S = tuple(map(int, S)) if len(S) else (0,)
        B = tuple(sorted(map(int, np.argsort(projections)[-h:])))
    return lower, upper, (S, B)


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
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"Restricted SOCP failed: Gurobi status {model.Status}")
        candidate = np.clip(mu.X, -M * support, M * support)
        # Gurobi Pi for a <= constraint in a minimization problem is nonpositive.
        gradient = M * (upper.Pi + lower.Pi)
        return candidate, float(model.ObjVal), gradient


def ip_estimation(
    data, s, epsilon, lambda_upper, delta=0.05, *, tol=None, seed=None, C=1,
    max_iter=1000, max_inner_iter=1000,
):
    """Return (mu_hat, info) using Algorithm 1 and the shared outer cutting plane.

    Initialization matches brute_force_estimation: C=1 by default, the same block seed,
    and tol=sqrt(K*lambda_upper/n)/100 by default. inner_tol=tol/4 and sep_tol=tol/8.
    The inner loop requires upper_F - L <= inner_tol. Generated (S,B) pairs
    persist across outer iterations. Bounds concern the continuous-direction
    objective, within solver tolerances; info['objective'] is an upper bound.
    Inner/oracle failures raise rather than returning an uncertified result.
    An outer iteration limit instead returns info['converged']=False.
    """
    start = perf_counter()
    for name, value in (("max_iter", max_iter), ("max_inner_iter", max_inner_iter)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    block_means, center, support, initial_mu, tol = mom_initialization(
        data, s, epsilon, lambda_upper, delta, tol, seed, C=C,
    )
    inner_tol, sep_tol = tol / 4, tol / 8
    stats = {"oracle_calls": 1, "inner_iterations": 0}

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        _, initial_upper, first_pair = separation_oracle(
            block_means, initial_mu, s, sep_tol, env=env,
        )
        M = np.abs(center) + initial_upper + 1e-8 * max(1, np.max(np.abs(center)), initial_upper)
        pairs = [first_pair]
        seen = {first_pair}

        def solve_support(current_support):
            for _ in range(max_inner_iter):
                candidate, L, gradient = solve_restricted_socp(
                    block_means, current_support, M, pairs, env=env,
                )
                stats["inner_iterations"] += 1
                _, U, pair = separation_oracle(block_means, candidate, s, sep_tol, env=env)
                stats["oracle_calls"] += 1
                if U - L <= inner_tol:
                    return candidate, L, U, gradient
                if pair in seen:
                    raise RuntimeError("An existing (S,B) constraint is still violated; check solver tolerances")
                pairs.append(pair)
                seen.add(pair)
            raise RuntimeError("Inner constraint generation reached max_inner_iter before certification")

        estimate, info = cutting_plane(
            s, support, initial_mu, initial_upper, solve_support, tol, max_iter=max_iter,
        )

    info.update(stats)
    info.update({
        "C": C, "K": len(block_means), "constraints": len(pairs),
        "inner_tol": inner_tol, "sep_tol": sep_tol, "runtime": perf_counter() - start,
    })
    return estimate, info
