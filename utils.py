"""Shared building blocks: MoM blocks, hard thresholding, support cutting planes, Gurobi setup."""

from contextlib import contextmanager

import gurobipy as gp
import numpy as np


def odd_ceil(x):
    """Smallest odd integer >= x."""
    k = int(np.ceil(x))
    return k + (k % 2 == 0)


def block_means(data, K, seed=None):
    """Means of K balanced blocks cut from one seeded random permutation of the rows."""
    blocks = np.array_split(np.random.default_rng(seed).permutation(len(data)), K)
    return np.array([data[block].mean(axis=0) for block in blocks])


def complexity(s, d):
    """s*log(e*d/s), the complexity of 2s-sparse directions; it equals d at full support."""
    return s * (1 + np.log(d / s))


def mom_blocks(data, s, epsilon, *, delta=0.05, C=3, seed=None):
    """Block means with K the smallest odd integer >= C * max(s*log(e*d/s), epsilon*n, log(1/delta))."""
    data = check_data(data, s)
    n, d = data.shape
    if not 0 <= epsilon < 0.5 or not 0 < delta < 1 or not 0 < C < np.inf:
        raise ValueError("require 0 <= epsilon < 0.5, 0 < delta < 1 and finite C > 0")
    K = odd_ceil(C * max(complexity(s, d), epsilon * n, np.log(1 / delta)))
    if K > n:
        raise ValueError(f"Required K={K} exceeds n={n}; change the experiment settings.")
    return block_means(data, K, seed)


def check_sparsity(s, d):
    if isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer)) or not 1 <= s <= d:
        raise ValueError("s must be an integer with 1 <= s <= d")


def check_data(data, s):
    """Return data as a float (n, d) array after checking it and the sparsity 1 <= s <= d."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or min(data.shape) == 0 or not np.isfinite(data).all():
        raise ValueError("data must be a finite, nonempty (n, d) array")
    check_sparsity(s, data.shape[1])
    return data


def top_s(x, s):
    """0/1 indicator of the s largest-magnitude entries of x (ties keep the lower index)."""
    support = np.zeros(len(x))
    support[np.argsort(-np.abs(x), kind="stable")[:s]] = 1
    return support


@contextmanager
def gurobi_env():
    """A silent, single-threaded Gurobi environment."""
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", 1)
        env.start()
        yield env


def new_model(env, **params):
    """A silent, single-threaded Gurobi model with the given parameters."""
    model = gp.Model(env=env)
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    for name, value in params.items():
        model.setParam(name, value)
    return model


def cutting_plane(s, support, mu, upper, solve_support, tol, max_iter=1000, rtol=0.0):
    """Minimize a nonnegative objective over ||mu||_0 <= s with outer support cuts; return (mu, info).

    solve_support(z) returns (mu, L, U, g): mu is feasible on the binary support z with
    objective at most U, and L + g @ (z_new - z) lower-bounds the optimum on every support z_new.
    The start (support, mu) must be feasible with objective at most upper.
    Stops when the gap is at most max(tol, rtol * |best upper|); info['objective'] is that upper bound.
    """
    if not 0 < tol < np.inf or max_iter < 1:
        raise ValueError("require a positive finite tol and max_iter >= 1")
    current = np.array(support, dtype=float)
    best_mu, best_upper, best_support = np.array(mu, dtype=float), float(upper), current.copy()
    lower_bound, converged = 0.0, False
    with gurobi_env() as env, new_model(env, MIPGap=0, MIPGapAbs=0, FeasibilityTol=1e-9,
                                         IntFeasTol=1e-9, OptimalityTol=1e-9) as master:
        z = master.addMVar(len(current), vtype=gp.GRB.BINARY)
        eta = master.addVar(lb=0)
        master.addConstr(z.sum() <= s)
        master.setObjective(eta)
        for iteration in range(1, max_iter + 1):
            candidate, L, U, gradient = solve_support(current.copy())
            if U < best_upper:
                best_mu, best_upper, best_support = np.array(candidate, dtype=float), float(U), current.copy()
            master.addConstr(eta >= L + np.asarray(gradient) @ (z - current))
            master.optimize()
            if master.Status != gp.GRB.OPTIMAL:
                raise RuntimeError(f"Support MILP failed: Gurobi status {master.Status}")
            lower_bound = max(lower_bound, master.ObjBound)
            if best_upper - lower_bound <= max(tol, rtol * abs(best_upper)):
                converged = True
                break
            current = np.rint(z.X)
    return best_mu, {
        "converged": converged, "objective": best_upper, "lower_bound": lower_bound,
        "gap": max(0.0, best_upper - lower_bound), "tol": tol, "iterations": iteration,
        "support": best_support.astype(int).tolist(),
    }
