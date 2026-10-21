"""Sparse Depersin-Lecue descent: exact q-sparse PSD directions, capped block weights, hard-thresholded steps.

For block means Z_i and an s-sparse x, E_q(x) = max_M min_w sum_i w_i (Z_i - x)' M (Z_i - x) over trace-one PSD
matrices M supported on at most q = min(2s, d) coordinates and weights w on the simplex capped at 1/((1-alpha)K).
On a fixed support S it is the SDP min_w lambda_max(sum_i w_i r_iS r_iS'). A best-first support branch-and-bound
over these SDPs finds an incumbent M with h_x(M) >= gamma * U for a certified upper bound U >= E_q(x).
sparse_dl_descent_estimation steps x <- H_s(x + m v), v the leading eigenvector of M and m the median of
v'(Z_i - x), and returns the iterate with the smallest U.
"""

import heapq
from time import perf_counter

import clarabel
import numpy as np
from scipy import sparse

from utils import mom_blocks, top_s

MAX_NODES = 100_000  # branch-and-bound nodes per game; reaching it leaves the gamma gap uncertified


def trimmed_average(values, keep):
    """min of w @ values over the simplex capped at 1/keep: the smallest keep values with a fractional endpoint."""
    values = np.sort(values)
    full = int(keep)
    tail = (keep - full) * values[full] if full < len(values) else 0.
    return float((values[:full].sum() + tail) / keep)


def capped_projection(w, cap):
    """A point of {0 <= w <= cap, sum(w) = 1} near w: bisection on the shift, then scaled down to sum one."""
    low, high = w.min() - cap, w.max()  # sum(clip(w - low, 0, cap)) = K cap >= 1
    for _ in range(100):
        middle = (low + high) / 2
        if np.clip(w - middle, 0, cap).sum() >= 1:
            low = middle
        else:
            high = middle
    w = np.clip(w - low, 0, cap)
    return w / w.sum()


def game_sdp(r, keep):
    """Bounds on min_w lambda_max(sum_i w_i r_i r_i') for the rows r_i of r; return (lower, upper, M, w).

    upper is lambda_max at the solver weights made feasible and lower = h(M) at the dual matrix made PSD with
    unit trace, so lower <= value <= upper whatever the solver accuracy.
    """
    K, m = r.shape
    cols, rows = np.tril_indices(m)  # Clarabel's triangle: upper entries (rows <= cols) column by column
    scale = np.where(rows == cols, 1., np.sqrt(2))
    eye = sparse.eye(K)
    linear = sparse.hstack([sparse.vstack([np.ones((1, K)), -eye, eye]), sparse.csc_matrix((2 * K + 1, 1))])
    psd = np.c_[(r[:, rows] * r[:, cols] * scale).T, -1. * (rows == cols)]  # svec(t I - sum_i w_i r_i r_i')
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    solution = clarabel.DefaultSolver(
        sparse.csc_matrix((K + 1, K + 1)), np.r_[np.zeros(K), 1.], sparse.vstack([linear, psd], format="csc"),
        np.r_[1., np.zeros(K), np.full(K, 1 / keep), np.zeros(len(rows))],
        [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(2 * K), clarabel.PSDTriangleConeT(m)], settings).solve()
    if solution.status not in (clarabel.SolverStatus.Solved, clarabel.SolverStatus.AlmostSolved):
        raise RuntimeError(f"Game SDP: {solution.status}")
    w = capped_projection(np.array(solution.x[:K]), 1 / keep)
    upper = float(np.linalg.eigvalsh(r.T @ (w[:, None] * r))[-1])
    M = np.zeros((m, m))
    M[rows, cols] = np.array(solution.z[2 * K + 1:]) / scale
    values, vectors = np.linalg.eigh(M + np.triu(M, 1).T)
    values = np.maximum(values, 0)
    M = (vectors * (values / values.sum())) @ vectors.T
    return trimmed_average(np.einsum("ij,jk,ik->i", r, M, r), keep), upper, M, w


def sparse_bound(C, I, A, q):
    """Upper bound on lambda_max(C[S, S]) over I <= S <= A with |S| = q: interlacing, trace, and Gershgorin."""
    sub = C[np.ix_(A, A)]
    diag = np.diag(sub)
    fixed = np.isin(A, I)
    trace = diag[fixed].sum() + np.sort(diag[~fixed])[::-1][:q - len(I)].sum()
    rows = diag + np.sort(np.abs(sub - np.diag(diag)), axis=1)[:, ::-1][:, :q - 1].sum(axis=1)
    return min(float(np.linalg.eigvalsh(sub)[-1]), float(trace), float(rows.max()))


def support_search(r, q, keep, gamma, hint=None):
    """Best-first branch-and-bound for E_q on the residual rows r; return the incumbent and certified bounds.

    A node (I, A) holds the supports S with I <= S <= A and |S| = q. Its bound is the smaller of the dense game
    SDP on A (solved when the node is first expanded) and sparse_bound at the weights of an ancestor's SDP.
    Each expanded node rounds its dense matrix to I plus the largest free diagonal entries, then includes or
    excludes the free coordinate with the largest one. Nodes with bound <= lower / gamma are discarded; the
    returned upper bound is the largest bound over discarded nodes, solved supports, and the open nodes.
    """
    d = r.shape[1]
    leaves, best, counts = {}, [-np.inf, None, None], {"nodes": 0, "sdps": 0}

    def leaf(S):
        if S not in leaves:
            counts["sdps"] += 1
            lower, upper, M, _ = game_sdp(r[:, S], keep)
            leaves[S] = upper
            if lower > best[0]:
                best[:] = lower, S, M
        return leaves[S]

    if hint is not None:
        leaf(tuple(hint))
    heap, tie, closed, top, root = [(-np.inf, 0, (), tuple(range(d)), None, None)], 1, -np.inf, -np.inf, None
    while heap:
        bound, _, I, A, M, C = heapq.heappop(heap)
        bound = -bound
        if bound <= best[0] / gamma or counts["nodes"] >= MAX_NODES:
            top = bound
            break
        counts["nodes"] += 1
        if q in (len(I), len(A)):
            closed = max(closed, leaf(I if len(I) == q else A))
            continue
        if M is None:
            counts["sdps"] += 1
            _, upper, M, w = game_sdp(r[:, A], keep)
            C, bound = r.T @ (w[:, None] * r), min(bound, upper)
            root = bound if root is None else root
        free = [j for j in A if j not in I]
        order = np.argsort(-np.diag(M)[[A.index(j) for j in free]], kind="stable")
        leaf(tuple(sorted(I + tuple(free[k] for k in order[:q - len(I)]))))
        if bound <= best[0] / gamma:
            closed = max(closed, bound)
            continue
        j = free[order[0]]
        for child_I, child_A, child_M in ((tuple(sorted(I + (j,))), A, M), (I, tuple(k for k in A if k != j), None)):
            child = min(bound, sparse_bound(C, child_I, child_A, q))
            if child <= best[0] / gamma:
                closed = max(closed, child)
            else:
                heapq.heappush(heap, (-child, tie, child_I, child_A, child_M, C))
                tie += 1
    upper = max(best[0], closed, top)
    return {"support": best[1], "matrix": best[2], "lower": best[0], "upper": upper, "root_bound": root,
            "certified": best[0] >= gamma * upper, **counts}


def sparse_dl_descent_estimation(data, s, epsilon, *, alpha=0.35, gamma=0.95, delta=0.05, C=3, seed=None,
                                 max_iter=30, patience=3):
    """Return (mu, info): the iterate with the smallest certified game bound U, from coordinate MoM + HT.

    alpha must exceed the fraction of blocks that contain outliers. The defaults (alpha = 0.35 with the shared
    block rule C = 3) were the most accurate in Experiments 1 and 2; they are empirical, since the contraction
    argument needs alpha < 0.276 and, for the patience rule, a much smaller alpha.
    Stops after max_iter iterations, after patience iterations without a smaller U, or at a fixed point.
    info['history'] has, per iteration, U, L = h_x(M), the root (dense) bound, node and SDP counts, and time.
    """
    if not 0 <= alpha < 0.5 or not 0 < gamma <= 1:
        raise ValueError("require 0 <= alpha < 0.5 and 0 < gamma <= 1")
    start = perf_counter()
    blocks = mom_blocks(data, s, epsilon, delta=delta, C=C, seed=seed)
    K, d = blocks.shape
    q, keep = min(2 * s, d), (1 - alpha) * K
    center = np.median(blocks, axis=0)
    x = center * top_s(center, s)
    history, best, support, stale, termination = [], None, None, 0, "max_iter"
    for _ in range(max_iter):
        tick = perf_counter()
        residuals = blocks - x
        game = support_search(residuals, q, keep, gamma, hint=support)
        support = game["support"]
        history.append({key: game[key] for key in ("upper", "lower", "root_bound", "nodes", "sdps", "certified")}
                       | {"support": list(support), "time": perf_counter() - tick})
        if best is None or game["upper"] < best["upper"]:
            best, stale = {"x": x, "upper": game["upper"], "lower": game["lower"]}, 0
        else:
            stale += 1
            if stale >= patience:
                termination = "no_improvement"
                break
        v = np.zeros(d)
        v[list(support)] = np.linalg.eigh(game["matrix"])[1][:, -1]
        step = x + np.median(residuals @ v) * v
        step = step * top_s(step, s)
        if np.array_equal(step, x):
            termination = "fixed_point"
            break
        x = step
    return best["x"], {
        "converged": all(h["certified"] for h in history), "termination": termination, "objective": best["upper"],
        "lower_bound": best["lower"], "iterations": len(history), "K": K, "alpha": alpha,
        "support": (best["x"] != 0).astype(int).tolist(), "nodes": sum(h["nodes"] for h in history),
        "sdps": sum(h["sdps"] for h in history), "history": history, "runtime": perf_counter() - start}

