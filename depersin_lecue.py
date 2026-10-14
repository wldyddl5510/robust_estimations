"""Depersin-Lecue covSDPofMeans (arXiv:1906.03058, Algorithm 4) with a CVXPY/Clarabel covering SDP.

K is supplied explicitly; a practical K does not confer the paper's theorem. Balanced blocks keep
all observations (the paper discards remainders). There is no hard thresholding.
"""

import math
from time import perf_counter

import cvxpy as cp
import numpy as np

from utils import block_means

SDP_ETA = 1e-4  # SolveSDP's acceptance interval [0.9981, 1] is derived for eta = 1e-4


class CoveringSDP:
    """min tr(M) + sum(y) s.t. rho * z_i' M z_i + 0.9K * y_i >= 1, M PSD, y >= 0, for a parametric rho.

    Each solve repairs the primal into a feasible covering (upper bound) and rescales the duals into
    a feasible packing (lower bound).
    """

    def __init__(self, z, tol):
        self.z = z
        self.c = .9 * len(z)
        self.rho = cp.Parameter(nonneg=True)
        self.m = cp.Variable((z.shape[1], z.shape[1]), PSD=True)
        self.y = cp.Variable(len(z), nonneg=True)
        values = cp.hstack([cp.sum(cp.multiply(np.outer(row, row), self.m)) for row in z])
        self.constraint = self.rho * values + self.c * self.y >= 1
        self.problem = cp.Problem(cp.Minimize(cp.trace(self.m) + cp.sum(self.y)), [self.constraint])
        self.tol = min(tol / 100, 1e-8)

    def solve(self, rho):
        self.rho.value = rho
        self.problem.solve(solver="CLARABEL", tol_gap_abs=self.tol, tol_gap_rel=self.tol,
                           tol_feas=self.tol, max_iter=500, max_threads=1)
        if self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError("Covering SDP: " + self.problem.status)
        w, v = np.linalg.eigh((self.m.value + self.m.value.T) / 2)
        m = (v * np.maximum(w, 0)) @ v.T
        y = np.maximum(self.y.value, 0)
        value = rho * np.einsum("ij,jk,ik->i", self.z, m, self.z) + self.c * y
        correction = max(1., 1 / max(float(value.min()), 1e-300)) * (1 + 1e-12)
        m *= correction
        y *= correction
        packing = np.maximum(self.constraint.dual_value, 0)
        norm = max(float(np.linalg.eigvalsh(rho * (self.z.T @ (packing[:, None] * self.z)))[-1]),
                   float(np.max(self.c * packing)), 1.)
        packing /= norm * (1 + 1e-12)
        lower, upper = float(packing.sum()), float(np.trace(m) + y.sum())
        return {"matrix": m, "lower": lower, "upper": upper, "converged": upper <= (1 + SDP_ETA) * lower,
                "iterations": self.problem.solver_stats.num_iters}


def depersin_lecue_estimation(data, *, K, seed=None, tol=1e-5):
    """Return (mu, info); tol sets the SDP solver tolerances, not the paper's fixed eta.

    The outer iteration count uses the conservative bound from Algorithm 1's square-root
    denominator, and each step bisects rho with ceil(log2(1e9 d^4)) steps. Without an accepted
    rho, the paper's initial/current-center rule ends the search; that return is not a
    statistical certificate when the theorem's assumptions fail.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or min(data.shape) < 1 or not np.isfinite(data).all():
        raise ValueError("data must be a finite nonempty (n, d) array")
    if isinstance(K, (bool, np.bool_)) or not isinstance(K, (int, np.integer)) or not 1 <= K <= len(data):
        raise ValueError("K must be an integer in [1, n]")
    if not 0 < tol < np.inf:
        raise ValueError("tol must be positive and finite")
    start = perf_counter()
    blocks = block_means(data, K, seed)
    d = blocks.shape[1]
    initial = np.median(blocks, axis=0)
    x = initial.copy()
    spread = float(np.median(np.linalg.norm(blocks - initial, axis=1)))
    stats = {"sdp_calls": 0, "sdp_iterations": 0}
    if spread == 0:
        return initial, {"converged": True, "termination": "zero_median_radius", "K": K, "iterations": 0,
                         **stats, "runtime": perf_counter() - start}
    max_outer = math.ceil(math.log(8 * math.sqrt(d)) / math.log(1 / math.sqrt(.81)))
    max_bisect = math.ceil(math.log2(1e9 * d**4))
    converged, termination = True, "outer_iteration_bound"
    for outer in range(max_outer):
        z = blocks - x
        sdp = CoveringSDP(z, tol)

        def solve(rho):
            result = sdp.solve(rho)
            stats["sdp_calls"] += 1
            stats["sdp_iterations"] += result["iterations"]
            return result

        low, high = 0., d / spread**2
        result = solve(high)
        if not result["converged"]:
            converged, termination = False, "sdp_gap"
            break
        accepted = .9981 <= result["upper"] <= 1.
        for _ in range(max_bisect):
            if accepted:
                break
            middle = (low + high) / 2
            result = solve(middle)
            if not result["converged"]:
                break
            if .9981 <= result["upper"] <= 1.:
                accepted = True
            elif result["upper"] < .9981:
                high = middle
            else:
                low = middle
        if not result["converged"]:
            converged, termination = False, "sdp_gap"
            break
        if not accepted:
            result = solve(1 / (d * spread)**2)
            if not result["converged"]:
                converged, termination = False, "sdp_gap"
            elif result["upper"] < 1:
                x, termination = initial.copy(), "initial_center_certificate"
            else:
                termination = "current_center_certificate"
            break
        if np.trace(result["matrix"]) <= 0:
            converged, termination = False, "zero_sdp_matrix"
            break
        direction = np.linalg.eigh(result["matrix"])[1][:, -1]
        x = x + float(np.median(z @ direction)) * direction
    return x, {"converged": converged, "termination": termination, "K": K, "iterations": outer + 1,
               **stats, "runtime": perf_counter() - start}
