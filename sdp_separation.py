"""One-sided SDP certificate for Algorithm 1 at decision_threshold = L + inner_tol.

The maximization objective counts relaxed blocks, not F(mu). Only a corrected dual upper
bound below h = (K+1)/2 certifies F(mu) < decision_threshold. All other outcomes require the
existing exact separation oracle. Bounds include floating-point safeguards, as do the other
numerical certificates in this repository; they are not interval-arithmetic proofs.
"""

import warnings

import cvxpy as cp
import numpy as np


def _dual_upper_bound(residuals, threshold, sparsity, duals):
    """Bound the SDP maximum using any finite duals, without assuming stationarity.

    Constraint order: global PSD, trace, entrywise Q norm, w norm, p >= 0, p <= 1,
    local PSDs, c >= threshold*p. Nonnegative multipliers are clipped; negative PSD
    eigenvalues are charged against bounds on the primal matrix traces. The Lagrangian's
    remaining coefficients are bounded using ||w||2 <= 1, ||Q||F <= tr(Q) <= 1,
    0 <= p <= 1, and |c_k| <= ||a_k||2. Norm penalties absorb coefficients entrywise.
    Thus even inaccurate duals give an upper bound, though possibly a uselessly large one.
    """
    K, d = residuals.shape
    if len(duals) != K + 7 or any(value is None for value in duals):
        return np.inf
    values = [np.asarray(value, dtype=float) for value in duals]
    if not all(np.isfinite(value).all() for value in values):
        return np.inf
    global_psd = (values[0] + values[0].T) / 2
    local_psd = np.array([(value + value.T) / 2 for value in values[6:-1]])
    lam, gamma, eta = (max(0., float(value)) for value in values[1:4])
    alpha, beta, delta = (np.maximum(value, 0.) for value in (values[4], values[5], values[-1]))
    row_norms = np.linalg.norm(residuals, axis=1)

    constant = global_psd[0, 0] + local_psd[:, 0, 0].sum() + lam + eta * np.sqrt(sparsity) + beta.sum()
    w_coefficient = 2 * (global_psd[0, 1:] + residuals.T @ local_psd[:, 0, 2])
    Q_coefficient = (global_psd[1:, 1:] + residuals.T @ (local_psd[:, 2, 2, None] * residuals)
                     + (sparsity * gamma - lam) * np.eye(d))
    p_coefficient = 1 + alpha - beta + 2 * local_psd[:, 0, 1] + local_psd[:, 1, 1] - threshold * delta
    c_coefficient = 2 * local_psd[:, 1, 2] + delta
    correction = (np.linalg.norm(np.maximum(np.abs(w_coefficient) - eta, 0.))
                  + np.linalg.norm(Q_coefficient - np.clip(Q_coefficient, -gamma, gamma), ord="fro")
                  + np.maximum(p_coefficient, 0.).sum() + np.abs(c_coefficient) @ row_norms)

    # If Z >= -epsilon I, <Z, M> >= -epsilon tr(M) for every M >= 0.
    # Here tr(M_global) <= 2 and tr(M_k) <= 2 + ||a_k||2**2.
    eps = np.finfo(float).eps
    global_error = (max(0., -np.linalg.eigvalsh(global_psd)[0])
                    + 64 * eps * (d + 1) * max(1., np.linalg.norm(global_psd, ord="fro")))
    local_error = (np.maximum(0., -np.linalg.eigvalsh(local_psd)[:, 0])
                   + 192 * eps * np.maximum(1., np.linalg.norm(local_psd, axis=(1, 2))))
    correction += 2 * global_error + local_error @ (2 + row_norms**2)
    # Account for cancellation in coefficients, not just the size of the resulting bound.
    magnitude = (1 + K + 2 * np.abs(global_psd).sum()
                 + np.abs(local_psd).sum(axis=(1, 2)) @ (2 + row_norms**2)
                 + lam * (1 + d) + sparsity * gamma * d + eta * (1 + np.sqrt(sparsity))
                 + alpha.sum() + beta.sum() + (1 + abs(threshold)) * delta.sum())
    padding = 128 * eps * (K + d + 1) * magnitude + 1e-9 * max(1, K)
    upper = constant + correction + padding
    return float(np.nextafter(upper, np.inf)) if np.isfinite(upper) else np.inf


class SDPSeparationCertifier:
    """Reusable CVXPY/Clarabel model of the block-count relaxation.

    time_limit bounds the native SDP solve, not Python model construction/canonicalization.
    A timeout, missing duals or solver failure returns infinity and cannot certify anything.
    Neither the primal objective nor the relaxed direction is used for a cut or an F bound.
    """

    def __init__(self, block_means, s, *, time_limit=0.25, max_iter=60):
        self.block_means = np.asarray(block_means, dtype=float)
        K, d = self.block_means.shape
        self.sparsity = 2 * s
        self.time_limit, self.max_iter = time_limit, max_iter
        self.status = "not_run"
        self.residuals = cp.Parameter((K, d))
        self.products = cp.Parameter((K, d * d))
        self.threshold = cp.Parameter(nonneg=True)
        w, Q = cp.Variable(d), cp.Variable((d, d), symmetric=True)
        p, c = cp.Variable(K), cp.Variable(K)
        column = cp.reshape(w, (d, 1), order="C")
        projections = self.residuals @ w
        quadratic = self.products @ cp.vec(Q, order="C")
        self.constraints = [
            cp.bmat([[np.ones((1, 1)), column.T], [column, Q]]) >> 0,
            cp.trace(Q) <= 1,
            cp.sum(cp.abs(Q)) <= self.sparsity * cp.trace(Q),
            cp.norm1(w) <= np.sqrt(self.sparsity),
            p >= 0,
            p <= 1,
        ]
        for k in range(K):
            self.constraints.append(cp.bmat([
                [1, p[k], projections[k]],
                [p[k], p[k], c[k]],
                [projections[k], c[k], quadratic[k]],
            ]) >> 0)
        self.constraints.append(c >= self.threshold * p)
        self.problem = cp.Problem(cp.Maximize(cp.sum(p)), self.constraints)

    def upper_bound(self, mu, decision_threshold):
        """Return an upper bound on the relaxed block count, or infinity if inconclusive."""
        self.status = "not_run"
        if not np.isfinite(decision_threshold) or decision_threshold <= 0:
            return np.inf  # At a nonpositive threshold, w=Q=c=0, p=1 is feasible.
        try:
            residuals = self.block_means - np.asarray(mu, dtype=float)
            if not np.isfinite(residuals).all():
                return np.inf
            # A power-of-two scale preserves the relaxation by congruence and avoids large
            # a_k a_k' coefficients. Scale c and the threshold by the same factor.
            magnitude = max(float(np.max(np.abs(residuals))), decision_threshold)
            scale = np.ldexp(1., np.frexp(magnitude)[1] - 1)
            residuals = residuals / scale
            threshold = decision_threshold / scale
            self.residuals.value = residuals
            self.products.value = np.einsum("ki,kj->kij", residuals, residuals).reshape(len(residuals), -1)
            self.threshold.value = threshold
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Solution may be inaccurate.*", category=UserWarning)
                self.problem.solve(solver="CLARABEL", warm_start=True, max_threads=1,
                                   time_limit=self.time_limit, max_iter=self.max_iter,
                                   tol_gap_abs=1e-7, tol_gap_rel=1e-7, tol_feas=1e-8)
            self.status = self.problem.status
            if self.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return np.inf
            return _dual_upper_bound(residuals, threshold, self.sparsity,
                                     [constraint.dual_value for constraint in self.constraints])
        except (cp.error.SolverError, ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError):
            self.status = "failed"
            return np.inf
