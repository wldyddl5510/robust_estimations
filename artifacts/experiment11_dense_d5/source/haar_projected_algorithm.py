"""Independent Haar projections, dense means, and globally sparse aggregation."""

from time import perf_counter

import gurobipy as gp
import numpy as np

from haar_aggregation import aggregate_haar_estimates
from IP_algorithm import dense_estimation, deterministic_upper_bound
from utils import mom_initialization


def random_haar_projections(d, r, J, rng):
    """Return independent (J, r, d) orthonormal-row Haar projection matrices.

    Gaussian thin QR, with positive diagonal in R, samples a uniform frame;
    its row space is uniform on Gr(r, d). No sqrt(d/r) rescaling is applied.
    """
    for name, value in (("d", d), ("r", r), ("J", J)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if r > d:
        raise ValueError("r must not exceed d")
    projections = np.empty((J, r, d))
    for j in range(J):
        Q, R = np.linalg.qr(rng.normal(size=(d, r)), mode="reduced")
        Q *= np.where(np.diag(R) < 0, -1.0, 1.0)
        projections[j] = Q.T
    return projections


def haar_projected_estimation(
    data, s, epsilon, lambda_upper, delta=0.05, *, r, J, tol=1e-5, seed=None, C=2,
    objective="least_squares", aggregation_tol=1e-5, aggregation_rtol=1e-5,
    max_iter=1000, max_inner_iter=1000, oracle_strategy="hybrid",
    oracle_formulation="indicator", perspective=False,
):
    """Estimate a sparse mean using J independent Haar projections of rank r.

    Each dense problem uses the full sphere V_r and the odd block count
    ceil(C * max(r, epsilon*n, log(J/delta))). Projection and block randomness
    use independent child streams; all projections share the block partition.
    Orthonormal rows preserve the validity of the input covariance upper bound.

    Only the final aggregation imposes ||mu||_0 <= s. Its domain also includes
    |mu_i| <= M_i, with M_i = |original coordinate MoM_i| + U_det + margin.
    U_det is the original-data median top-2s residual norm at the initial HT
    MoM. This data-driven box defines the estimator; it is not a certificate
    that every unboxed aggregation optimum or the population mean lies inside.

    J is independent of d/r; rank-deficient stacked projections are allowed.
    The reported gap certifies the boxed sparse aggregation, not statistical
    error or identifiability. Runtime includes projections and box construction.
    """
    start = perf_counter()
    if objective not in ("least_squares", "max"):
        raise ValueError("objective must be 'least_squares' or 'max'")
    if not np.isfinite(aggregation_tol) or aggregation_tol <= 0:
        raise ValueError("aggregation_tol must be positive and finite")
    if not np.isfinite(aggregation_rtol) or aggregation_rtol < 0:
        raise ValueError("aggregation_rtol must be nonnegative and finite")
    for name, value in (("max_iter", max_iter), ("max_inner_iter", max_inner_iter)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or min(data.shape) == 0:
        raise ValueError("data must be a nonempty (n, d) array")
    tol = 1e-5 if tol is None else tol
    projection_seed, block_seed = np.random.SeedSequence(seed).spawn(2)
    projections = random_haar_projections(
        data.shape[1], r, J, np.random.default_rng(projection_seed),
    )
    original_blocks, center, _, initial_mu, _ = mom_initialization(
        data, s, epsilon, lambda_upper, delta, tol, seed, C=C,
    )
    box_upper = deterministic_upper_bound(original_blocks, initial_mu, s)
    M = np.abs(center) + box_upper
    M += 1e-8 * max(1.0, float(M.max()))
    if not np.isfinite(M).all():
        raise ValueError("data-driven aggregation bounds must be finite")
    estimates = np.empty((J, r))
    stats = {"oracle_calls": 0, "inner_iterations": 0, "constraints": 0,
             "heuristic_calls": 0, "heuristic_cuts": 0, "deferred_supports": 0,
             "deterministic_certificates": 0}
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        for j, P in enumerate(projections):
            blocks, center, _, _, projected_tol = mom_initialization(
                data @ P.T, r, epsilon, lambda_upper, delta / J, tol,
                block_seed, C=C, complexity=r,
            )
            estimates[j], info = dense_estimation(
                blocks, center, r, projected_tol, max_inner_iter=max_inner_iter,
                env=env, oracle_strategy=oracle_strategy,
                oracle_formulation=oracle_formulation, perspective=perspective,
            )
            for key in stats:
                stats[key] += info[key]
        estimate, aggregation_info = aggregate_haar_estimates(
            projections, estimates, s, M, objective=objective, tol=aggregation_tol,
            rtol=aggregation_rtol, max_iter=max_iter, env=env,
        )
    stats.update(aggregation_info)
    stats.update({
        "projection_method": "haar", "C": C, "K": len(blocks), "r": r, "J": J,
        "direction_sparsity": r, "projection_delta": delta / J,
        "box_K": len(original_blocks), "box_upper": box_upper, "M": M.tolist(),
        "projection_tol": projected_tol, "aggregation_tol": aggregation_tol,
        "aggregation_rtol": aggregation_rtol, "aggregation": "cutting_plane",
        "objective_type": objective, "oracle_strategy": oracle_strategy,
        "oracle_formulation": oracle_formulation, "perspective": perspective,
        "runtime": perf_counter() - start,
    })
    return estimate, stats
