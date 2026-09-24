"""Compare sparse mean estimators and the plain sample mean on one t sample."""

import argparse
from time import perf_counter

import numpy as np

from brute_force import NetSizeError, brute_force_estimation
from IP_algorithm import ip_estimation
from utils import (
    adversarial_sparse_contamination,
    mom_initialization,
    sparse_t_dist_data_generation,
)


def coordinate_mom_estimation(data, s, epsilon, lambda_upper, delta=0.05, *, tol=None, seed=None, C=2):
    """Coordinate-wise MoM followed by hard thresholding to the largest s entries."""
    start = perf_counter()
    block_means, _, _, estimate, _ = mom_initialization(
        data, s, epsilon, lambda_upper, delta, tol=tol, seed=seed, C=C,
    )
    return estimate, {"converged": True, "C": C, "K": len(block_means), "runtime": perf_counter() - start}


def plain_sample_mean_estimation(data):
    """Return the coordinate-wise average of every observed row."""
    start = perf_counter()
    estimate = np.mean(data, axis=0)
    return estimate, {"converged": True, "runtime": perf_counter() - start}


def run_experiment(
    n, epsilon, nu, s, delta=0.05, *, d, scale=1, seed=42, tol=None, C=2,
    strength=30, net_radius=0.25, max_net_points=200_000,
):
    """Return per-method error, support recovery, and runtime in seconds.

    The t shape matrix is scale * I_d. Draw one loc ~ Uniform(1, 3) per run
    and put it on s randomly chosen mean coordinates; the others are zero.
    All methods use the same contaminated data. Sparse methods share C and the block seed.
    tol overrides the optimization tolerance; the MoM estimate does not depend on it.
    net_radius controls only the brute-force covering net (default: 1/4).
    max_net_points limits the full net before allocation (default: 200,000).
    Oversized nets are recorded as skipped; other solver failures still raise.
    Runtime includes estimator preprocessing, but excludes shared data generation
    and metric calculation. Support recovery uses an absolute threshold of 1e-8;
    it is not reported for the dense plain sample mean.
    """
    if n < 1 or not 1 <= s <= d:
        raise ValueError("require n >= 1 and 1 <= s <= d")
    if not np.isfinite(nu) or nu <= 2:
        raise ValueError("nu must be finite and greater than 2 for a finite covariance")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite scalar")
    if not 0 <= epsilon < 0.5 or not 0 < delta < 1:
        raise ValueError("require 0 <= epsilon < 0.5 and 0 < delta < 1")

    np.random.seed(seed)
    loc = np.random.uniform(1, 3)
    data, truth = sparse_t_dist_data_generation(n, loc, nu, d, s, scale=scale)
    data = adversarial_sparse_contamination(data, epsilon, s, strength=strength)
    # Sigma = nu / (nu - 2) * scale * I_d; scale itself is not the covariance.
    lambda_upper = 2 * nu / (nu - 2) * scale

    results = {}
    for name, estimator in (
        ("brute_force", brute_force_estimation),
        ("algorithm_1", ip_estimation),
        ("coordinate_mom", coordinate_mom_estimation),
        ("sample_mean", plain_sample_mean_estimation),
    ):
        options = ({"net_radius": net_radius, "max_net_points": max_net_points}
                   if name == "brute_force" else {})
        try:
            if name == "sample_mean":
                estimate, info = estimator(data)
            else:
                estimate, info = estimator(
                    data, s, epsilon, lambda_upper, delta, tol=tol, seed=seed, C=C, **options,
                )
        except NetSizeError as error:
            results[name] = {
                "error": None, "support_recovery": None, "runtime": None,
                "status": "skipped", "reason": str(error), "net_size": error.size,
            }
            continue
        if not info["converged"]:
            raise RuntimeError(f"{name} did not converge")
        results[name] = {
            "error": float(np.linalg.norm(estimate - truth)),
            "support_recovery": (None if name == "sample_mean" else
                                 float(np.sum((np.abs(estimate) > 1e-8) & (truth != 0)) / s)),
            "runtime": info["runtime"],
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", "-n", type=int, required=True, help="sample size")
    parser.add_argument("--epsilon", type=float, required=True, help="contamination rate")
    parser.add_argument("--nu", type=float, required=True, help="t degrees of freedom (> 2)")
    parser.add_argument("--s", "-s", type=int, required=True, help="mean sparsity")
    parser.add_argument("--delta", type=float, default=0.05, help="failure probability (default: 0.05)")
    parser.add_argument("--d", "--dim", "-d", type=int, required=True, help="dimension")
    parser.add_argument("--scale", type=float, default=1, help="t shape = scale * I_d (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="data and block seed (default: 42)")
    parser.add_argument("--tol", type=float, help="optimization tolerance (default: sqrt(K*lambda_upper/n)/4)")
    parser.add_argument("--C", type=float, default=2, help="block-count multiplier (default: 2)")
    parser.add_argument("--strength", type=float, default=30, help="adversarial contamination strength (default: 30)")
    parser.add_argument("--net-radius", type=float, default=0.25, help="brute-force covering radius in (0, 1] (default: 0.25)")
    parser.add_argument("--max-net-points", type=int, default=200_000, help="brute-force net size limit (default: 200,000)")
    args = parser.parse_args()
    try:
        results = run_experiment(**vars(args))
    except ValueError as error:
        parser.error(str(error))
    print(f"{'method':<18} {'l2_error':>12} {'support_recovery':>18} {'runtime_s':>12}")
    for name, metrics in results.items():
        if metrics.get("status") == "skipped":
            print(f"{name:<18} {'—':>12} {'—':>18} {'—':>12}  skipped: {metrics['reason']}")
            continue
        recovery = ("—" if metrics["support_recovery"] is None else
                    f"{metrics['support_recovery']:.6f}")
        print(f"{name:<18} {metrics['error']:>12.6f} "
              f"{recovery:>18} {metrics['runtime']:>12.6f}")


if __name__ == "__main__":
    main()
