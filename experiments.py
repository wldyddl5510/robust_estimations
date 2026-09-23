"""Compare three sparse mean estimators on one contaminated t sample."""

import argparse
from time import perf_counter

import numpy as np

from brute_force import brute_force_estimation
from IP_algorithm import ip_estimation
from utils import (
    adversarial_sparse_contamination,
    mom_initialization,
    sparse_t_dist_data_generation,
)


def coordinate_mom_estimation(data, s, epsilon, lambda_upper, delta=0.05, *, seed=None):
    """Coordinate-wise MoM followed by hard thresholding to the largest s entries."""
    start = perf_counter()
    _, _, _, estimate, _ = mom_initialization(
        data, s, epsilon, lambda_upper, delta, seed=seed,
    )
    return estimate, {"converged": True, "runtime": perf_counter() - start}


def run_experiment(n, epsilon, nu, s, delta, *, d, scale=1, seed=42):
    """Return per-method error, support recovery, and runtime in seconds.

    The t shape matrix is scale * I_d. Draw one loc ~ Uniform(1, 3) per run
    and put it on s randomly chosen mean coordinates; the others are zero.
    All methods use the same contaminated data, K rule, and seeded block split.
    Runtime includes estimator preprocessing, but excludes shared data generation
    and metric calculation. Support recovery uses an absolute threshold of 1e-8.
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
    data = adversarial_sparse_contamination(data, epsilon, s)
    # Sigma = nu / (nu - 2) * scale * I_d; scale itself is not the covariance.
    lambda_upper = 2 * nu / (nu - 2) * scale

    results = {}
    for name, estimator in (
        ("brute_force", brute_force_estimation),
        ("algorithm_1", ip_estimation),
        ("coordinate_mom", coordinate_mom_estimation),
    ):
        estimate, info = estimator(data, s, epsilon, lambda_upper, delta, seed=seed)
        if not info["converged"]:
            raise RuntimeError(f"{name} did not converge")
        results[name] = {
            "error": float(np.linalg.norm(estimate - truth)),
            "support_recovery": float(np.sum((np.abs(estimate) > 1e-8) & (truth != 0)) / s),
            "runtime": info["runtime"],
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", "-n", type=int, required=True, help="sample size")
    parser.add_argument("--epsilon", type=float, required=True, help="contamination rate")
    parser.add_argument("--nu", type=float, required=True, help="t degrees of freedom (> 2)")
    parser.add_argument("--s", "-s", type=int, required=True, help="mean sparsity")
    parser.add_argument("--delta", type=float, required=True, help="failure probability")
    parser.add_argument("--d", "--dim", "-d", type=int, required=True, help="dimension")
    parser.add_argument("--scale", type=float, default=1, help="t shape = scale * I_d (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="data and block seed (default: 42)")
    args = parser.parse_args()
    try:
        results = run_experiment(**vars(args))
    except ValueError as error:
        parser.error(str(error))
    print(f"{'method':<18} {'l2_error':>12} {'support_recovery':>18} {'runtime_s':>12}")
    for name, metrics in results.items():
        print(f"{name:<18} {metrics['error']:>12.6f} "
              f"{metrics['support_recovery']:>18.6f} {metrics['runtime']:>12.6f}")


if __name__ == "__main__":
    main()
