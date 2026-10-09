"""Compare sparse mean estimators and the plain sample mean on one t sample."""

import argparse
from time import perf_counter

import numpy as np

from brute_force import NetSizeError, brute_force_estimation
from IP_algorithm import ip_estimation
from random_block_selection import random_block_selection
from projected_algorithm import projected_estimation
from random_direction_algorithm import random_mom_estimation, random_trimmed_estimation
from trimmed_IP_algorithm import trimmed_ip_estimation
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


def geometric_mom_estimation(data, s, epsilon, lambda_upper, delta=0.05, *, tol=None, seed=None, C=2):
    """Geometric median of the shared block means, then top-s hard thresholding."""
    start = perf_counter()
    block_means, *_ = mom_initialization(
        data, s, epsilon, lambda_upper, delta, tol=tol, seed=seed, C=C,
    )
    K, d = block_means.shape
    median = block_means.mean(axis=0)
    for _ in range(10_000):
        residuals = block_means - median
        distances = np.linalg.norm(residuals, axis=1)
        at_point = distances < 1e-12
        weights = 1 / distances[~at_point]
        if at_point.any():
            if not len(weights):
                break
            direction = np.sum(residuals[~at_point] * weights[:, None], axis=0)
            length = np.linalg.norm(direction)
            if length <= at_point.sum():
                break
            step = (1 - at_point.sum() / length) * direction / weights.sum()
            updated = median + step
        else:
            updated = np.average(block_means, axis=0, weights=weights)
        if np.linalg.norm(updated - median) <= 1e-10 * max(1, np.linalg.norm(median)):
            median = updated
            break
        median = updated
    else:
        raise RuntimeError("Geometric median iteration did not converge")
    estimate = np.zeros(d)
    active = np.argsort(-np.abs(median))[:s]
    estimate[active] = median[active]
    return estimate, {"converged": True, "C": C, "K": K, "runtime": perf_counter() - start}


def plain_sample_mean_estimation(data):
    """Return the coordinate-wise average of every observed row."""
    start = perf_counter()
    estimate = np.mean(data, axis=0)
    return estimate, {"converged": True, "runtime": perf_counter() - start}


def sample_mean_ht_estimation(data, s):
    """Average all observations and retain the s largest absolute coordinates."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or not data.shape[0]:
        raise ValueError("data must be a nonempty (n, d) array")
    if isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer)) or not 1 <= s <= data.shape[1]:
        raise ValueError("s must be an integer between 1 and d")
    start = perf_counter()
    mean = np.mean(data, axis=0)
    estimate = np.zeros_like(mean)
    active = np.argsort(-np.abs(mean), kind="stable")[:s]
    estimate[active] = mean[active]
    return estimate, {"converged": True, "runtime": perf_counter() - start, "s": int(s)}


def run_experiment(
    n, epsilon, nu, s, delta=0.05, *, d, scale=1, seed=42, tol=1e-5, C=2,
    strength=30, net_radius=0.25, max_net_points=200_000, r=None,
    projected_objectives=("least_squares",), aggregation_tol=1e-5, aggregation_rtol=1e-5,
    aggregation_max_iter=1000, projection_method="partition", J=None, methods=None,
    tilde_J=None, direction_seed=None, block_selection_seed=None, subsample_multiplier=None,
    block_selection_with_replacement=False,
):
    """Return per-method error, support recovery, and runtime in seconds.

    The t shape matrix is scale * I_d. Draw one loc ~ Uniform(1, 3) per run
    and put it on s randomly chosen mean coordinates; the others are zero.
    All methods use the same contaminated data. MoM methods share C and the block seed;
    Algorithm 2 uses the raw observations and does not use C or the block seed.
    tol defaults to 1e-5; the MoM estimate does not depend on it.
    net_radius controls only the brute-force covering net (default: 1/4).
    max_net_points limits the full net before allocation (default: 200,000).
    Oversized nets are recorded as skipped; other solver failures still raise.
    Passing r adds projected_algorithm_1 (a random partition into d/r blocks of r coordinates),
    one method per entry of projected_objectives: 'least_squares' keeps the name
    projected_algorithm_1 and 'max' adds projected_algorithm_1_max.
    projection_method='haar' instead adds haar_projected_algorithm_1[_max], using
    J independent Haar projections with dense directions; J must be supplied.
    methods optionally selects from brute_force, algorithm_1, algorithm_2, projected,
    random_mom, random_trimmed, random_block_selection, coordinate_mom, geometric_mom,
    and sample_mean. random_block_selection samples a uniform subset without
    replacement by default and caps K_sub at K; block_selection_with_replacement
    enables the earlier sampling rule. block_selection_seed controls its stream,
    and C scales both K and K_sub_requested=C*max(s*log(e*d/s),log(4/delta)),
    rounded up to an odd integer. The optional subsample_multiplier alias must equal C.
    sample_mean averages all rows and retains the s largest absolute coordinates.
    None retains the old defaults. The random methods require positive integer
    tilde_J random directions, in addition to d fixed coordinate directions.
    direction_seed defaults to seed and is independent of the MoM block stream.
    Their stopping gaps refer only to this fixed finite direction pool.
    Algorithm 2 is opt-in and uses tol with its hybrid indicator oracle defaults.
    Both aggregations use support cutting planes with a separate squared-loss
    max(aggregation_tol, aggregation_rtol * abs(objective)) gap and
    aggregation_max_iter limit.
    Runtime includes estimator preprocessing, but excludes shared data generation
    and metric calculation. Support recovery uses an absolute threshold of 1e-8;
    it is not reported for the dense plain sample mean.
    """
    tol = 1e-5 if tol is None else tol
    if n < 1 or not 1 <= s <= d:
        raise ValueError("require n >= 1 and 1 <= s <= d")
    if not np.isfinite(nu) or nu <= 2:
        raise ValueError("nu must be finite and greater than 2 for a finite covariance")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite scalar")
    if not 0 <= epsilon < 0.5 or not 0 < delta < 1:
        raise ValueError("require 0 <= epsilon < 0.5 and 0 < delta < 1")
    if projection_method not in ("partition", "haar"):
        raise ValueError("projection_method must be 'partition' or 'haar'")
    if r is not None and (not isinstance(r, (int, np.integer)) or not 1 <= r <= d):
        raise ValueError("r must be an integer in [1, d]")
    if projection_method == "partition":
        if r is not None and d % r:
            raise ValueError("partition r must divide d")
        if J is not None:
            raise ValueError("partition J is determined by d/r; supply J only for Haar projections")
    elif r is None or not isinstance(J, (int, np.integer)) or J < 1:
        raise ValueError("Haar projections require r and a positive integer J")
    if not projected_objectives or any(objective not in ("least_squares", "max")
                                       for objective in projected_objectives):
        raise ValueError("projected_objectives must contain 'least_squares' and/or 'max'")
    method_names = {"brute_force", "algorithm_1", "algorithm_2", "projected", "random_mom",
                    "random_trimmed", "random_block_selection", "coordinate_mom", "geometric_mom", "sample_mean"}
    if methods is not None:
        if not methods or not set(methods) <= method_names:
            raise ValueError("methods must be a nonempty selection of supported method names")
        if "projected" in methods and r is None:
            raise ValueError("the projected method requires r")
        if {"random_mom", "random_trimmed"}.intersection(methods):
            if (isinstance(tilde_J, (bool, np.bool_))
                    or not isinstance(tilde_J, (int, np.integer)) or tilde_J < 1):
                raise ValueError("random methods require a positive integer tilde_J")

    np.random.seed(seed)
    loc = np.random.uniform(1, 3)
    data, truth = sparse_t_dist_data_generation(n, loc, nu, d, s, scale=scale)
    data = adversarial_sparse_contamination(data, epsilon, s, strength=strength)
    # Sigma = nu / (nu - 2) * scale * I_d; scale itself is not the covariance.
    lambda_upper = 2 * nu / (nu - 2) * scale

    estimators = [
        ("brute_force", brute_force_estimation),
        ("algorithm_1", ip_estimation),
        ("coordinate_mom", coordinate_mom_estimation),
        ("geometric_mom", geometric_mom_estimation),
        ("sample_mean", sample_mean_ht_estimation),
    ]
    method_options = {"brute_force": {"net_radius": net_radius, "max_net_points": max_net_points}}
    if methods is not None:
        if "algorithm_2" in methods:
            estimators.insert(2, ("algorithm_2", trimmed_ip_estimation))
        if "random_block_selection" in methods:
            estimators.append(("random_block_selection", random_block_selection))
            method_options["random_block_selection"] = {"block_selection_seed": block_selection_seed,
                                                         "subsample_multiplier": subsample_multiplier,
                                                         "replace": block_selection_with_replacement}
        for name, estimator in (("random_mom", random_mom_estimation),
                                ("random_trimmed", random_trimmed_estimation)):
            if name in methods:
                estimators.append((name, estimator))
                method_options[name] = {"tilde_J": tilde_J, "direction_seed": direction_seed}
        estimators = [(name, estimator) for name, estimator in estimators if name in methods]
    if r is not None and (methods is None or "projected" in methods):
        projected_estimator = projected_estimation
        prefix = "projected_algorithm_1"
        if projection_method == "haar":
            from haar_projected_algorithm import haar_projected_estimation
            projected_estimator = haar_projected_estimation
            prefix = "haar_projected_algorithm_1"
        for position, objective in enumerate(projected_objectives, start=2):
            name = prefix + ("_max" if objective == "max" else "")
            estimators.insert(position, (name, projected_estimator))
            method_options[name] = {"r": r, "objective": objective,
                                    "aggregation_tol": aggregation_tol, "aggregation_rtol": aggregation_rtol,
                                    "max_iter": aggregation_max_iter}
            if projection_method == "haar":
                method_options[name]["J"] = J

    results = {}
    for name, estimator in estimators:
        options = method_options.get(name, {})
        try:
            if name == "sample_mean":
                estimate, info = estimator(data, s)
            elif name == "algorithm_2":
                estimate, info = estimator(data, s, epsilon, delta, tol=tol)
            elif name == "random_trimmed":
                estimate, info = estimator(data, s, epsilon, delta, tol=tol, seed=seed, **options)
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
    parser.add_argument("--delta", type=float, default=0.05, help="failure probability (default: 0.05)")
    parser.add_argument("--d", "--dim", "-d", type=int, required=True, help="dimension")
    parser.add_argument("--scale", type=float, default=1, help="t shape = scale * I_d (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="data and block seed (default: 42)")
    parser.add_argument("--tol", type=float, default=1e-5, help="optimization tolerance (default: 1e-5)")
    parser.add_argument("--C", type=float, default=2, help="block-count multiplier (default: 2)")
    parser.add_argument("--strength", type=float, default=30, help="adversarial contamination strength (default: 30)")
    parser.add_argument("--net-radius", type=float, default=0.25, help="brute-force covering radius in (0, 1] (default: 0.25)")
    parser.add_argument("--max-net-points", type=int, default=200_000, help="brute-force net size limit (default: 200,000)")
    parser.add_argument("--r", type=int,
                        help="projection dimension (partition requires r to divide d)")
    parser.add_argument("--projection-method", choices=("partition", "haar"), default="partition",
                        help="projection construction (default: partition)")
    parser.add_argument("--J", "-J", type=int,
                        help="number of independent Haar projections (required for haar)")
    parser.add_argument("--tilde-J", dest="tilde_J", type=int,
                        help="random direction count, plus d coordinate directions (required for random methods)")
    parser.add_argument("--direction-seed", type=int,
                        help="random direction seed (default: --seed; separate from the MoM block stream)")
    parser.add_argument("--block-selection-seed", type=int,
                        help="block selection seed (default: --seed; independent of block construction)")
    parser.add_argument("--subsample-multiplier", type=float, default=None,
                        help="compatibility alias for --C; when supplied it must equal --C")
    parser.add_argument("--block-selection-with-replacement", action="store_true",
                        help="use the earlier uncapped draws with replacement (default: no replacement)")
    parser.add_argument("--methods", nargs="+",
                        choices=("brute_force", "algorithm_1", "algorithm_2", "projected", "random_mom",
                                 "random_trimmed", "random_block_selection", "coordinate_mom", "geometric_mom", "sample_mean"),
                        help="run only these methods (default: original methods, plus projected when r is supplied)")
    parser.add_argument("--projected-objectives", nargs="+", choices=("least_squares", "max"),
                        default=["least_squares"], help="aggregation objectives (default: least_squares)")
    parser.add_argument("--aggregation-tol", type=float, default=1e-5,
                        help="projected aggregation absolute squared-loss gap (default: 1e-5)")
    parser.add_argument("--aggregation-rtol", type=float, default=1e-5,
                        help="projected aggregation relative gap; 0 uses absolute only (default: 1e-5)")
    parser.add_argument("--aggregation-max-iter", type=int, default=1000,
                        help="projected aggregation cutting-plane iteration limit (default: 1000)")
    args = parser.parse_args()
    try:
        results = run_experiment(**vars(args))
    except ValueError as error:
        parser.error(str(error))
    labels = {
        "sample_mean": "Sample mean + HT",
        "random_block_selection": "Random block selection",
        "algorithm_2": "Algorithm 2 trimmed mean",
        "random_mom": "Random-direction MoM",
        "random_trimmed": "Random-direction trimmed",
        "projected_algorithm_1": "Projection-partition LS",
        "projected_algorithm_1_max": "Projection-partition max",
        "haar_projected_algorithm_1": "Haar-projection LS",
        "haar_projected_algorithm_1_max": "Haar-projection max",
    }
    print(f"{'method':<26} {'l2_error':>12} {'support_recovery':>18} {'runtime_s':>12}")
    for name, metrics in results.items():
        name = labels.get(name, name)
        if metrics.get("status") == "skipped":
            print(f"{name:<26} {'—':>12} {'—':>18} {'—':>12}  skipped: {metrics['reason']}")
            continue
        recovery = ("—" if metrics["support_recovery"] is None else
                    f"{metrics['support_recovery']:.6f}")
        print(f"{name:<26} {metrics['error']:>12.6f} "
              f"{recovery:>18} {metrics['runtime']:>12.6f}")


if __name__ == "__main__":
    main()
