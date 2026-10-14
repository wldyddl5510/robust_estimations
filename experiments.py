"""Method registry for experiment runners: run any estimator by name with a shared setting."""

import numpy as np

from baselines import coordinate_mom_estimation, geometric_mom_estimation, sample_mean_ht_estimation
from depersin_lecue import depersin_lecue_estimation
from finite_directions import brute_force_estimation, random_mom_estimation, random_trimmed_estimation
from IP_algorithm import ip_estimation
from projection import haar_estimation, partition_estimation
from trimmed_IP_algorithm import trimmed_ip_estimation

METHODS = ("sample_mean", "coordinate_mom", "geometric_mom", "algorithm_1", "block_subsampling",
           "algorithm_2", "brute_force", "random_mom", "random_trimmed", "partition", "haar", "dl")


def run_method(name, data, *, s, epsilon, delta=0.05, C=2, seed=None, tol=1e-5, **options):
    """Return (estimate, info) of the named method.

    options are method-specific: tilde_J (random_*), r (partition), r and J (haar), K (dl),
    max_iter (optimization methods). Methods ignore setting entries they do not use, e.g. C and
    seed for Algorithm 2.
    """
    mom = {"delta": delta, "C": C, "seed": seed}
    calls = {
        "sample_mean": lambda: sample_mean_ht_estimation(data, s, **options),
        "coordinate_mom": lambda: coordinate_mom_estimation(data, s, epsilon, **mom, **options),
        "geometric_mom": lambda: geometric_mom_estimation(data, s, epsilon, **mom, **options),
        "algorithm_1": lambda: ip_estimation(data, s, epsilon, **mom, tol=tol, **options),
        "block_subsampling": lambda: ip_estimation(data, s, epsilon, **mom, tol=tol, subsample=True, **options),
        "algorithm_2": lambda: trimmed_ip_estimation(data, s, epsilon, delta=delta, tol=tol, **options),
        "brute_force": lambda: brute_force_estimation(data, s, epsilon, **mom, tol=tol, **options),
        "random_mom": lambda: random_mom_estimation(data, s, epsilon, **mom, tol=tol, **options),
        "random_trimmed": lambda: random_trimmed_estimation(data, s, epsilon, delta=delta, seed=seed, tol=tol,
                                                            **options),
        "partition": lambda: partition_estimation(data, s, epsilon, **mom, tol=tol, **options),
        "haar": lambda: haar_estimation(data, s, epsilon, **mom, tol=tol, **options),
        "dl": lambda: depersin_lecue_estimation(data, seed=seed, tol=tol, **options),
    }
    if name not in calls:
        raise ValueError(f"unknown method {name!r}; choose from {METHODS}")
    return calls[name]()


def metrics(estimate, truth):
    """L2 error and the fraction of true support coordinates estimated as nonzero (|x| > 1e-8)."""
    return {"error": float(np.linalg.norm(estimate - truth)),
            "support_recovery": float(np.sum((np.abs(estimate) > 1e-8) & (truth != 0)) / np.count_nonzero(truth))}
