"""Closed-form baselines: hard-thresholded sample mean, coordinate-wise MoM, and geometric MoM."""

from time import perf_counter

import numpy as np

from utils import check_data, mom_blocks, top_s


def sample_mean_ht_estimation(data, s):
    """Average of all rows, keeping its s largest-magnitude coordinates."""
    data = check_data(data, s)
    start = perf_counter()
    mean = np.mean(data, axis=0)
    return mean * top_s(mean, s), {"converged": True, "runtime": perf_counter() - start}


def coordinate_mom_estimation(data, s, epsilon, *, delta=0.05, C=2, seed=None):
    """Coordinate-wise median of the MoM block means, keeping its s largest-magnitude coordinates."""
    start = perf_counter()
    blocks = mom_blocks(data, s, epsilon, delta=delta, C=C, seed=seed)
    center = np.median(blocks, axis=0)
    estimate = center * top_s(center, s)
    return estimate, {"converged": True, "C": C, "K": len(blocks), "runtime": perf_counter() - start}


def geometric_mom_estimation(data, s, epsilon, *, delta=0.05, C=2, seed=None):
    """Geometric median of the MoM block means, keeping its s largest-magnitude coordinates.

    Weiszfeld iterations from the mean, with the Vardi-Zhang step at a block mean.
    """
    start = perf_counter()
    blocks = mom_blocks(data, s, epsilon, delta=delta, C=C, seed=seed)
    median = blocks.mean(axis=0)
    for _ in range(10_000):
        residuals = blocks - median
        distances = np.linalg.norm(residuals, axis=1)
        at_point = distances < 1e-12
        weights = 1 / distances[~at_point]
        if at_point.any():
            if not len(weights):
                break
            direction = np.sum(residuals[~at_point] * weights[:, None], axis=0)
            length = np.linalg.norm(direction)
            if length <= at_point.sum():
                break  # the current block mean is the geometric median
            updated = median + (1 - at_point.sum() / length) * direction / weights.sum()
        else:
            updated = np.average(blocks, axis=0, weights=weights)
        done = np.linalg.norm(updated - median) <= 1e-10 * max(1, np.linalg.norm(median))
        median = updated
        if done:
            break
    else:
        raise RuntimeError("Geometric median iteration did not converge")
    estimate = median * top_s(median, s)
    return estimate, {"converged": True, "C": C, "K": len(blocks), "runtime": perf_counter() - start}
