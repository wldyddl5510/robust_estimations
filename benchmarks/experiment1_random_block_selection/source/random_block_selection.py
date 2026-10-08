"""Algorithm 1 after uniform sampling of the original MoM blocks."""

from time import perf_counter

import numpy as np

from IP_algorithm import ip_estimation_from_block_means
from utils import mom_initialization


def random_block_selection(
    data, s, epsilon, lambda_upper, delta=0.05, *, tol=1e-5, seed=None, C=2,
    block_selection_seed=None, subsample_multiplier=2, replace=False,
    max_iter=1000, max_inner_iter=1000,
    oracle_strategy="hybrid", oracle_formulation="indicator", perspective=False,
):
    """Construct the usual K blocks, sample K_sub rows, and run Algorithm 1.

    K_sub_requested is the smallest odd integer >=
    subsample_multiplier*(s*log(d/s) + log(4/delta)); multiplier defaults to 2.
    By default, choose a uniform subset WITHOUT replacement, with
    K_sub = min(K, K_sub_requested). Both counts are odd, so the minimum is odd.
    A random permutation selects the subset; selected rows are restored to
    original block order. If all K are used, this is the original Algorithm 1.
    replace=True reproduces the earlier independent draws with replacement,
    without capping the count. Sampling occurs once, before optimization.

    seed controls the original partition, exactly as in ip_estimation.
    block_selection_seed defaults to seed, with a separate SeedSequence stream
    (spawn_key=(2,)) so sampling does not reuse the partition RNG stream.
    The subset's coordinate medians initialize the optimization. Default
    tolerance is 1e-5, including tol=None; all Algorithm 1 solver options apply.

    info['K'] is the number of original blocks and info['K_sub'] is the number
    of selected rows; K_sub_requested records the uncapped formula. With
    replacement, selected_block_indices preserves draw order and multiplicity.
    Runtime includes original block construction, selection, and optimization.
    Bounds and stopping gaps concern only the sampled-block objective.
    """
    start = perf_counter()
    if (isinstance(subsample_multiplier, (bool, np.bool_))
            or not isinstance(subsample_multiplier, (int, float, np.integer, np.floating))
            or not np.isfinite(subsample_multiplier) or subsample_multiplier <= 0):
        raise ValueError("subsample_multiplier must be positive and finite")
    if not isinstance(replace, (bool, np.bool_)):
        raise ValueError("replace must be a boolean")
    tol = 1e-5 if tol is None else tol
    block_means, _, _, _, tol = mom_initialization(
        data, s, epsilon, lambda_upper, delta, tol, seed, C=C,
    )
    K, d = block_means.shape
    threshold = subsample_multiplier * (s * np.log(d / s) + np.log(4.) - np.log(delta))
    K_sub_requested = int(np.ceil(threshold))
    K_sub_requested += K_sub_requested % 2 == 0
    K_sub = K_sub_requested if replace else min(K, K_sub_requested)
    selection_seed = seed if block_selection_seed is None else block_selection_seed
    stream = np.random.SeedSequence(selection_seed, spawn_key=(2,))
    rng = np.random.default_rng(stream)
    selected = (rng.integers(K, size=K_sub) if replace
                else np.sort(rng.permutation(K)[:K_sub]))
    estimate, info = ip_estimation_from_block_means(
        block_means[selected], s, tol=tol, max_iter=max_iter,
        max_inner_iter=max_inner_iter, oracle_strategy=oracle_strategy,
        oracle_formulation=oracle_formulation, perspective=perspective,
    )
    info.update({
        "C": C, "K": K, "K_sub": K_sub, "K_sub_requested": K_sub_requested,
        "subsample_multiplier": float(subsample_multiplier),
        "selected_block_indices": selected.tolist(),
        "unique_selected_blocks": int(np.unique(selected).size),
        "sampling_with_replacement": bool(replace),
        "block_selection_seed": int(selection_seed) if isinstance(selection_seed, (int, np.integer)) else selection_seed,
        "block_selection_stream": [2],
        "runtime": perf_counter() - start,
    })
    return estimate, info
