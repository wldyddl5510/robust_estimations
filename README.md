# Robust estimation via Integer Programming

[`results.md`](results.md) collects experiments run with the current code. Their outputs (JSON,
figures, logs) stay local: `.gitignore` excludes them, so only code, tests, and documents are tracked.

Experiments run before the October 2026 refactor, with their results, runners, and frozen `source/`
code snapshots, are kept locally in `previous_experiments/` (also excluded from git). They used the
earlier block rule and API, and remain in the git history up to commit `1686be1`.

Use the `robust_ip_estimation` conda environment (`python -m pip install -r requirements.txt`).
All optimization uses Gurobi with one thread and needs a working license; the Depersin–Lecué
baseline uses CVXPY/Clarabel.

## Modules

| Module | Contents |
| --- | --- |
| `utils.py` | MoM blocks and the block rule, hard thresholding `top_s`, the shared support cutting plane, Gurobi setup |
| `data.py` | (skew-)t sample generation and the adversarial sparse contamination |
| `IP_algorithm.py` | Algorithm 1: separation oracle (10), heuristic separation, restricted SOCP (17), the inner loop `ConstraintGeneration`, `ip_estimation` (with optional block subsampling), and `dense_estimation` |
| `trimmed_IP_algorithm.py` | Algorithm 2: the trimmed-mean oracles (21), (22) plugged into the same inner loop |
| `finite_directions.py` | Brute force over a covering net, and MoM/trimmed objectives over a random direction bank |
| `projection.py` | Projected Algorithm 1: coordinate partitions or Haar projections, dense fits, sparse LS aggregation |
| `baselines.py` | Sample mean + HT, coordinate-wise MoM, geometric MoM |
| `depersin_lecue.py` | Depersin–Lecué covSDPofMeans with a Clarabel covering SDP |
| `experiments.py` | `run_method(name, data, ...)` registry for experiment runners, and `metrics` |
| `plot_histograms.py` | Per-method error histograms from an errors JSON file |

```python
import numpy as np
from data import adversarial_sparse_contamination, sparse_t_dist_data_generation
from experiments import metrics, run_method
from IP_algorithm import ip_estimation

np.random.seed(42)
data, truth = sparse_t_dist_data_generation(200, loc=3, nu=5, d=4, s=2)
data = adversarial_sparse_contamination(data, epsilon=0.05, s=2)
estimate, info = ip_estimation(data, 2, 0.05, seed=42)
print(metrics(estimate, truth), info["converged"], info["runtime"])
estimate, info = run_method("haar", data, s=2, epsilon=0.05, seed=42, r=2, J=4)
```

Every estimator returns `(estimate, info)`. The MoM-based ones share the keywords
`delta=0.05, C=3, seed=None, tol, max_iter=1000`. Algorithms 1 and 2 default to `tol=None`, which
uses `rtol * U0` with `rtol=1e-3` and U0 the deterministic upper bound at the initial estimate, so
the tolerance scales with the data (`rtol` itself if U0 = 0); the other optimization methods default
to the absolute `tol=1e-5`. `run_method(..., tol=None)` keeps each method's default. `info` always reports `converged`
and `runtime` (seconds, including preprocessing); optimization methods add `objective` (a certified
upper bound), `lower_bound`, `gap`, `tol`, `iterations`, the selected `support`, and `K`. Reaching
`max_iter` returns `converged=False`, while an uncertified inner solve raises.

Run the tests with `python -m unittest discover -s tests`.

## Block rule

K is the smallest odd integer at least `C * max(s*log(e*d/s), epsilon*n, log(1/delta))`; the
complexity term equals d at full support. Blocks are balanced, use all samples, and come from one
seeded permutation, so the same `seed` gives the same partition in every MoM method. A rule
requiring K > n raises. This rule replaced the earlier `s*log(d/s)` complexity (d at s = d) used by
the recorded experiments.

## Algorithm 1

`ip_estimation` minimizes F(mu) = max_v median_k <block_mean_k - mu, v> over ||mu||_0 <= s, with v
ranging over unit vectors with at most min(2s, d) nonzeros. `utils.cutting_plane` searches supports
with a Gurobi MILP over dual support cuts; each support is solved by `ConstraintGeneration`, which
alternates the restricted SOCP (17) over generated (S, B) pairs with separation (10):

- Generated (S, B) pairs persist across supports.
- Cheap separation comes first: sparse candidate directions and at most four fixed-(S, B) convex
  solves. Its values are only lower bounds and never certificates.
- On a support being certified, a failed heuristic is followed by `swap_separation`. The pairs
  whose restricted convex combination `c = sum_k alpha_k (block_mean_k - mu)` is within tol/4 of L
  are refined as they are and with one support coordinate i replaced by j, in decreasing order of
  `c_j^2 - c_i^2`, alternating the fixed-pair direction (the NNLS min-norm point of the rows' hull)
  with its top-h blocks. If no move finds a violated pair, each move is refined again without one
  block of its min-norm corral (the blocks with positive NNLS weight, largest first): only those can
  raise the fixed-pair value when dropped, and the top-h reselection picks the replacement. It adds
  the deepest of up to `SWAP_HITS = 8` violated new pairs found within `SWAP_BUDGET = 0.5` times the
  mean exact-call time (0.25 s before the first call), and otherwise falls through to the exact MIP.
  It never certifies. Algorithm 2 does not use it.
- A support's first visit returns a valid cut after at most three restricted solves, bounded above
  by the deterministic bound `median_k ||block_mean_k - mu||_(min(2s,d), 2)`. A repeated support
  must certify `upper_F - L <= tol/4` with that bound or the exact MIP (10), which stops as soon as
  it finds a violated cut (incumbent above the threshold by half its gap tolerance) or certifies the bound. A repeated support whose restricted value L is
  already within tol of the incumbent upper bound returns its cut without certification: it cannot
  improve the incumbent by more than tol, and the outer search ends if the master returns to it.
- The MIP (10) writes each block row as a big-M constraint with M_i = ||r_i||_(min(2s,d)) + max
  level. With a threshold G it also screens rows with ||r_i|| < G and adds b_i + b_j <= 1 for row
  pairs whose segment has top-k norm below G; a certificate then reports
  max(MIP bound, largest screened value), and an infeasible screened model is itself a certificate.
- `info` counts `oracle_cuts`, `oracle_certificates`, `revisit_prunes`, deferred supports and
  deterministic certificates, plus `swap_calls` and `swap_cuts`. It records `restricted_time`,
  `heuristic_time`, `swap_time`, and `oracle_time`.
- Coordinate directions give the box |mu_j| <= |center_j| + F(initial), so the restricted problems
  are bounded.

`subsample=True` is block subsampling: only K_sub = min(K, smallest odd integer >=
`C * max(s*log(e*d/s), log(4/delta))`) blocks of the same partition, drawn uniformly without
replacement from an independent seed stream, enter the optimization (`info` adds `K_sub` and
`selected_block_indices`). With K_sub = K this is exactly Algorithm 1.
`ip_estimation_from_block_means(block_means, s)` runs Algorithm 1 on any odd collection of block
means, and `dense_estimation` runs it on the full support without the support search.

## Algorithm 2

`trimmed_ip_estimation(data, s, epsilon, delta=..., tol=...)` uses the raw observations with
`k = floor(epsilon*n) + max(ceil(log(2*C(d, min(2s,d))/delta)), ceil(n*min(epsilon, 1/2-epsilon)/2))`
removals per tail (epsilon < 1/2, natural log) and q = n - 2k retained values, asserting 2k < n.

`subsample=True` (method `trimmed_subsampling`) runs Algorithm 2 on n_sub rows drawn uniformly
without replacement from an independent seed stream, for moment order `p > 1` (default 2):
`n_sub = ceil(D_s / epsilon^(2-2/p))` with `D_s = s*log(e*d/s) + log(4/delta)`, or `ceil(n/4)` at
epsilon = 0. With `h = min(floor(epsilon*n), floor(epsilon*n_sub + sqrt(2*epsilon*n_sub*log(2/delta))
+ 2*log(2/delta)/3))` outliers allowed in the subsample, it trims
`k_sub = h + max(ceil(log(4*C(d, min(2s,d))/delta)), ceil(min(h, n_sub/2 - h)/2))` per tail and
asserts 2k_sub < n_sub. When n_sub >= n it is exactly Algorithm 2. `info` adds `n_sub`, `h`,
`k_sub`, and `selected_indices`. Its witness B holds the n - k largest projections,
the restricted SOCP caps simplex weights at 1/q, and support cuts are rebuilt from the cone vectors
(with an explicit dual fallback) so that inaccurate solver duals cannot produce invalid cuts.
The inner loop and stopping rules are Algorithm 1's.

## Finite direction sets

`brute_force_estimation` evaluates the MoM objective on a deterministic radius-1/4 net of the
2s-sparse unit ball (rounding to a grid of spacing 2r/sqrt(k), then projecting onto the ball, gives
the covering) and raises `NetSizeError` above 200,000 rows. `random_mom_estimation` and
`random_trimmed_estimation` use d coordinate directions plus `tilde_J` random sparse unit vectors
drawn from a seed stream independent of the blocks. Each support is an LP on
`max_j |target_j - <v_j, mu>|`; gaps certify only the finite-set objective.

## Projections

`partition_estimation(..., r)` splits the coordinates into d/r random blocks of r (r must divide d);
each block's data run dense Algorithm 1 with sparsity min(s, r), sharing the seed's block partition.
`haar_estimation(..., r, J)` draws J independent Haar projections with orthonormal rows; each
projected sample runs dense Algorithm 1 on the full sphere V_r with K from (r, r, delta/J), all
projections sharing one partition from an independent child stream. Both then minimize
sum_j ||P_j mu - estimate_j||^2 over ||mu||_0 <= s by cutting planes, to the gap max(tol, tol * objective).
Partition aggregation is exact per support; Haar aggregation restricts mu to the data-driven box
|mu_i| <= |coordinate MoM_i| + U_det, which defines the estimator rather than certifying that the
unboxed optimum lies inside.

## Literature baseline

`depersin_lecue_estimation(data, K=..., seed=..., tol=...)` follows arXiv:1906.03058, Algorithm 4,
with an explicit K, balanced blocks that keep every sample, the conservative outer iteration bound,
and rho bisection with the [0.9981, 1] acceptance interval derived for eta = 1e-4. `tol` only sets
the SDP solver tolerances. A practical K does not confer the paper's guarantee.

## Removed in the October 2026 cleanup

The specialized PTZ positive-SDP backend, the Cherapanamjeri–Flammarion–Bartlett estimator, max
aggregation with its refinement, the separation-oracle ablation options (benchmarked in
`previous_experiments/artifacts/benchmarks/oracle_acceleration/`), the separate dense
Algorithm 1 wrapper (`ip_estimation` with s = d optimizes the same objective), block subsampling with
replacement, and the `lambda_upper` argument (it only entered a tolerance check). Their last versions
remain in commit `1686be1` and in the frozen snapshots under `previous_experiments/`.

A second pass (October 21, 2026) kept only the separation speedups with a measured gain (the swap
search and its corral drops, see `results.md`). The block-count SDP screen (`sdp_separation.py`,
`sdp_time_limit`), the geometric plane/BnB oracle (`geometric_separation.py`, `geometric_time_limit`,
`geometric_max_nodes`; for q = 2 an exact arc sweep), the sorting–IHT witness search (`sorting_iht.py`,
`iht_time_limit`), the pre-dualization directional LP (`directional_mom.py`, `formulation="lp"`), the
oracle's `return_direction` option, their tests and benchmark runners, and the separation-MIP
prototypes are in `previous_experiments/oracle_speedups_2026_10/`, together with the Algorithm 1
source they ran against (`source_snapshot/`).
