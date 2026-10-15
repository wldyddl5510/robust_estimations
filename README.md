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
| `sdp_separation.py` | Algorithm 1: one-sided block-count SDP certificate before exact separation, with corrected dual upper bounds |
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
- A support's first visit returns a valid cut after at most three restricted solves, bounded above
  by the deterministic bound `median_k ||block_mean_k - mu||_(min(2s,d), 2)`. A repeated support
  must certify `upper_F - L <= tol/4` with that bound, the SDP screen below, or the exact MIP (10), which stops as soon as
  it finds a violated cut (incumbent above the threshold by half its gap tolerance) or certifies the bound. A repeated support whose restricted value L is
  already within tol of the incumbent upper bound returns its cut without certification: it cannot
  improve the incumbent by more than tol, and the outer search ends if the master returns to it.
- Immediately before an exact MIP call, the SDP screen uses the existing
  `threshold = L + inner_tol`, where L is the restricted SOCP value r and `inner_tol = tol/4`.
  With `a_k = block_mean_k - mu`, it maximizes `sum(p)` subject to
  `[1, w'; w, Q] >= 0` (PSD), `tr(Q) <= 1`, `sum_ij abs(Q_ij) <= 2s tr(Q)`,
  `||w||_1 <= sqrt(2s)`, `0 <= p <= 1`, and, for each block,
  `[1, p_k, a_k'w; p_k, p_k, c_k; a_k'w, c_k, a_k'Q a_k] >= 0` (PSD),
  `c_k >= threshold*p_k`. A corrected dual upper bound strictly below `h = (K+1)/2`
  certifies `F(mu) < threshold`; the returned F upper bound is **threshold**, never the SDP
  block-count objective. A large SDP value does not establish a violated direction or add a cut.
  Missing dependencies, solver failures, time/iteration limits, and noncertifying bounds all
  fall back to exact separation. The SDP is specific to Algorithm 1; Algorithm 2 disables it.
  The screen is optional and off by default (`sdp_time_limit=0`); pass e.g. `sdp_time_limit=0.25`
  (seconds of native solve budget per screen, `IP_algorithm.SDP_TIME_LIMIT`) to `ip_estimation`,
  `ip_estimation_from_block_means`, or `dense_estimation` to enable it. The reusable
  CVXPY/Clarabel model has a 60-iteration limit. Python construction/canonicalization
  is outside that native time limit, so the extra screen need not improve every runtime.
  Certification uses dual multipliers with explicit stationarity and PSD-residual corrections
  plus floating-point safeguards; an approximate primal objective is never an upper certificate.
- An optional geometric oracle in `geometric_separation.py` runs before the SDP/MIP when
  `geometric_time_limit > 0` (default: **0, disabled**). It first sweeps sparse planes for an
  actual median witness, then branches over `q-2` coordinates on each size-`q` support, with
  `q = min(2s,d)`. In this mode both the existing heuristic and plane cuts use
  `T_cut = L + tol/8`; certification uses `T_cert = L + tol/4`. These decision conditions
  overlap: a cut and a certificate can both be valid when `T_cut < F(mu) < T_cert`.
  Global certification requires covering every support and every remaining box. Otherwise it
  returns `UNKNOWN` and continues to the enabled SDP screen or exact MIP. A count upper bound
  is never returned as an F upper bound: a geometric certificate returns `T_cert`.
  The certificate kernel uses exact rational arithmetic on the represented residuals and
  outward rational square-root bounds. The half-angle parameterization of the circle turns
  each angular condition into a quadratic inequality, including the omitted direction
  `(-1,0)` explicitly. Heuristic trigonometry is used only to propose rechecked witnesses.
  Integration also computes an exact bound on residual subtraction error and adjusts both
  thresholds conservatively before calling the geometric kernel.
  `geometric_max_nodes=5000` is the **total** BnB node budget per call; setting it to 0 enables
  plane search without BnB. The time budget is checked between work units and is a soft limit.
  For example, `ip_estimation(..., geometric_time_limit=0.25)` enables this
  alternative with exact-MIP fallback. All three Algorithm 1 entry points accept these options.
  Algorithm 2 keeps the geometric oracle disabled.
- The MIP (10) writes each block row as a big-M constraint with M_i = ||r_i||_(min(2s,d)) + max
  level. With a threshold G it also screens rows with ||r_i|| < G and adds b_i + b_j <= 1 for row
  pairs whose segment has top-k norm below G; a certificate then reports
  max(MIP bound, largest screened value), and an infeasible screened model is itself a certificate.
- `info` counts `oracle_cuts`, `oracle_certificates`, `revisit_prunes`, deferred supports and
  deterministic certificates, plus `sdp_calls`, `sdp_certificates`, and `sdp_fallbacks`.
  It records `restricted_time`, `heuristic_time`, `oracle_time`, and `sdp_certificate_time`
  (including SDP import, model construction, canonicalization, solve and bound correction).
  Geometric diagnostics add `geometric_calls`, `geometric_cuts`, `geometric_certificates`,
  `geometric_fallbacks`, `geometric_nodes`, `geometric_supports`, and `geometric_time`.
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
