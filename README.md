# Robust estimation via Integer Programming

Use the `robust_ip_estimation` conda environment and install dependencies with
`python -m pip install -r requirements.txt`.

The net baseline evaluates a deterministic Euclidean net (radius 1/4 by default),
solves each fixed-support LP with Gurobi, and searches supports with the
draft's dual cutting planes and a Gurobi MILP. Gurobi requires a working license.
The original optimization programs, including their test reference models, use Gurobi.
The new non-sparse literature estimators use a specialized positive-SDP routine
or CVXPY/Clarabel; see [non-sparse implementation and limitations](README_non_sparse.md).
SciPy is used for t-distribution sampling and a nearest-neighbor check of the net.

The shared outer loop is `utils.cutting_plane`. Each estimator supplies a
`solve_support(z)` callback returning `(mu, L, U, gradient)`: a feasible estimate,
the cut value, its objective upper bound, and the cut slope. The helper maintains
the incumbent, accumulates cuts, solves the support MILP, and checks the global
gap. Separate L and U also allow Algorithm 1 to use an approximate inner oracle.

`IP_algorithm.ip_estimation` implements Algorithm 1: Gurobi solves both the mixed
integer conic separation problem (10) and the restricted SOCP (17).
Generated (S,B) constraints are retained across support searches.
By default, `oracle_strategy="hybrid"` generates constraints from feasible sparse
directions and a bounded alternating heuristic: fix `(S,B)` for a convex SOCP,
update B from the largest half of the projections, and propose new sparse
supports. This avoids solving a support MIP inside the heuristic. On the first
visit to a mean support, at most three restricted SOCP solves produce a valid
outer cut, even if the constraint set is incomplete. Repeated supports use
certified separation; a heuristic stall then falls back to (10). Dense projected
estimates always certify their final inner gap.

Initialization and incomplete support solves use the deterministic upper bound
`median_k ||block_means[k] - mu||_(min(2*s,d),2)`. A heuristic value is only a
lower bound and is never used as `U_best` or as a stopping certificate. Exact
oracle calls stop when an incumbent provides a violated cut or Gurobi's global
bound certifies `upper_F - L <= tol/4`. Set `oracle_strategy="exact"` to call the
MIP at initialization and every inner iteration, as before.

The median row-norm cutoff was already present. `oracle_formulation="bigm"`
adds explicit per-block big-M inequalities, with
`M_k = median(row_bounds) + row_bounds[k]`. The default remains `"indicator"`:
the toy benchmark did not show a consistent big-M speedup. When `2*s >= d`,
the redundant direction-support binaries are omitted. `perspective=True` adds
`v_j**2 <= zeta_j * energy_j` and `sum(energy) <= 1` in sparse directions. These
are valid rotated-cone constraints, but add variables/cones, so perspective is
optional and off by default. These options are accepted by `ip_estimation`,
`dense_estimation`, and `projected_estimation`; the standalone oracle uses the
keyword `formulation` instead of `oracle_formulation`.

Hyperplane-arrangement pruning is not implemented. The homogeneous equations
`(block_means[k]-mu) @ v - G = 0` all admit `(v,G)=(0,0)`, making naive sign-pattern
feasibility pruning ineffective. A positive-G dehomogenization and explicit
zero-optimum handling would be needed to adapt the
[LTS arrangement method](https://arxiv.org/pdf/2604.11584).
The programs have separate wrappers: `brute_force.solve_fixed_support_lp`,
`IP_algorithm.separation_oracle`, and `IP_algorithm.solve_restricted_socp`.
The SOCP uses convex quadratic cone constraints and `QCPDual=1` to obtain the
linear box-constraint duals. For both LP and SOCP, the cut gradient is
`M * (upper.Pi + lower.Pi)`, using Gurobi's nonpositive duals for <= constraints
in a minimization problem (see the
[Gurobi Pi documentation](https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/constraintlinear.html#pi)).
Models use one Gurobi thread.

Oracle acceleration was checked on 2026-09-30 with Gurobi 13.0.3, n=120,
multivariate t5 noise with covariance I, two replacement outliers, tol=0.01,
three data/block seeds and three serial repetitions per seed. Times below are
the mean of the three per-seed median end-to-end wall times. Initialization,
restricted solves, support search, and projection aggregation are included;
independent certificate checks are excluded.

| Toy case | Original code (s) | Default hybrid (s) | Speedup | Exact MIP calls, sum over seeds |
| --- | ---: | ---: | ---: | ---: |
| Sparse d=4, s=1, K=5 | 0.0818 | 0.0688 | 1.19x | 17 -> 6 |
| Sparse d=6, s=2, K=7 | 0.2727 | 0.1996 | 1.37x | 38 -> 14 |
| Projected d=6, s=2, r=2, K=7 | 0.1421 | 0.0988 | 1.44x | 43 -> 11 |

All 135 runs across the original and four improved configurations converged.
The original indicator oracle independently checked the returned means (each
local dense mean for projection): every reported upper bound and requested gap
passed within solver precision. Analytic/exhaustive tests also check oracle
optima, incomplete-set dual cuts, and forced heuristic failure. Estimates need
not be identical when the objective has multiple near-minimizers.

These are small toy timings, not a large-instance scaling claim: the d=4 seed-0
default run was slower (0.0617 -> 0.0841 s). Explicit big-M with exact separation
alone averaged 0.0873/0.3343/0.1546 s, and hybrid big-M averaged
0.0793/0.2034/0.0968 s. Hybrid big-M plus perspective averaged
0.0674/0.2406/0.1013 s; perspective is skipped for dense directions, so the last
difference is timing variability. This mixed evidence is why indicator remains
the default formulation and perspective remains off.

Raw runs, certificates, and source provenance are in
[`oracle_benchmark_results.json`](oracle_benchmark_results.json).
[`benchmark_oracle.py`](benchmark_oracle.py) reproduces the datasets and current
ablations:

```sh
python benchmark_oracle.py --module-dir . \
  --variants exact_indicator hybrid_indicator exact_bigm hybrid_bigm hybrid_perspective \
  --output /tmp/oracle_ablations.json
```

For a before/after comparison, save the original source directory first and pass
it as `--module-dir ORIGINAL --baseline-dir ORIGINAL --variants baseline`, then
run the modified directory with `--baseline-dir ORIGINAL`. The recorded original
was the user's working tree (including pre-existing edits), saved at
`/tmp/robust_oracle_baseline`; it is not the Git HEAD version. Current
`exact_indicator` additionally removes redundant dense support binaries.

```python
import numpy as np
from utils import sparse_t_dist_data_generation, adversarial_sparse_contamination
from brute_force import brute_force_estimation

np.random.seed(42)
data, truth = sparse_t_dist_data_generation(200, loc=3, nu=5, d=4, s=2)
data = adversarial_sparse_contamination(data, epsilon=0.05, s=2)
estimate, info = brute_force_estimation(
    data, s=2, epsilon=0.05, lambda_upper=2 * 5 / (5 - 2), seed=42,
)
if info["converged"]:
    error = np.linalg.norm(estimate - truth)
    recovered = (np.abs(estimate) > 1e-8) & (truth != 0)
    support_recovery = recovered.sum() / np.count_nonzero(truth)
    print(error, support_recovery, info["runtime"])

from IP_algorithm import ip_estimation
ip_estimate, ip_info = ip_estimation(
    data, s=2, epsilon=0.05, lambda_upper=2 * 5 / (5 - 2), seed=42,
)
```

`lambda_upper` bounds the covariance eigenvalue, not the t-distribution's scale
matrix eigenvalue. The default block count uses C=2. Algorithm 1 and the
projected estimator use an absolute dense-stage tolerance of `1e-5`. The direct
brute-force API uses `sqrt(K * lambda_upper / n) / 4` when no tolerance is supplied.
Blocks are balanced and use
all samples. Use the same block seed for both estimators. If the required odd
K exceeds n, the function raises an error instead of changing the rule.
Both estimators call `utils.mom_initialization` to share exactly this setup.
The multiplier is configurable with the keyword argument `C` in all three
estimators and `run_experiment`, or with `experiments.py --C 1`.
K is the smallest odd integer at least `C * max(complexity, epsilon*n, log(1/delta))`,
where `complexity = d` when `s == d` and `s*log(d/s)` otherwise. The same
full-support convention applies to the `random_block_selection` K_sub rule.

`info` reports convergence, the net objective, its global lower bound and gap,
iterations, K, net size, and end-to-end runtime in seconds (including net
construction and projections). A run reaching `max_iter` returns
`converged=False`. The gap concerns the finite-net objective, within solver
tolerances; it is not a certificate for the continuous supremum.

For Algorithm 1, `info['objective']` is an upper bound for the returned
estimate's continuous-direction objective, and the gap uses that bound.
Its info additionally reports exact MIP calls (`oracle_calls`), heuristic calls
and cuts (`heuristic_calls`, `heuristic_cuts`), partial support returns
(`deferred_supports`), deterministic inner certificates, total inner solves,
and the number of generated (S,B) constraints. An inner iteration limit or an uncertified
solver result raises an error; an outer iteration limit returns
`converged=False`. Runtime includes initialization and all oracle/SOCP/MILP work.
The separation model has 2*d+K+1 variables for sparse directions (an additional
d with perspective) and d+K+1 for dense directions. On 2026-09-23, after installing the
new license at `~/gurobi.lic`, Gurobi 13.0.3 in `robust_ip_estimation` solved
a 201-variable quadratic model and a 2,001-variable linear model to optimality.
The previous size-limited-license restrictions no longer blocked these checks;
no `GRB_LICENSE_FILE` environment variable was needed.

`trimmed_IP_algorithm.trimmed_ip_estimation` implements standalone Algorithm 2
with the same outer support cutting plane, using the raw n observations rather
than block means. Its trimming count and retained count are
`k = floor(epsilon*n) + max(ceil(log(2/delta)), ceil(epsilon*n/2))` and
`q = n - 2*k`, with natural logarithm. The estimator asserts `2*k < n`; it does
not reduce an invalid k automatically. Its API needs neither `lambda_upper`
nor `seed` nor `C`, since it does not construct MoM blocks. It uses the same
absolute `tol=1e-5` default as Algorithm 1, with inner tolerance `tol/4` and
separation tolerance `tol/8`.

```python
from trimmed_IP_algorithm import trimmed_ip_estimation

estimate, info = trimmed_ip_estimation(
    data, s=2, epsilon=0.05, delta=0.05,
    tol=1e-5, oracle_strategy="hybrid", oracle_formulation="indicator",
)
```

For each direction, Algorithm 2 averages the projections after discarding the
k smallest and k largest values. Its witness B contains the n-k largest
projections. The fixed-(S,B) heuristic maximizes the average of the q smallest
projections in B; the restricted SOCP uses simplex weights capped at `1/q`.
The default hybrid strategy reuses constraints, feasible sparse directions,
warm starts, and certification on revisited supports, as in Algorithm 1.
Initialization uses a feasible witness and the deterministic upper bound;
`oracle_strategy="exact"` instead calls certified separation at initialization
and each inner iteration.

The separation oracle supports `oracle_formulation="indicator"` (default) and
`"bigm"`, with optional `perspective=True` for sparse directions. At each oracle
call, let `p = min(2*s,d)` and let R_i be the Euclidean norm of the p largest
absolute coordinates of `data[i] - mu`. The threshold satisfies
`0 <= theta <= theta_ub`, where theta_ub is the (n-k)-th R_i in ascending order.
Explicit per-observation big-M constants are `theta_ub + R_i`; the same bounds
limit the hinge variables in the indicator formulation. The k-trimmed mean of
the R_i is a valid objective cutoff and deterministic upper bound, but is not
used as the upper bound on theta. Heuristic objective values remain lower
bounds until certified separation supplies an upper bound.

Restricted solves reconstruct valid support cuts from normalized cone vectors
and check them against feasible capped-simplex weights. They tighten solver
precision when needed and recover the explicit conic dual if its reported
dual vectors are inaccurate. A suboptimal barrier status is accepted only
when these independently evaluated bounds meet the requested inner accuracy.
Uncertified inner results raise; an outer iteration limit returns
`info['converged']=False`. `info` includes `n`, `k`, `effective_n=n-2*k`,
objective bounds, the gap, and the same oracle/heuristic counters as Algorithm 1.

Select Algorithm 2 explicitly in experiments; the default method list is
unchanged. It uses the same contaminated sample as the other selected methods,
and `--tol` sets its absolute stopping tolerance. Projection variants in this
repository continue to use Algorithm 1.

```sh
conda run -n robust_ip_estimation python experiments.py \
    --n 40 --d 2 --s 1 --epsilon 0.05 --nu 5 --seed 42 \
    --methods algorithm_2 coordinate_mom geometric_mom sample_mean
```

For Python experiments, use for example
`run_experiment(40, 0.05, 5, 1, d=2, methods=["algorithm_2", "sample_mean"])`.
The result key is `algorithm_2`, with the same error, support-recovery, and
runtime fields as Algorithm 1. `C` and the experiment's covariance bound only
affect methods that use MoM blocks; the experiment seed still controls the
shared generated data.

The **random-direction** methods approximate Section 6's inner supremum with
one fixed finite pool. `random_direction_algorithm.random_mom_estimation`
uses Algorithm 1's block medians; `random_trimmed_estimation` uses Algorithm 2's
trimmed projected means. Both require a positive integer `tilde_J`, counting
random draws in addition to all d coordinate directions. Each random draw
chooses a support of size `min(2*s,d)` uniformly, draws independent standard
Gaussian entries on it, and normalizes to the unit sphere. Antipodal copies
are unnecessary because the objective uses absolute residuals.

Directional targets are precomputed in memory batches and held fixed for the
entire optimization. Each fixed-support problem is an LP for
`max_j abs(target_j - direction_j @ mu)`; the existing outer support MILP and
dual cutting planes impose sparsity. These methods do not call the continuous
separation oracle or generate `(S,B)` cone constraints. The coordinate directions
preserve the box argument: with coordinate centers c and the initial feasible
sampled objective U, every estimate with objective at most U satisfies
`abs(mu_l - c_l) <= U`. Thus `M_l = abs(c_l) + U + margin` retains an optimizer
of the finite-pool problem. The outer support search can still be costly.

```python
from random_direction_algorithm import random_mom_estimation, random_trimmed_estimation

mom_estimate, mom_info = random_mom_estimation(
    data, s=2, epsilon=0.05, lambda_upper=2 * 5 / (5 - 2),
    tilde_J=1000, seed=42, direction_seed=17,
)
trimmed_estimate, trimmed_info = random_trimmed_estimation(
    data, s=2, epsilon=0.05, tilde_J=1000, direction_seed=17,
)
```

`direction_seed` defaults to `seed`. Directions use a separate random stream,
so changing their count or seed preserves the MoM blocks for a fixed `seed`.
With the same direction seed, increasing `tilde_J` extends the existing bank.
The MoM block rule and trimmed-mean trimming rule match their original
estimators. Both APIs accept `tol` (default `1e-5`), `max_iter` (default 1000),
and `batch_size` for target precomputation. The stored direction matrix is dense,
using `O(d * (tilde_J + d))` memory; batching limits projection workspace.
Their reported objectives, gaps,
and convergence apply only to the sampled pool plus coordinate directions;
they do not certify the continuous supremum or a theoretical sample count.

Select these methods explicitly; the experiment defaults stay unchanged:

```sh
conda run -n robust_ip_estimation python experiments.py \
    --n 200 --d 8 --s 2 --epsilon 0.01 --nu 5 --seed 42 \
    --methods random_mom random_trimmed sample_mean --tilde-J 1000 \
    --direction-seed 17
```

The Python runner accepts `methods=["random_mom", "random_trimmed"]`,
`tilde_J=1000`, and optionally `direction_seed=17`. The independent Haar
projection count `J` retains its existing meaning.

The **projection-partition** method,
`projected_algorithm.projected_estimation(..., r=r)`, randomly permutes the d
coordinates and partitions them into J = d/r equal blocks of r coordinates, so
r must divide d and every coordinate lies in exactly one projection. Each
projected sample runs dense Algorithm 1 without its support cutting plane: all
r coordinates stay active, directions lie in V_{min(2s,r)}, and K/tol use
(r, min(r,s)) in place of (d,s). The projections share a block partition; the
coordinate permutation uses an independent child stream of the seed.
Because the blocks are disjoint, both aggregations keep the block estimates on
their selected support: least squares selects the s largest magnitudes, while
max minimizes the largest squared mass discarded from any one block.

For projection-partition, `aggregate_estimates` combines the projected means with `utils.cutting_plane`,
starting from the zero support and vector. Both objectives are squared losses:
`objective='least_squares'` minimizes the sum, and `objective='max'` minimizes
the maximum, subject to at most s nonzero coordinates. No hard thresholding or
big-M bounds are used in aggregation. Coordinates in no projection remain zero.
For least squares the binary-support value is affine, with slope
`-sums_i**2/counts_i`. For max, a fixed-support convex QCP supplies simplex dual
weights; weighted least squares gives the same form of valid affine lower cut.
Both fixed-support wrappers return `(mu, L, U, gradient)`, separating the cut's
lower value from the feasible estimate's objective upper bound.

Aggregation stops when `gap <= max(aggregation_tol, aggregation_rtol * abs(U))`,
where U is the incumbent objective. Both tolerances default to `1e-5`; set
`aggregation_rtol=0` for the previous absolute rule. The inner max certificate
uses one quarter of this mixed tolerance at its own U. These tolerances are
separate from the dense-stage `tol`. The shared `cutting_plane` keeps `rtol=0`
by default, so Algorithm 1 and brute-force retain their absolute stopping rule.
`max_iter=1000` limits the aggregation support search.
`info` reports aggregation `objective`, `lower_bound`, `gap`, `iterations`,
`converged`, `tol`, `rtol`, `gap_tolerance`, and the incumbent binary `support`;
`projection_tol` records the dense-stage tolerance. An outer iteration limit
returns `converged=False`; an uncertified inner solve raises.

After max aggregation converges, `refine_max_estimate` minimizes the summed
squared loss on that same support, with each projected loss bounded by the
incumbent's actual maximum. Feasibility allows `1e-9 * max(1, T)`, where T is
that maximum, matching the normalized solver precision. The secondary problem
has a unique optimum on covered coordinates of the selected support; it does
not break ties between different supports. Uncovered
coordinates remain zero. The unconstrained coordinate averages are used directly
when feasible; otherwise convex QPs with tangent cuts solve the constrained
problem, avoiding barrier degeneracy at the max optimum. Feasible interpolation
preserves that cap within the stated allowance, and QP lower bounds certify the
secondary sum-loss gap to `1e-5 * max(1, sum_loss)` (the helper's `tol`).
`refined`, `objective_before_refinement`, and `refinement_max_slack` record this
step, and the final objective/gap are recomputed from the returned estimate.
Runtime includes refinement.
Add projections to experiments with `--r 5` (r must divide `--d`); compare the objectives with
`--projected-objectives least_squares max`. Use `--aggregation-tol`, `--aggregation-rtol`, and
`--aggregation-max-iter` to control only the aggregation search.

The **Haar-projection** method,
`haar_projected_algorithm.haar_projected_estimation(..., r=r, J=J)`, draws J
independent Gaussian d-by-r matrices and applies thin QR with normalized signs.
The resulting r-by-d matrices have orthonormal rows and uniformly distributed
row spaces. J is a required independent input: r need not divide d, and J*r may
be smaller than d. Projection randomness uses a separate seeded stream from the
sample blocks. Each projected sample runs dense Algorithm 1 with the full
direction sphere V_r and no sparsity constraint on its mean. Its block count is
the smallest odd integer at least `C * max(r, epsilon*n, log(J/delta))`; it raises
if that count exceeds n. The original covariance bound remains valid because
each projection has operator norm one.

Global LS or max aggregation imposes sparsity in the original d coordinates
through `utils.cutting_plane`. For each support, a convex QP/QCP supplies the
estimate, objective bounds, and box-dual cut. The data-driven box is
`M_i = abs(original_coordinate_MoM_i) + U_det + margin`, where U_det is the
median top-min(2s,d) residual norm around the original hard-thresholded MoM.
The original data use the original sparse block rule for this initialization.
This box is part of the Haar estimator's definition; it is not certified to
contain every optimizer of the unboxed aggregation problem. Sparse recovery also
requires the projections to distinguish sparse vectors; arbitrary J is accepted
without asserting that property. Max aggregation refines the summed loss on
its selected support while preserving the attained maximum within numerical
tolerance. Dense-stage and aggregation tolerances default to 1e-5.

Run Haar projections with both objectives and the three inexpensive baselines:

```sh
conda run -n robust_ip_estimation python experiments.py \
    --n 200 --d 8 --s 2 --epsilon 0.01 --nu 5 --seed 42 \
    --projection-method haar --r 3 --J 4 \
    --projected-objectives least_squares max \
    --methods projected coordinate_mom geometric_mom sample_mean
```

The corresponding Python options are `projection_method="haar", r=3, J=4`.
`--methods` selects only the named methods; omitting it preserves the original
runner defaults. Existing partition result keys remain `projected_algorithm_1`
and `projected_algorithm_1_max`; Haar results use `haar_projected_algorithm_1`
and `haar_projected_algorithm_1_max`.

The covering is constructive: for k=min(2s,d), a grid of spacing
2*radius/sqrt(k) gives rounding error at most radius. All grid points within
radius 1+radius are projected onto the unit ball, which cannot increase the
distance to any point in that ball. All supports of size at most k are included.
This is a covering guarantee, not a claim of minimum net cardinality. Net size
is combinatorial; `max_net_points` (default 200,000) stops oversized allocations
without silently substituting random directions or changing the radius.
The experiment runner records such a baseline as `skipped`, reports the full
net size, and continues the other estimators. A skipped run has no error or
runtime measurement; other solver failures still raise.
Use `--max-net-points` to raise the allocation limit when memory allows it.
For nets above 200,000 points, each fixed-support LP keeps the minimum and
maximum directional medians for every distinct projection onto its active
coordinates. This preserves the fixed-support objective while avoiding millions
of redundant LP constraints; net creation and all projections remain timed.

Run a single comparison with:

```sh
conda run -n robust_ip_estimation python experiments.py \
    --n 200 --epsilon 0.05 --nu 5 --s 2 --delta 0.05 \
    --d 4 --scale 1 --seed 42
```

`experiments.py` generates a sparse-mean multivariate t sample and applies
`adversarial_sparse_contamination` once, then runs the net baseline, Algorithm 1,
coordinate-wise MoM with top-s hard thresholding, and the sample mean with top-s hard thresholding on
the same data. The sparse estimators use the same K rule and block seed.
Dimension is required via `--d` (or `--dim`).
Each experiment draws one `loc` from Uniform(1, 3) and uses it on all s randomly
chosen active coordinates. The seed controls loc, support, data, and block
partition, so the same seed reproduces a run. The optional defaults are scale=1,
seed=42, C=2, delta=0.05, and strength=30; n, epsilon, nu, s, and d are required.
Use `--strength` to set the adversarial contamination strength. Here `scale` is a scalar:
the t shape matrix is `scale * I_d`, so for finite nu > 2 the experiment sets
`lambda_upper = 2 * nu / (nu - 2) * scale` using the clean covariance.
Use `--tol` to override the optimization tolerance, with
`0 < tol <= sqrt(K*lambda_upper/n)`. Algorithm 1 uses inner tolerance `tol/4`
and separation tolerance `tol/8`; the coordinate-wise MoM estimate is unaffected.
The CLI and `run_experiment` default to `tol=1e-5` for the compared methods.
Algorithm 1 and the projected estimator also use `1e-5` when `tol=None`.
Older results retain the C and tolerance settings used for those runs.

Use `--net-radius` (Python: `net_radius`) to change only the brute-force net's
covering radius, with `0 < net_radius <= 1`. Increasing it reduces the grid size
and weakens the covering guarantee. Radius 1 does not preserve the draft's
1/4-net guarantee; the implementation still includes +/- coordinate directions
and the full projected lattice for the chosen radius.

The output contains L2 error, support recovery (fraction of true active
coordinates with estimated magnitude above 1e-8), and runtime in seconds.
Runtime includes each estimator's computation, including the net, projections,
and optimization where applicable, and excludes shared data generation and metric
calculation. The `sample_mean` method keeps the s largest absolute sample mean
coordinates and sets the others to zero; its runtime includes hard thresholding
and its support recovery is computed in the same way as the other sparse methods.
Nonconverged runs raise an error. For Python use,
`experiments.run_experiment(n, epsilon, nu, s, delta, d=d, **options)` returns a dict
keyed by method, with `error`, `support_recovery`, and `runtime` for each method.

Run correctness checks with `python -m unittest discover -v`. The checks include
exhaustive support/(S,B) reference models and analytic distance-to-box examples
that validate LP/SOCP objectives and dual cutting planes.

`results.md` records the experiments in order. `plot_histograms.py` draws a
figure's per-method L2 error histograms from a per-seed errors JSON (format in its
docstring), e.g. `python plot_histograms.py experiment1_errors.json`; each method
keeps a fixed color across figures.


## One-sided versus two-sided Eq. (10) benchmark (2026-09-30)

Adding `b_k=0 => (block_mean[k]-mu) @ v <= G` preserves the oracle optimum:
with exactly `(K+1)/2` selected blocks, the selected-block lower inequalities
already maximize the signed median. The added implication constrains the
nonselected blocks and can alter relaxations and the search tree.

This comparison changes only that implication within each pair. It uses the
current production source without editing it: the benchmark constructs a
temporary function variant and instruments both sides identically. For big-M,
the additional inequality is `projection <= G + row_bound[k] * b_k`, valid
because `G >= 0`. Warm starts, cutoff, perspective setting, tolerances, data,
and solver thread count remain identical within each pair.

The suite has 18 standalone setups (d=2..20, s=1..6, K=5..31), five seeds and
three serial repeats, with alternating execution order. It includes Gaussian,
correlated, t5, t2.5, skewed, block-outlier, shifted-center and tied data. Exact
oracle tolerance is 1e-5. Decision-mode thresholds are the same heuristic witness
plus max(0.005, 5% of its value), with the same feasible warm start. Both indicator
and big-M are tested; four sparse setups also use perspective. Standalone
block-outlier percentages describe corrupted block means, not raw samples.

All 2,400 oracle solves returned certified results and the paired optimum
intervals overlapped. All 240 end-to-end hybrid runs converged. Independent
one-sided oracle checks passed for 150 returned sparse/local projected candidates;
repeated runs reproduced estimates and reported objectives within 1e-10.
The 3-second per-oracle and 20-second estimator budgets were never hit. Gurobi
13.0.3 used one thread. Cold license startup and independent verification are
outside the timed interval; model construction and result extraction are included.
A sandbox WLS renewal failure required rerunning the affected standalone cases
and the initial end-to-end attempt with network access. These environmental
failures are documented in the JSON metadata and are not solver timings.

Times below are averages of each seed's three-run median. Speedup means
one-sided time / two-sided time, so values below one favor the original model.

| Standalone setup | d | s | K | One-sided indicator (ms) | Two-sided indicator (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_gaussian | 2 | 2 | 5 | 3.57 | 4.21 | 0.85x |
| sparse_gaussian | 6 | 1 | 7 | 11.07 | 13.64 | 0.81x |
| sparse_t5 | 8 | 2 | 9 | 33.66 | 33.97 | 0.99x |
| wide_sparse | 12 | 1 | 9 | 38.24 | 36.16 | 1.06x |
| less_sparse | 8 | 3 | 9 | 28.17 | 30.08 | 0.94x |
| more_blocks | 6 | 2 | 15 | 73.66 | 87.60 | 0.84x |
| dense_many_blocks | 3 | 3 | 25 | 34.74 | 50.92 | 0.68x |
| correlated | 8 | 2 | 9 | 21.15 | 23.65 | 0.89x |
| heavy_tail | 8 | 2 | 9 | 23.95 | 26.05 | 0.92x |
| outlier_20pct | 8 | 2 | 15 | 75.63 | 81.99 | 0.92x |
| outlier_40pct | 6 | 2 | 15 | 28.48 | 35.37 | 0.81x |
| displaced_center | 8 | 2 | 9 | 26.39 | 33.46 | 0.79x |
| skewed | 6 | 2 | 9 | 25.98 | 32.97 | 0.79x |
| larger_sparse | 12 | 2 | 21 | 611.91 | 667.74 | 0.92x |
| large_K | 8 | 2 | 31 | 1924.95 | 2032.35 | 0.95x |
| wide_d20 | 20 | 2 | 15 | 616.30 | 691.10 | 0.89x |
| dense_d6_large_K | 6 | 6 | 31 | 528.98 | 634.29 | 0.83x |
| ties_near_zero | 6 | 2 | 9 | 2.92 | 4.85 | 0.60x |

Across the 90 setup/seed pairs, the geometric mean time increase from adding
the second implication was 18.2% for exact indicator and 17.2% for exact big-M.
With decision stopping it was 32.7% and 29.9%, respectively. Some individual
seeds improved: 21/90 for exact indicator and 19/90 for exact big-M. Only 1/18
indicator setup averages and 3/18 big-M setup averages improved, by about 3-6%.
Perspective did not reverse the conclusion in its four-case subset.
For Gaussian d=8,s=2,K=31, average nodes decreased from 19,226 to 17,022,
but average time increased from 1.925 to 2.032 seconds and solver Work increased.
Reducing node count did not compensate for the extra per-node/model work.

End-to-end runs use n=160, raw-sample replacement contamination 0-3%, hybrid
separation with indicator, and five seeds times three repeats. All tolerance
settings and blocks are paired. Sparse/dense-stage tol=0.01 except
`ip_tight_tol`, which uses 1e-5; aggregation retains its defaults.

| End-to-end setup | One-sided (s) | Two-sided (s) | Speedup |
| --- | ---: | ---: | ---: |
| ip_normal | 0.0581 | 0.0634 | 0.92x |
| ip_t5 | 0.1020 | 0.1030 | 0.99x |
| ip_correlated | 0.0840 | 0.0884 | 0.95x |
| ip_heavy_tail | 0.1762 | 0.1776 | 0.99x |
| ip_tight_tol | 0.0996 | 0.1085 | 0.92x |
| projection_r2 | 0.1266 | 0.1448 | 0.87x |
| projection_r4 | 0.1643 | 0.2024 | 0.81x |
| projection_max | 0.1177 | 0.1360 | 0.87x |

All eight end-to-end setup averages were unchanged or slower with the second
implication; t5 and heavy-tail differences were under 2%. Projection cases were
14-23% slower. Individual-seed improvements remain possible (12/40 pairs),
so this is evidence for retaining the current default on these toys, not a claim
that the extra constraint can never help. Production solver code is unchanged.

The [comparison figure](benchmarks/two_sided/comparison.png) shows every seed.
Raw [oracle results](benchmarks/two_sided/oracle_results.json),
[end-to-end results](benchmarks/two_sided/e2e_results.json), and
[summary](benchmarks/two_sided/summary.json) include bounds, nodes, work, call
counts, configurations, and source fingerprints. Reproduce with the
`robust_ip_estimation` environment (a working Gurobi license connection is needed):

```sh
python benchmarks/two_sided/benchmark_two_sided.py --repo . \
  --stage oracle --oracle-modes exact decision --perspective \
  --output /tmp/two_sided_oracle.json
python benchmarks/two_sided/benchmark_two_sided.py --repo . \
  --stage e2e --output /tmp/two_sided_e2e.json
python benchmarks/two_sided/summarize_two_sided.py \
  --oracle /tmp/two_sided_oracle.json --e2e /tmp/two_sided_e2e.json --outdir /tmp
```


## Random block selection

`random_block_selection.py` constructs the same balanced, seeded K blocks as
Algorithm 1, then chooses a uniform subset without replacement. The same C
(default 2) scales the full collection and the requested subset:

$$
K_{\mathrm{sub,requested}}=\operatorname{oddceil}\left(
C\max\{s\log(ed/s),\log(4/\delta)\}\right),\qquad
K_{\mathrm{sub}}=\min(K,K_{\mathrm{sub,requested}}).
$$

Logs are natural, and oddceil rounds up to the smallest odd integer.
At full support, s*log(e*d/s)=d. The original K rule is unchanged: it uses
d at full support and s*log(d/s) otherwise. Each selected block is used once;
selected rows are restored to their original order. If K_sub=K, the problem
equals full Algorithm 1. Initialization and optimization use the subset only.

```python
estimate, info = random_block_selection(
    data, s=5, epsilon=.01, lambda_upper=739.0987715913293,
    delta=.05, seed=0, C=3, replace=False,
)
print(info["K"], info["K_sub_requested"], info["K_sub"])
```

For n=1000, d=s=5, epsilon=.01 and delta=.05, C=1,2,3,4,5 gives
K=11,21,31,41,51 and K_sub=5,11,15,21,25. `block_selection_seed` defaults
to seed in an independent SeedSequence stream. The subset is chosen once
per fit. Changing C also changes the full partition, so subsets across C
are not claimed to be nested. `subsample_multiplier` is retained as a
compatibility alias, but an explicit value must equal C.

```bash
python experiments.py --n 1000 --d 5 --s 5 --nu 3 --epsilon .01 \
    --delta .05 --seed 0 --methods random_block_selection --C 3
```

Tolerance defaults to 1e-5; Algorithm 1 options and heuristics apply.
Runtime includes full block construction, selection and optimization.
`replace=True` permits independent draws with replacement and no cap using
the same new requested-count formula. Old experiment directories retain
their frozen sources and historical subset rules. This method remains opt-in.
