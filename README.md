# Robust estimation via Integer Programming

Use the `robust_ip_estimation` conda environment and install dependencies with
`python -m pip install -r requirements.txt`.

The net baseline evaluates a deterministic Euclidean net (radius 1/4 by default),
solves each fixed-support LP with Gurobi, and searches supports with the
draft's dual cutting planes and a Gurobi MILP. Gurobi requires a working license.
All optimization programs, including the test reference models, use Gurobi.
SciPy is used for t-distribution sampling and a nearest-neighbor check of the net.

The shared outer loop is `utils.cutting_plane`. Each estimator supplies a
`solve_support(z)` callback returning `(mu, L, U, gradient)`: a feasible estimate,
the cut value, its objective upper bound, and the cut slope. The helper maintains
the incumbent, accumulates cuts, solves the support MILP, and checks the global
gap. Separate L and U also allow Algorithm 1 to use an approximate inner oracle.

`IP_algorithm.ip_estimation` implements Algorithm 1: Gurobi solves both the mixed
integer conic separation problem (10) and the restricted SOCP (17).
Generated (S,B) constraints are retained across support searches.
The oracle uses an absolute gap, and the inner stopping condition uses its
global upper bound: `upper_F - L <= tol/4`, with separation tolerance `tol/8`.
The programs have separate wrappers: `brute_force.solve_fixed_support_lp`,
`IP_algorithm.separation_oracle`, and `IP_algorithm.solve_restricted_socp`.
The SOCP uses convex quadratic cone constraints and `QCPDual=1` to obtain the
linear box-constraint duals. For both LP and SOCP, the cut gradient is
`M * (upper.Pi + lower.Pi)`, using Gurobi's nonpositive duals for <= constraints
in a minimization problem (see the
[Gurobi Pi documentation](https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/constraintlinear.html#pi)).
Models use one Gurobi thread.

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
matrix eigenvalue. The default block count uses C=1 and the default absolute
stopping tolerance is `sqrt(K * lambda_upper / n) / 100`. Blocks are balanced and use
all samples. Use the same block seed for both estimators. If the required odd
K exceeds n, the function raises an error instead of changing the rule.
Both estimators call `utils.mom_initialization` to share exactly this setup.
The multiplier is configurable with the keyword argument `C` in all three
estimators and `run_experiment`, or with `experiments.py --C 1`.
K is the smallest odd integer at least `C * max(s*log(d/s), epsilon*n, log(1/delta))`.

`info` reports convergence, the net objective, its global lower bound and gap,
iterations, K, net size, and end-to-end runtime in seconds (including net
construction and projections). A run reaching `max_iter` returns
`converged=False`. The gap concerns the finite-net objective, within solver
tolerances; it is not a certificate for the continuous supremum.

For Algorithm 1, `info['objective']` is an upper bound for the returned
estimate's continuous-direction objective, and the gap uses that bound.
Its info additionally reports oracle calls, total inner solves, and the number
of generated (S,B) constraints. An inner iteration limit or an uncertified
solver result raises an error; an outer iteration limit returns
`converged=False`. Runtime includes initialization and all oracle/SOCP/MILP work.
The separation model has 2*d+K+1 variables. On 2026-09-23, after installing the
new license at `~/gurobi.lic`, Gurobi 13.0.3 in `robust_ip_estimation` solved
a 201-variable quadratic model and a 2,001-variable linear model to optimality.
The previous size-limited-license restrictions no longer blocked these checks;
no `GRB_LICENSE_FILE` environment variable was needed.

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
and coordinate-wise MoM with top-s hard thresholding on the same data. All three
use the same K rule and block seed. Dimension is required via `--d` (or `--dim`).
Each experiment draws one `loc` from Uniform(1, 3) and uses it on all s randomly
chosen active coordinates. The seed controls loc, support, data, and block
partition, so the same seed reproduces a run. The optional defaults are scale=1,
seed=42, C=1, and strength=3; n, epsilon, nu, s, delta, and d are required.
Use `--strength` to set the adversarial contamination strength. Here `scale` is a scalar:
the t shape matrix is `scale * I_d`, so for finite nu > 2 the experiment sets
`lambda_upper = 2 * nu / (nu - 2) * scale` using the clean covariance.
Use `--tol` to override the optimization tolerance, with
`0 < tol <= sqrt(K*lambda_upper/n)`. Algorithm 1 uses inner tolerance `tol/4`
and separation tolerance `tol/8`; the coordinate-wise MoM estimate is unaffected.
The default is now `sqrt(K*lambda_upper/n)/100`; older results record the
different C and tolerance settings used for those runs.

Use `--net-radius` (Python: `net_radius`) to change only the brute-force net's
covering radius, with `0 < net_radius <= 1`. Increasing it reduces the grid size
and weakens the covering guarantee. Radius 1 does not preserve the draft's
1/4-net guarantee; the implementation still includes +/- coordinate directions
and the full projected lattice for the chosen radius.

The output contains L2 error, support recovery (fraction of true active
coordinates with estimated magnitude above 1e-8), and runtime in seconds.
Runtime includes each estimator's block construction and optimization, including
the net and projections, and excludes shared data generation and metric
calculation. Nonconverged runs raise an error. For Python use,
`experiments.run_experiment(n, epsilon, nu, s, delta, d=d, **options)` returns a dict
keyed by method, with `error`, `support_recovery`, and `runtime` for each method.

Run correctness checks with `python -m unittest discover -v`. The checks include
exhaustive support/(S,B) reference models and analytic distance-to-box examples
that validate LP/SOCP objectives and dual cutting planes.

`results.md` records the clean-data comparison at d=10, epsilon=0, and s=2 or 3.
