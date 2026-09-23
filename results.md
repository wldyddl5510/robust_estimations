# Clean-data comparison at d = 10

On September 23, 2026, we ran each estimator once for sparsity s = 2 and s = 3, with contamination rate epsilon = 0. All runs used n = 200, a 10-dimensional t distribution with 5 degrees of freedom and shape matrix I, confidence parameter delta = 0.05, seed 42, and block multiplier C = 1. The clean covariance is (5/3)I, so the supplied bound was lambda_upper = 10/3. The brute-force method used a covering-net radius of 1.

The location magnitude drawn in both runs was 1.749080237694725. The true supports (zero-based coordinates) were (3, 9) for s = 2 and (0, 3, 9) for s = 3. The centered samples were identical up to floating-point error (maximum difference 8.88e-16). Both runs used K = 5 blocks of 40 samples, with the same block partition. No observations were replaced. The adversarial strength parameter was set to 5 but has no effect when epsilon = 0.

The outer tolerance was the default e_tol = sqrt(K * lambda_upper / n) / 100 = 0.0028867513459481294 in both runs. Algorithm 1 used inner tolerance e_tol / 4 and separation tolerance e_tol / 8. These settings are the default 1/100 tolerance scale for future experiments.

Results for $(d, s, \delta, \epsilon, n) = (10, 2, 0.05, 0.0, 200)$:

| Method | L2 error | Support recovery | Runtime (s) |
| ---: | --- | ---: | ---: | ---: |
| Brute-force, radius 1 | 0.241929416 | 1.000000 | 0.079839666 |
| Algorithm 1 | 0.163102700 | 1.000000 | 0.201522750 |
| Coordinate-wise MoM with hard thresholding | 0.241929416 | 1.000000 | 0.000142750 |

Results for $(d, s, \delta, \epsilon, n) = (10, 3, 0.05, 0.0, 200)$:

| Method | L2 error | Support recovery | Runtime (s) |
| ---: | --- | ---: | ---: | ---: |
| Brute-force, radius 1 | 0.279062020 | 1.000000 | 0.168647875 |
| Algorithm 1 | 0.172566815 | 1.000000 | 0.238805834 |
| Coordinate-wise MoM with hard thresholding | 0.279062020 | 1.000000 | 0.000126667 |

All methods recovered the true support in these runs. Both optimization methods converged in two outer iterations. The brute-force net had 4,561 points for s = 2 and 29,305 points for s = 3; its returned estimate matched the MoM initialization in both cases. Algorithm 1 reduced the MoM L2 error by about 33% for s = 2 and 38% for s = 3. These are single-seed results.

Runtime includes each estimator's preprocessing, net construction and projections where applicable, and optimization. It excludes shared sample generation, metric calculation, and result printing. Support recovery is the fraction of true active coordinates whose estimated magnitude exceeds 1e-8. The radius-1 net has a weaker covering guarantee than the draft's radius-1/4 net.

Reproduce the runs in the `robust_ip_estimation` conda environment:

```sh
conda run -n robust_ip_estimation python experiments.py \
    --n 200 --epsilon 0 --nu 5 --s 2 --delta 0.05 \
    --dim 10 --scale 1 --seed 42 --C 1 --strength 5 --net-radius 1

conda run -n robust_ip_estimation python experiments.py \
    --n 200 --epsilon 0 --nu 5 --s 3 --delta 0.05 \
    --dim 10 --scale 1 --seed 42 --C 1 --strength 5 --net-radius 1
```
