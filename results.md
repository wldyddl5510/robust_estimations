# Experiments

Experiments setup:
* Clean target distribution: multivariate $t_{\nu}$ with $s$-sparse mean, $\nu$ degrees of freedom, and shape matrix $I$.
* Adding $\epsilon$ portion of the contamination.
* Parameters for the experiments: 
    * $n$: number of samples.
    * $s$: sparsity level.
    * $\delta$: confidence level.
    * $\epsilon$: contamination level.
    * $d$: dimension. Should be $2s \leq d$.
    * $\overline \lambda$: Temporarily set $2 \lambda_{\max}(\Sigma)$.
* The target mean is generated using the following procedure:
    * Sample $a \sim \text{Uniform}(1,3)$ one time.
    * Sample $s$ coordinates randomly from $1, \dots, d$.
    * Set the $s$-coordinates $a$ and the rest of the coordinates $0$.
* For the brutal-force algorithm (comparing all discretized covering nets), used the radius of the net by $1$, which has to be $< \sqrt{2}$. Smaller nets will make the procedure more accurate, but note the net size increases exponentially. For example, if we set the radius to be $1/4$, total number of net directions in $d = 10$ and $s = 4$ is 484328693, requiring 38.75 GB memory and LP constraints $\approx 969000000$. So the Brutal-force method is not very scalable.
* Runtime includes each estimator's preprocessing, net construction and projections where applicable, and optimization. It excludes shared sample generation, metric calculation, and result printing. Support recovery is the fraction of true active coordinates whose estimated magnitude exceeds 1e-8. The radius-1 net has a weaker covering guarantee than the draft's radius-1/4 net.

## Experiments 1

The location magnitude drawn in both runs was 1.749080237694725. The true supports (zero-based coordinates) were (3, 9) for s = 2 and (0, 3, 9) for s = 3. Both runs used K = 5 blocks of 40 samples, with the same block partition. No observations were replaced. The adversarial strength parameter was set to 5 but has no effect when epsilon = 0.

The outer tolerance was the default e_tol = sqrt(K * lambda_upper / n) / 100 = 0.0028867513459481294 in both runs. Algorithm 1 used inner tolerance e_tol / 4 and separation tolerance e_tol / 8. These settings are the default 1/100 tolerance scale for future experiments.

Results for $(d, s, \delta, \epsilon, n) = (10, 2, 0.05, 0.0, 200)$:

| Method | L2 error | Support recovery | Runtime (s) |
| --- | ---: | ---: | ---: |
| Brute-force, radius 1 | 0.241929416 | 1.000000 | 0.079839666 |
| Algorithm 1 | 0.163102700 | 1.000000 | 0.201522750 |
| Coordinate-wise MoM with hard thresholding | 0.241929416 | 1.000000 | 0.000142750 |

Results for $(d, s, \delta, \epsilon, n) = (10, 3, 0.05, 0.0, 200)$:

| Method | L2 error | Support recovery | Runtime (s) |
| --- | ---: | ---: | ---: |
| Brute-force, radius 1 | 0.279062020 | 1.000000 | 0.168647875 |
| Algorithm 1 | 0.172566815 | 1.000000 | 0.238805834 |
| Coordinate-wise MoM with hard thresholding | 0.279062020 | 1.000000 | 0.000126667 |

All methods recovered the true support in these runs. Both optimization methods converged in two outer iterations. The brute-force net had 4,561 points for s = 2 and 29,305 points for s = 3; its returned estimate matched the MoM initialization in both cases. 

Reproduce the runs in the `robust_ip_estimation` conda environment:

```sh
conda run -n robust_ip_estimation python experiments.py \
    --n 200 --epsilon 0 --nu 5 --s 2 --delta 0.05 \
    --dim 10 --scale 1 --seed 42 --C 1 --strength 5 --net-radius 1

conda run -n robust_ip_estimation python experiments.py \
    --n 200 --epsilon 0 --nu 5 --s 3 --delta 0.05 \
    --dim 10 --scale 1 --seed 42 --C 1 --strength 5 --net-radius 1
```
