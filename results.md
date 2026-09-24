# Experiments

Experiments setup: multivariate $t_3$ with shape $I_{20}$ and a 5-sparse mean. Each seed draws one amplitude from Uniform$(1,3)$ and a random support. Seeds 0–9 use the same clean samples in both experiments. Defaults: $C=2$, $\delta=0.05$, attack strength $30$, $\overline\lambda=6$, and $e_{\mathrm{tol}}=\sqrt{K\overline\lambda/n}/4$. Runtimes include estimator preprocessing but exclude shared data generation; up to four one-thread runs were processed concurrently. Support recovery is the fraction of true active coordinates selected; it is not defined for the dense sample mean. Both figures use identical 0.5-wide bins and x/y limits.

## Experiments 1

Results for $(d, s, \delta, \epsilon, n) = (20, 5, 0.05, 0, 100)$; $K=15$:

Brute-force skipped: radius-1 net 408,076,993 directions; 65.3 GB array > 16 GB RAM.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery |
| --- | ---: | ---: | ---: |
| Algorithm 1 | 0.443619 | 15.077854 | 1.000 |
| Coordinate-wise MoM | 0.366593 | 0.000254 | 1.000 |
| Sample mean | 0.892022 | 0.000007 | — |

![L2 error histograms for Experiment 1](experiment1_error_histograms.png)

## Experiments 2

Results for $(d, s, \delta, \epsilon, n) = (20, 5, 0.05, 0.1, 100)$; $K=21$:

Brute-force skipped: same 408,076,993-direction, 65.3 GB radius-1 net.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery |
| --- | ---: | ---: | ---: |
| Algorithm 1 | 1.865050 | 291.855069 | 0.780 |
| Coordinate-wise MoM | 2.078442 | 0.000290 | 0.700 |
| Sample mean | 2.878923 | 0.000007 | — |

![L2 error histograms for Experiment 2](experiment2_error_histograms.png)
