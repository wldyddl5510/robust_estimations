# Experiments

Experiments 1–7 use a multivariate $t_\nu$ with shape $I_d$ and an $s$-sparse mean. Experiments 8–11 use a centered skew-$t_\nu$ with an $s$-sparse mean. Each seed draws one amplitude from Uniform$(1,3)$ and a random support. For fixed $(n,d,s,\nu)$, seeds 0–9 use the same clean samples across contamination settings. Experiments 1–5 set $\nu=3$, $\overline\lambda=6$, and $e_{\mathrm{tol}}=\sqrt{K\overline\lambda/n}/4$; Experiments 6–7 set $\nu=2.5$, $\overline\lambda=10$, and $e_{\mathrm{tol}}=10^{-5}$. Shared defaults: $C=2$, $\delta=0.05$, attack strength $30$. Geometric MoM + HT takes the Euclidean geometric median of the same $K$ block means and retains its $s$ largest-magnitude coordinates ([Minsker and Strawn](https://arxiv.org/abs/2307.03111)). Runtimes include estimator preprocessing but exclude shared data generation; up to four one-thread runs were processed concurrently. Support recovery is the fraction of true active coordinates selected; it is not defined for the dense sample mean. Within each figure, all plotted methods use identical bins and x/y limits; the limits may differ between experiments.

## Experiments 1

Results for $(d, s, \delta, \epsilon, n) = (20, 5, 0.05, 0, 100)$; $K=15$:

Brute-force skipped: radius-1 net 408,076,993 directions; 65.3 GB array > 16 GB RAM.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery |
| --- | ---: | ---: | ---: |
| Algorithm 1 | 0.443619 | 15.077854 | 1.000 |
| Projected Algorithm 1 (LS) | 0.329578 | 1.305803 | 1.000 |
| Projected Algorithm 1 (max) | 0.323841 | 1.066333 | 1.000 |
| Coordinate-wise MoM | 0.366593 | 0.000254 | 1.000 |
| Geometric MoM + HT | 0.287253 | 0.000779 | 1.000 |
| Random MoM (tilde J=10) | 0.439761 | 0.026397 | 1.000 |
| Random trimmed mean (tilde J=10) | 0.349879 | 0.025899 | 1.000 |
| Random MoM (tilde J=20) | 0.462883 | 0.052925 | 1.000 |
| Random trimmed mean (tilde J=20) | 0.357993 | 0.026268 | 1.000 |
| Random MoM (tilde J=50) | 0.627385 | 0.026494 | 1.000 |
| Random trimmed mean (tilde J=50) | 0.408631 | 0.025924 | 1.000 |
| Random MoM (tilde J=100) | 0.612289 | 0.028904 | 1.000 |
| Random trimmed mean (tilde J=100) | 0.422202 | 0.026316 | 1.000 |
| Random MoM (tilde J=1000) | 0.457077 | 0.076217 | 1.000 |
| Random trimmed mean (tilde J=1000) | 0.360803 | 0.064885 | 1.000 |
| Sample mean | 0.892022 | 0.000007 | — |

Projected methods: $J=10$, $r=10$, projected $K=7$; support cutting-plane aggregation with absolute/relative tolerances $10^{-5}$ and same-support LS refinement for max. Both completed seeds 0–9 with full coordinate coverage; four concurrent one-thread runs. Existing baseline rows retain their recorded results. Per-seed errors and metrics: `experiment1_errors.json`, `experiment1_projected_results.json`.

Random MoM and random trimmed mean: both completed seeds 0–9 with $\widetilde J=10$ random sparse directions plus 20 coordinate directions (30 total), and $e_{\mathrm{tol}}=10^{-5}$. Both use the same direction seed as the data seed, hence the same direction bank per seed; historical sample hashes and truths were verified. Random MoM uses $K=15$ and random trimmed mean uses $k=4$ ($n-2k=92$). These gaps certify the finite direction bank objective. Runs were sequential with one solver thread; runtimes include estimator preprocessing but exclude data generation. The historical Algorithm 1 tolerance is approximately 0.237171, so its tolerance and timing conditions differ. Per-seed estimates, bounds, metrics, settings, and source hashes: [experiment1_random_j10_results.json](experiment1_random_j10_results.json).

Random MoM and random trimmed mean: both completed seeds 0–9 with $\widetilde J=20$ random sparse directions plus 20 coordinate directions (40 total), and $e_{\mathrm{tol}}=10^{-5}$. Both use the same direction seed as the data seed, hence the same direction bank per seed; historical sample hashes and truths were verified. Random MoM uses $K=15$ and random trimmed mean uses $k=4$ ($n-2k=92$). These gaps certify the finite direction bank objective. Runs were sequential with one solver thread; runtimes include estimator preprocessing but exclude data generation. The historical Algorithm 1 tolerance is approximately 0.237171, so its tolerance and timing conditions differ. Per-seed estimates, bounds, metrics, settings, and source hashes: [experiment1_random_j20_results.json](experiment1_random_j20_results.json). The J=20 single-pass MoM mean includes a 0.281 s first-run outlier. A warmed alternating J=10/20 benchmark (five repeats per seed, 50 fits per method and J) gives MoM means 0.024661 vs 0.025081 s (+1.7%), and trimmed means 0.025000 vs 0.025296 s (+1.2%). Medians remain about 0.026 s; these small differences do not establish a scaling trend. Measurements: [experiment1_random_j10_j20_timing.json](experiment1_random_j10_j20_timing.json).

Random MoM and random trimmed mean: both completed seeds 0–9 with $\widetilde J=50$ random sparse directions plus 20 coordinate directions (70 total), and $e_{\mathrm{tol}}=10^{-5}$. Both use the same direction seed as the data seed, hence the same direction bank per seed; historical sample hashes and truths were verified. Random MoM uses $K=15$ and random trimmed mean uses $k=4$ ($n-2k=92$). These gaps certify the finite direction bank objective. Runs were sequential with one solver thread; runtimes include estimator preprocessing but exclude data generation. The historical Algorithm 1 tolerance is approximately 0.237171, so its tolerance and timing conditions differ. Per-seed estimates, bounds, metrics, settings, and source hashes: [experiment1_random_j50_results.json](experiment1_random_j50_results.json). A warmed benchmark rotates J=10/20/50 order with five repeats per seed (50 fits per method and J). MoM mean runtimes for J=10/20/50 are 0.024633/0.024125/0.026534 s; trimmed mean runtimes are 0.025342/0.024968/0.025729 s. J=50 versus J=10 changes the means by +7.7% (MoM) and +1.5% (trimmed). These small timing differences do not establish a scaling trend. Measurements: [experiment1_random_j10_j20_j50_timing.json](experiment1_random_j10_j20_j50_timing.json).

Random MoM and random trimmed mean: both completed seeds 0–9 with $\widetilde J=100$ random sparse directions plus 20 coordinate directions (120 total), and $e_{\mathrm{tol}}=10^{-5}$. Both use the same direction seed as the data seed, hence the same direction bank per seed; historical sample hashes and truths were verified. Random MoM uses $K=15$ and random trimmed mean uses $k=4$ ($n-2k=92$). These gaps certify the finite direction bank objective. Runs were sequential with one solver thread; runtimes include estimator preprocessing but exclude data generation. The historical Algorithm 1 tolerance is approximately 0.237171, so its tolerance and timing conditions differ. Per-seed estimates, bounds, metrics, settings, and source hashes: [experiment1_random_j100_results.json](experiment1_random_j100_results.json). A warmed benchmark rotates J=10/20/50/100 order with five repeats per seed (50 fits per method and J). MoM mean runtimes for J=10/20/50/100 are 0.025436/0.025478/0.025996/0.029750 s; trimmed mean runtimes are 0.025638/0.025407/0.025755/0.025687 s. J=100 versus J=10 changes the means by +17.0% (MoM) and +0.2% (trimmed). Measurements: [experiment1_random_j10_j20_j50_j100_timing.json](experiment1_random_j10_j20_j50_j100_timing.json).

Random MoM and random trimmed mean: both completed seeds 0–9 with $\widetilde J=10000$ random sparse directions plus 20 coordinate directions (1020 total), and $e_{\mathrm{tol}}=10^{-5}$. Both use the same direction seed as the data seed, hence the same direction bank per seed; historical sample hashes and truths were verified. Random MoM uses $K=15$ and random trimmed mean uses $k=4$ ($n-2k=92$). These gaps certify the finite direction bank objective. Runs were sequential with one solver thread; runtimes include estimator preprocessing but exclude data generation. The historical Algorithm 1 tolerance is approximately 0.237171, so its tolerance and timing conditions differ. Per-seed estimates, bounds, metrics, settings, and source hashes: [experiment1_random_j1000_results.json](experiment1_random_j1000_results.json). A warmed benchmark rotates J=10/100/1000 order with five repeats per seed (50 fits per method and J). MoM mean runtimes for J=10/100/1000 are 0.031607/0.037054/0.068891 s; trimmed mean runtimes are 0.029333/0.037969/0.063605 s. J=1000 versus J=10 runtime ratios are 2.18 (MoM) and 2.17 (trimmed). Measurements: [experiment1_random_j10_j100_j1000_timing.json](experiment1_random_j10_j100_j1000_timing.json).

![L2 error histograms for Experiment 1](experiment1_error_histograms.png)

## Experiments 2

Results for $(d, s, \delta, \epsilon, n) = (20, 5, 0.05, 0.1, 100)$; $K=21$:

Brute-force skipped: same 408,076,993-direction, 65.3 GB radius-1 net.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery |
| --- | ---: | ---: | ---: |
| Algorithm 1 | 1.865050 | 291.855069 | 0.780 |
| Coordinate-wise MoM | 2.078442 | 0.000290 | 0.700 |
| Geometric MoM + HT | 1.675652 | 0.000983 | 0.780 |
| Sample mean | 2.878923 | 0.000007 | — |

![L2 error histograms for Experiment 2](experiment2_error_histograms.png)

## Experiments 3

Results for $(d, s, \delta, \epsilon, n) = (30, 5, 0.05, 0, 150)$; $K=19$:

Brute-force skipped: radius-1 net 41,972,797,833 directions; 10.1 TB array > 16 GB RAM.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery |
| --- | ---: | ---: | ---: |
| Algorithm 1 | 0.246961 | 47.287453 | 1.000 |
| Projected Algorithm 1 (LS) | 0.523946 | 1.121061 | 0.980 |
| Projected Algorithm 1 (max) | 0.525952 | 1.039686 | 0.980 |
| Coordinate-wise MoM | 0.255429 | 0.000201 | 1.000 |
| Geometric MoM + HT | 0.216059 | 0.000498 | 1.000 |
| Sample mean | 0.782199 | 0.000011 | — |

Projected methods: $J=10$, $r=10$, projected $K=7$; support cutting-plane aggregation with absolute/relative tolerances $10^{-5}$ and same-support LS refinement for max. All six methods completed seeds 0–9; four concurrent one-thread runs. Uncovered true support coordinates (zero-based): seed 4: [2]. Per-seed errors and metrics: `experiment3_errors.json`, `experiment3_results.json`.

![L2 error histograms for Experiment 3](experiment3_error_histograms.png)

## Experiments 4

Results for $(d, s, \delta, \epsilon, n) = (30, 5, 0.05, 0.1, 150)$; $K=31$:

Brute-force skipped: radius-1 net 41,972,797,833 directions; 10.1 TB array > 16 GB RAM.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery |
| --- | ---: | ---: | ---: |
| Algorithm 1 (7/10 completed) | 1.740224 | 1001.043614 | 0.857 |
| Coordinate-wise MoM (10/10) | 1.686140 | 0.000163 | 0.820 |
| Geometric MoM + HT (10/10) | 1.740425 | 0.000778 | 0.800 |
| Sample mean (10/10) | 2.837271 | 0.000005 | — |

Algorithm 1 timed out after 30 minutes for seeds 2, 5, and 7, including solo retries. Its means use only the seven completed seeds and are not 10-seed estimates.

![L2 error histograms for Experiment 4](experiment4_error_histograms.png)

## Experiments 5

Results for $(d, s, \delta, \epsilon, n) = (30, 10, 0.05, 0.05, 150)$; $K=23$:

Brute-force skipped: radius-1 net 1,018,604,017,927,145 directions; 244 PB array > 16 GB RAM.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery |
| --- | ---: | ---: | ---: |
| Algorithm 1 (8/10 completed) | 0.696091 | 1024.151613 | 1.000 |
| Coordinate-wise MoM (10/10) | 0.789623 | 0.000124 | 1.000 |
| Geometric MoM + HT (10/10) | 0.689913 | 0.000661 | 1.000 |
| Sample mean (10/10) | 1.702656 | 0.000005 | — |

Algorithm 1 seeds 7 and 9 were stopped at the user's request after about 74 minutes and marked timeout. Its means use only the eight completed seeds and are not 10-seed estimates.

![L2 error histograms for Experiment 5](experiment5_error_histograms.png)

## Experiments 6

Results for $(d, s, \delta, \epsilon, n) = (20, 3, 0.05, 0.02, 1000)$; $\nu=2.5$, $K=41$, $\overline\lambda=10$, $e_{\mathrm{tol}}=10^{-5}$:

Brute-force skipped: radius-1 net 3,093,169 directions > 200,000-point cap.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery |
| --- | ---: | ---: | ---: |
| Algorithm 1 | 0.362205 | 237.292572 | 1.000 |
| Coordinate-wise MoM | 0.341607 | 0.000201 | 1.000 |
| Geometric MoM + HT | 0.369129 | 0.000451 | 1.000 |
| Sample mean | 0.717273 | 0.000013 | — |

![L2 error histograms for Experiment 6](experiment6_error_histograms.png)

## Experiments 7

Results for $(d, s, \delta, \epsilon, n) = (20, 5, 0.05, 0.02, 1000)$; $\nu=2.5$, $K=41$, $\overline\lambda=10$, $e_{\mathrm{tol}}=10^{-5}$:

Brute-force skipped: radius-1 net 408,076,993 directions; 65.3 GB array > 16 GB RAM.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery |
| --- | ---: | ---: | ---: |
| Algorithm 1 | 0.482036 | 1957.417511 | 1.000 |
| Coordinate-wise MoM | 0.386067 | 0.000267 | 1.000 |
| Geometric MoM + HT | 0.370027 | 0.000592 | 1.000 |
| Sample mean | 0.728582 | 0.000018 | — |

![L2 error histograms for Experiment 7](experiment7_error_histograms.png)

## Experiments 8

Results for $(d, s, \delta, \epsilon, n) = (10, 3, 0.05, 0.01, 1000)$; $\nu=2.5$, $K=21$, $\overline\lambda=764.098772$, $e_{\mathrm{tol}}=10^{-5}$:

Skew-$t_\nu$ with shape $S=0.5I_d+0.5\mathbf{1}\mathbf{1}^\top$ and skew vector $b=10\mathbf{1}/\sqrt d$. Seeds 0–9; the radius-1 net has 29,305 directions. All methods completed all 10 seeds.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery |
| --- | ---: | ---: | ---: |
| Algorithm 1 | 0.500894 | 3.032932 | 1.000 |
| Brute-force | 0.496876 | 0.738198 | 1.000 |
| Projected Algorithm 1 (LS) | 0.401590 | 5.964727 | 1.000 |
| Projected Algorithm 1 (max) | 0.410981 | 6.110855 | 1.000 |
| Coordinate-wise MoM | 0.437631 | 0.000166 | 1.000 |
| Geometric MoM + HT | 0.377550 | 0.003708 | 1.000 |
| Random MoM (tilde J=1000) | 0.488809 | 0.057097 | 1.000 |
| Random trimmed mean (tilde J=1000) | 0.741115 | 0.106122 | 1.000 |
| Sample mean | 0.760233 | 0.000021 | — |

Projected methods: $J=10$, $r=5$, projected $K=21$; $e_{\mathrm{tol}}=10^{-5}$ for Algorithm 1 and every dense projection. Aggregation uses support cutting planes with absolute/relative tolerances $10^{-5}$ and same-support LS refinement for max. All seven methods completed seeds 0–9; four concurrent one-thread runs. All coordinates were covered in every seed. Per-seed errors and metrics: `experiment8_errors.json`, `experiment8_results.json`.

Random MoM and random trimmed mean: both completed seeds 0–9 using the identical historical Experiment 8 samples (verified hashes and true means), with $\widetilde J=1000$ random sparse directions plus 10 coordinate directions (1010 total), direction seed equal to data seed, and tolerance $10^{-5}$. Random MoM uses $K=21$; random trimmed mean uses $k=15$ ($n-2k=970$). Objective gaps certify the finite direction bank. Each method was run once per seed, sequentially with one solver thread; runtimes include preprocessing but exclude shared data generation. Historical baseline runtimes used four concurrent workers. Per-seed estimates, metrics, bounds, settings and source hashes: [experiment8_random_j1000_results.json](experiment8_random_j1000_results.json).

![L2 error histograms for Experiment 8](experiment8_error_histograms.png)

## Experiments 9

Results for $(d, s, \delta, \epsilon, n) = (10, 3, 0.05, 0.02, 1000)$; $\nu=2.5$, $K=41$, $\overline\lambda=764.098772$, $e_{\mathrm{tol}}=10^{-5}$:

Skew-$t_\nu$ with shape $S=0.5I_d+0.5\mathbf{1}\mathbf{1}^\top$ and skew vector $b=10\mathbf{1}/\sqrt d$. Seeds 0–9; the radius-1 net has 29,305 directions. All methods completed all 10 seeds.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery |
| --- | ---: | ---: | ---: |
| Algorithm 1 | 0.814742 | 51.383820 | 1.000 |
| Brute-force | 0.775652 | 0.811421 | 1.000 |
| Coordinate-wise MoM | 0.753275 | 0.000212 | 1.000 |
| Geometric MoM + HT | 0.666036 | 0.000769 | 1.000 |
| Sample mean | 1.394707 | 0.000015 | — |

![L2 error histograms for Experiment 9](experiment9_error_histograms.png)

## Experiments 10

Results for $(d, s, \delta, \epsilon, n) = (20, 5, 0.05, 0.05, 1000)$; $\nu=2.5$, $K=101$, $\overline\lambda=814.098772$, $e_{\mathrm{tol}}=10^{-5}$:

Skew-$t_\nu$ with shape $S=0.5I_d+0.5\mathbf{1}\mathbf{1}^\top$ and skew vector $b=10\mathbf{1}/\sqrt d$. Seeds 0–9 completed for the three fast methods.

Brute-force skipped: the radius-1 net has 408,076,993 directions and would require 65.3 GB for its raw float64 matrix.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery |
| --- | ---: | ---: | ---: |
| Algorithm 1 (0/10 completed) | — | — | — |
| Coordinate-wise MoM | 3.058356 | 0.000287 | 0.620 |
| Geometric MoM + HT | 3.179425 | 0.000850 | 0.600 |
| Sample mean | 4.950316 | 0.000010 | — |

Algorithm 1 seeds 0–3 each ran for over two hours without returning an estimator and were stopped as timeouts. Seeds 4–9 were not attempted after these four timeouts; no Algorithm 1 mean is available.

![L2 error histograms for completed methods in Experiment 10](experiment10_error_histograms.png)

## Experiments 11

Results for $(d, s, \delta, \epsilon, n) = (10, 4, 0.05, 0.03, 1000)$; $\nu=2.5$, $K=61$, $\overline\lambda=764.098772$, $e_{\mathrm{tol}}=10^{-5}$:

Skew-$t_\nu$ with shape $S=0.5I_d+0.5\mathbf{1}\mathbf{1}^\top$ and skew vector $b=10\mathbf{1}/\sqrt d$. Seeds 0–9; the radius-1 net has 110,125 directions.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery |
| --- | ---: | ---: | ---: |
| Algorithm 1 (9/10 completed) | 1.443974 | 1031.438921 | 1.000 |
| Brute-force | 1.837067 | 16.560599 | 0.900 |
| Coordinate-wise MoM | 1.584275 | 0.000257 | 0.950 |
| Geometric MoM + HT | 1.190783 | 0.000884 | 1.000 |
| Sample mean | 3.330161 | 0.000015 | — |

Algorithm 1 means use completed seeds 0, 1, 2, 3, 4, 5, 6, 7, 9 only; seeds 8 are still running. The Algorithm 1 means are not comparable 10-seed estimates.

![L2 error histograms for Experiment 11](experiment11_error_histograms.png)

## Experiments 12

Results for $(d, s, \delta, \epsilon, n) = (10, 3, 0.05, 0.01, 1000)$; $\nu=2.5$, $K=21$, $\overline\lambda=764.098772$, $e_{\mathrm{tol}}=10^{-5}$:

Skew-$t_\nu$ with $S=0.5I_d+0.5\mathbf{1}\mathbf{1}^\top$ and $b=10\mathbf{1}/\sqrt d$; attack strength 100, $r=5$, $J=d/r=2$. Seeds 0–9; all seven methods completed.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery |
| --- | ---: | ---: | ---: |
| Algorithm 1 | 0.597770 | 8.016217 | 1.000 |
| Brute-force | 0.606569 | 1.392120 | 1.000 |
| Projected Algorithm 1 (LS) | 0.537169 | 2.720179 | 1.000 |
| Projected Algorithm 1 (max) | 0.537169 | 2.696994 | 1.000 |
| Coordinate-wise MoM | 0.508337 | 0.000367 | 1.000 |
| Geometric MoM + HT | 0.470341 | 0.001664 | 1.000 |
| Sample mean | 1.902049 | 0.000051 | — |

![L2 error histograms for Experiment 12](experiment12_error_histograms.png)

## Experiments 13

Results for $(d, s, \delta, \epsilon, n) = (20, 5, 0.05, 0, 100)$; $\nu=3$, $C=2$, $e_{\mathrm{tol}}=10^{-5}$:

Experiment 1 speed comparison, seeds 0–9: the archived implementation before acceleration versus the current hybrid implementation, using identical data and tolerances. Projected methods use the same current coordinate partition, $r=10$, $J=d/r=2$. Four concurrent workers; one solver thread per worker; execution order alternated by seed. All runs converged.

| Method | Before mean runtime (s) | After mean runtime (s) | Speedup | Faster seeds |
| --- | ---: | ---: | ---: | ---: |
| Algorithm 1 | 62.508230 | 10.462445 | 5.97× | 10/10 |
| Projected LS | 0.289558 | 0.236186 | 1.23× | 8/10 |
| Projected max | 0.279559 | 0.237639 | 1.18× | 8/10 |

| Method | Before mean L2 error | After mean L2 error | Before / after mean support recovery |
| --- | ---: | ---: | ---: |
| Algorithm 1 | 0.430643 | 0.434042 | 1.000 / 1.000 |
| Projected LS | 0.338212 | 0.338214 | 1.000 / 1.000 |
| Projected max | 0.338212 | 0.338214 | 1.000 / 1.000 |

Algorithm 1 exact separation calls fell from 231.1 to 28.4 per seed on average. Its objective certificate intervals agree within 7.3e-9 across implementations; the returned estimates can differ despite matching objective values. The speed gain is substantial for Algorithm 1 and modest for the already fast projected methods. These are paired comparisons across seeds, with one timing measurement per version and seed.

Per-seed estimates, timings, certificates, settings and source hashes: [experiment1_speed_comparison.json](experiment1_speed_comparison.json). Both versions use the current tolerance and projection partition, so the historical Experiment 1 timing table is not the baseline for this comparison.

## Experiments 14

Results for $(d, s, \delta, \epsilon, n) = (10, 3, 0.05, 0.01, 1000)$; $\nu=2.5$, $C=2$, $e_{\mathrm{tol}}=10^{-5}$:

Experiment 8 speed comparison, seeds 0–9: centered skew-t with $S=0.5I_d+0.5\mathbf{1}\mathbf{1}^\top$, $b=10\mathbf{1}/\sqrt d$, attack strength 100, $\overline\lambda=764.098772$. Both implementations use identical data, tolerances and the current coordinate partition ($r=5$, $J=d/r=2$). Four concurrent workers; one solver thread per worker; execution order alternated by seed. All runs converged.

| Method | Before mean runtime (s) | After mean runtime (s) | Speedup | Faster seeds |
| --- | ---: | ---: | ---: | ---: |
| Algorithm 1 | 4.066053 | 0.864941 | 4.70× | 10/10 |
| Projected LS | 1.258986 | 0.858614 | 1.47× | 8/10 |
| Projected max | 1.254451 | 0.874593 | 1.43× | 8/10 |

| Method | Before mean L2 error | After mean L2 error | Before / after mean support recovery |
| --- | ---: | ---: | ---: |
| Algorithm 1 | 0.597770 | 0.646946 | 1.000 / 1.000 |
| Projected LS | 0.537169 | 0.537167 | 1.000 / 1.000 |
| Projected max | 0.537169 | 0.537167 | 1.000 / 1.000 |

Algorithm 1 exact separation calls fell from 45.3 to 5.4 per seed on average. Its objective certificate intervals agree within 6.4e-9 across implementations, while its mean L2 error increased by 8.2%. Matching objective values do not imply matching estimates or L2 errors; this is consistent with nonunique optima. Projected errors are essentially unchanged. Timing results use one measurement per version and seed.

Per-seed estimates, timings, certificates, settings and source hashes: [experiment8_speed_comparison.json](experiment8_speed_comparison.json). The baseline is the archived implementation before acceleration, rerun under the same settings; the historical Experiment 8 uses different projection and attack settings.


## Experiments 15

Results for $(d, s, \delta, \epsilon, n) = (20, 5, 0.05, 0.05, 1000)$; $\nu=2.5$, $C=2$, $K=101$, $e_{\mathrm{tol}}=10^{-5}$:

Centered skew-t with $S=0.5I_d+0.5\mathbf{1}\mathbf{1}^\top$, $b=10\mathbf{1}/\sqrt d$, attack strength 100, $\overline\lambda=814.098772$. Random coordinate partition: $r=5$, $J=d/r=4$. Seeds 0–9; Algorithm 1 and brute-force skipped; four one-thread workers. Limit: 1 hour/seed for LS then max combined.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery | Completed seeds |
| --- | ---: | ---: | ---: | ---: |
| Projected Algorithm 1 (LS) | 5.188187 | 381.698262 | 0.285714 | 7/10 |
| Projected Algorithm 1 (max) | 5.181456 | 384.561399 | 0.285714 | 7/10 |
| Coordinate-wise MoM | 3.467775 | 0.000508 | 0.540000 | 10/10 |
| Geometric MoM + HT | 3.905997 | 0.001449 | 0.460000 | 10/10 |
| Sample mean | 13.596759 | 0.000024 | — | 10/10 |

Timeout seeds: [1, 5, 7]; failed seeds: []. Means and histograms use returned estimators only; completion counts differ, so these are not matched 10-seed comparisons.

![L2 error histograms for Experiment 15](experiment15_error_histograms.png)

Per-seed results: [experiment15_results.json](experiment15_results.json).


## Experiments 16

Results for $(d, s, \delta, \epsilon, n) = (20, 5, 0.05, 0.01, 1000)$; $\nu=2.5$, $C=2$, $K=21$, $e_{\mathrm{tol}}=10^{-5}$:

Centered skew-t with $S=0.5I_d+0.5\mathbf{1}\mathbf{1}^\top$, $b=10\mathbf{1}/\sqrt d$, attack strength 100, $\overline\lambda=814.098772$. Random coordinate partition: $r=10$, $J=d/r=2$. Seeds 0–9; Algorithm 1 and brute-force skipped; four one-thread workers. Limit: 1 hour/seed for LS then max combined.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery | Completed seeds |
| --- | ---: | ---: | ---: | ---: |
| Projected Algorithm 1 (LS) | 0.922114 | 1.949102 | 1.000000 | 10/10 |
| Projected Algorithm 1 (max) | 0.922114 | 1.776757 | 1.000000 | 10/10 |
| Coordinate-wise MoM | 0.894656 | 0.000153 | 1.000000 | 10/10 |
| Geometric MoM + HT | 0.936265 | 0.000690 | 1.000000 | 10/10 |
| Sample mean | 3.042672 | 0.000015 | — | 10/10 |

All five methods completed all ten seeds.

![L2 error histograms for Experiment 16](experiment16_error_histograms.png)

Per-seed results: [experiment16_results.json](experiment16_results.json).


## Experiments 17

Results for $(d, s, \delta, \epsilon, n) = (20, 5, 0.05, 0.02, 1000)$; $\nu=2.5$, $C=2$, $K=41$, $e_{\mathrm{tol}}=10^{-5}$:

Centered skew-t with $S=0.5I_d+0.5\mathbf{1}\mathbf{1}^\top$, $b=10\mathbf{1}/\sqrt d$, attack strength 100, $\overline\lambda=814.098772$. Random coordinate partition: $r=10$, $J=d/r=2$. Seeds 0–9; Algorithm 1 and brute-force skipped; four one-thread workers. Limit: 1 hour/seed for LS then max combined.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery | Completed seeds |
| --- | ---: | ---: | ---: | ---: |
| Projected Algorithm 1 (LS) | 1.537994 | 29.858429 | 0.900000 | 10/10 |
| Projected Algorithm 1 (max) | 1.539662 | 29.313480 | 0.900000 | 10/10 |
| Coordinate-wise MoM | 1.506711 | 0.000218 | 0.920000 | 10/10 |
| Geometric MoM + HT | 1.562103 | 0.000929 | 0.900000 | 10/10 |
| Sample mean | 5.719132 | 0.000017 | — | 10/10 |

All five methods completed all ten seeds.

![L2 error histograms for Experiment 17](experiment17_error_histograms.png)

Per-seed results: [experiment17_results.json](experiment17_results.json).


## Experiments 18

Results for $(d, s, \delta, \epsilon, n) = (20, 5, 0.05, 0.03, 1000)$; $\nu=2.5$, $C=2$, $K=61$, $e_{\mathrm{tol}}=10^{-5}$:

Centered skew-t with $S=0.5I_d+0.5\mathbf{1}\mathbf{1}^\top$, $b=10\mathbf{1}/\sqrt d$, attack strength 100, $\overline\lambda=814.098772$. Random coordinate partition: $r=10$, $J=d/r=2$. Seeds 0–9; Algorithm 1 and brute-force skipped; four one-thread workers. Limit: 1 hour/seed for LS then max combined.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery | Completed seeds |
| --- | ---: | ---: | ---: | ---: |
| Projected Algorithm 1 (LS) | 2.003594 | 204.756288 | 0.860000 | 10/10 |
| Projected Algorithm 1 (max) | 2.298642 | 182.431536 | 0.820000 | 10/10 |
| Coordinate-wise MoM | 1.969372 | 0.000322 | 0.840000 | 10/10 |
| Geometric MoM + HT | 2.256754 | 0.001254 | 0.800000 | 10/10 |
| Sample mean | 8.363059 | 0.000018 | — | 10/10 |

All five methods completed all ten seeds.

![L2 error histograms for Experiment 18](experiment18_error_histograms.png)

Per-seed results: [experiment18_results.json](experiment18_results.json).


## Experiments 19

Results for $(d, s, \delta, \epsilon, n) = (20, 5, 0.05, 0.01, 1000)$; $\nu=2.5$, $C=2$, $K=21$, $e_{\mathrm{tol}}=10^{-5}$:

Centered skew-t with $S=0.5I_d+0.5\mathbf{1}\mathbf{1}^\top$, $b=10\mathbf{1}/\sqrt d$, attack strength 100, $\overline\lambda=814.098772$. Random coordinate partition: $r=10$, $J=d/r=2$. Seeds 0–99; Brute-force skipped; four one-thread workers. Limits: 1 hour/seed for LS then max combined; 1 hour/seed for Algorithm 1 separately.

Haar projections: $r=10$ and $r=5$, $J=10$, dense directions $V_r$, $K=21$; same data and seeds 0–99. Limit: 1 hour/seed for Haar LS then max combined.

Additional Haar projections: $r=5$, $J=20$, dense directions $V_5$, $K=21$; same data and seeds 0–99, same 1-hour seed limit.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery | Completed seeds |
| --- | ---: | ---: | ---: | ---: |
| Algorithm 1 | 1.243943 | 52.399818 | 0.969697 | 99/100 |
| Projected Algorithm 1 (LS) | 0.999439 | 2.208462 | 0.994000 | 100/100 |
| Projected Algorithm 1 (max) | 1.049063 | 2.185594 | 0.984000 | 100/100 |
| Haar-projection LS (r=10, J=10) | 1.061293 | 41.308871 | 0.981633 | 98/100 |
| Haar-projection max (r=10, J=10) | 1.217401 | 45.151798 | 0.963636 | 99/100 |
| Haar-projection LS (r=5, J=10) | 1.168008 | 30.404181 | 0.971134 | 97/100 |
| Haar-projection max (r=5, J=10) | 1.487302 | 50.400809 | 0.928000 | 100/100 |
| Haar-projection LS (r=5, J=20) | 1.071429 | 26.672358 | 0.979167 | 96/100 |
| Haar-projection max (r=5, J=20) | 1.349558 | 48.268395 | 0.955556 | 99/100 |
| Coordinate-wise MoM | 0.997306 | 0.000162 | 0.988000 | 100/100 |
| Geometric MoM + HT | 1.055154 | 0.000734 | 0.970000 | 100/100 |
| Random MoM (tilde J=1000) | 1.374459 | 1.145171 | 0.964000 | 100/100 |
| Random trimmed mean (tilde J=1000) | 0.945187 | 1.167080 | 0.988000 | 100/100 |
| Sample mean | 3.108600 | 0.000015 | — | 100/100 |

Algorithm 1: seed 48 reached the 1,000-iteration outer cutting-plane limit after 1,699.94 s (gap 0.050845 versus tolerance 0.00001); no timeout. Its unconverged estimate and runtime are excluded from the averages and histogram. Existing methods were not rerun.

Haar-projection LS (r=10): incomplete seeds [67, 93]; averages and histogram include converged runs only.
Seed 67: 1000-iteration cutting-plane limit; gap 5.95802, tolerance 0.000263836.
Seed 93: 1000-iteration cutting-plane limit; gap 3.40213, tolerance 0.000242141.

Haar-projection max (r=10): incomplete seeds [67]; averages and histogram include converged runs only.
Seed 67: 1000-iteration cutting-plane limit; gap 0.55208, tolerance 3.55602e-05.

Averages use each method's converged seeds, so seed subsets differ.

Haar max (r=10) seed 40 was retried after adding a primal-simplex fallback for a numerical QP failure; reported runtime is the successful retry only.

Interrupted license-connection failures were resumed without rerunning completed methods; runtimes exclude unsuccessful attempts.

Haar-projection LS (r=5): incomplete seeds [48, 67, 93]; averages and histogram include converged runs only.
Seed 48: 1000-iteration cutting-plane limit; gap 0.518558, tolerance 3.17865e-05.
Seed 67: 1000-iteration cutting-plane limit; gap 3.00398, tolerance 0.000111002.
Seed 93: 1000-iteration cutting-plane limit; gap 5.55311, tolerance 0.000128922.

Haar-projection max (r=5): all 100 seeds converged.

Matched-seed comparisons (r=10 → r=5):

- LS, 97 seeds: mean L2 error 1.066919 → 1.168008; mean runtime 39.046 → 30.404 s.
- max, 99 seeds: mean L2 error 1.217401 → 1.468083; mean runtime 45.152 → 42.464 s.

Haar-projection LS (r=5, J=20): incomplete seeds [48, 67, 75, 93]; averages and histogram include converged runs only.
Seed 48: 1000-iteration cutting-plane limit; gap 0.851401, tolerance 6.91057e-05.
Seed 67: 1000-iteration cutting-plane limit; gap 10.4488, tolerance 0.000261704.
Seed 75: 1000-iteration cutting-plane limit; gap 0.239709, tolerance 0.000228363.
Seed 93: 1000-iteration cutting-plane limit; gap 8.50622, tolerance 0.000288019.

Haar-projection max (r=5, J=20): incomplete seeds [93]; averages and histogram include converged runs only.
Seed 93: 1000-iteration cutting-plane limit; gap 0.177573, tolerance 2.41932e-05.

Max QCP certification failures on seeds [41, 56, 62] were retried with a presolve-disabled fallback; objective and tolerance are unchanged, and runtimes use successful retries only.

Matched-seed comparisons for J=20 (all three configurations converged):

- LS, 96 seeds, $(r,J)=(10,10) / (5,10) / (5,20)$: mean L2 error 1.040040 / 1.149822 / 1.071429; mean runtime 37.144 / 25.584 / 26.672 s.
- max, 98 seeds, $(r,J)=(10,10) / (5,10) / (5,20)$: mean L2 error 1.201188 / 1.439001 / 1.327345; mean runtime 42.279 / 37.850 / 39.640 s.

Random MoM and random trimmed mean: both completed seeds 0–99 using the identical historical Experiment 19 samples (verified hashes and true means), with $\widetilde J=1000$ random sparse directions plus 20 coordinate directions (1020 total), direction seed equal to data seed, and tolerance $10^{-5}$. Random MoM uses $K=21$; random trimmed mean uses $k=15$ ($n-2k=970$). Objective gaps certify the finite direction bank. Each method was run once per seed, sequentially with one solver thread; runtimes include preprocessing but exclude shared data generation. Historical baseline runtimes used four concurrent workers. Per-seed estimates, metrics, bounds, settings and source hashes: [experiment19_random_j1000_results.json](experiment19_random_j1000_results.json). Median runtimes: 0.134776 s (MoM) and 0.152533 s (trimmed).

![L2 error histograms for Experiment 19](experiment19_error_histograms.png)

Per-seed results: [experiment19_results.json](experiment19_results.json).


## Experiments 20

Results for $(d, s, \delta, \epsilon, n) = (20, 5, 0.05, 0.02, 1000)$; $\nu=2.5$, $C=2$, $K=41$, $e_{\mathrm{tol}}=10^{-5}$:

Centered skew-t with $S=0.5I_d+0.5\mathbf{1}\mathbf{1}^\top$, $b=10\mathbf{1}/\sqrt d$, attack strength 100, $\overline\lambda=814.098772$. Random coordinate partition: $r=10$, $J=d/r=2$. Seeds 0–99; Algorithm 1 and brute-force skipped; four one-thread workers. Limit: 1 hour/seed for LS then max combined.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery | Completed seeds |
| --- | ---: | ---: | ---: | ---: |
| Projected Algorithm 1 (LS) | 1.726797 | 30.043136 | 0.878000 | 100/100 |
| Projected Algorithm 1 (max) | 1.784936 | 30.274020 | 0.870000 | 100/100 |
| Coordinate-wise MoM | 1.531293 | 0.000255 | 0.918000 | 100/100 |
| Geometric MoM + HT | 1.652601 | 0.000948 | 0.894000 | 100/100 |
| Random MoM (tilde J=1000) | 2.179766 | 6.996453 | 0.834000 | 100/100 |
| Random trimmed mean (tilde J=1000) | 1.404469 | 4.889970 | 0.950000 | 100/100 |
| Sample mean | 5.844432 | 0.000018 | — | 100/100 |

All five methods completed all 100 seeds.

Random MoM and random trimmed mean: both completed seeds 0–99 using the identical historical Experiment 20 samples (verified hashes and true means), with $\widetilde J=1000$ random sparse directions plus 20 coordinate directions (1020 total), direction seed equal to data seed, and tolerance $10^{-5}$. Random MoM uses $K=41$; random trimmed mean uses $k=30$ ($n-2k=940$). Objective gaps certify the finite direction bank. Each method was run once per seed, sequentially with one solver thread; runtimes include preprocessing but exclude shared data generation. Historical baseline runtimes used four concurrent workers. Per-seed estimates, metrics, bounds, settings and source hashes: [experiment20_random_j1000_results.json](experiment20_random_j1000_results.json). Median runtimes: 0.455791 s (MoM) and 0.348618 s (trimmed).

![L2 error histograms for Experiment 20](experiment20_error_histograms.png)

Per-seed results: [experiment20_results.json](experiment20_results.json).


## Experiments 21

Results for $(d, s, \delta, \epsilon, n) = (30, 5, 0.05, 0.01, 1000)$; $\nu=2.5$, $C=2$, $K=21$, $e_{\mathrm{tol}}=10^{-5}$:

Centered skew-t with $S=0.5I_d+0.5\mathbf{1}\mathbf{1}^\top$, $b=10\mathbf{1}/\sqrt d$, attack strength 100, $\overline\lambda=864.098772$. Random coordinate partitions: $r=10$, $J=3$ and $r=15$, $J=2$. Seeds 0–99; Algorithm 1, projected max and brute-force skipped; four one-thread workers. Limit: 1 hour/seed for projected LS.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery | Completed seeds |
| --- | ---: | ---: | ---: | ---: |
| Projected Algorithm 1 (LS, r=10) | 1.174407 | 2.791410 | 0.986000 | 100/100 |
| Projected Algorithm 1 (LS, r=15) | 1.164001 | 28.871491 | 0.988000 | 100/100 |
| Coordinate-wise MoM | 1.128881 | 0.000179 | 0.988000 | 100/100 |
| Geometric MoM + HT | 1.263946 | 0.000788 | 0.976000 | 100/100 |
| Sample mean | 3.327734 | 0.000018 | — | 100/100 |

All five methods completed all 100 seeds.

![L2 error histograms for Experiment 21](experiment21_error_histograms.png)

Per-seed results: [experiment21_results.json](experiment21_results.json).


## Experiments 22

Results for $(d, s, \delta, \epsilon, n) = (40, 5, 0.05, 0.01, 1000)$; $\nu=2.5$, $C=2$, $K=21$, $e_{\mathrm{tol}}=10^{-5}$:

Centered skew-t with $S=0.5I_d+0.5\mathbf{1}\mathbf{1}^\top$, $b=10\mathbf{1}/\sqrt d$, attack strength 100, $\overline\lambda=914.098772$. Random coordinate partition: $r=10$, $J=d/r=4$. Seeds 0–99; Algorithm 1, projected max and brute-force skipped; four one-thread workers. Limit: 1 hour/seed for projected LS.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery | Completed seeds |
| --- | ---: | ---: | ---: | ---: |
| Projected Algorithm 1 (LS) | 1.275639 | 3.510773 | 0.964000 | 100/100 |
| Coordinate-wise MoM | 1.202494 | 0.000197 | 0.960000 | 100/100 |
| Geometric MoM + HT | 1.549655 | 0.000853 | 0.902000 | 100/100 |
| Random MoM (tilde J=1000) | 1.749871 | 0.394222 | 0.914000 | 100/100 |
| Random trimmed mean (tilde J=1000) | 0.827979 | 0.130175 | 1.000000 | 100/100 |
| Sample mean | 3.437052 | 0.000027 | — | 100/100 |

All four methods completed all 100 seeds.

Random MoM and random trimmed mean: both completed seeds 0–99 using the identical historical Experiment 22 samples (verified hashes and true means), with $\widetilde J=1000$ random sparse directions plus 40 coordinate directions (1040 total), direction seed equal to data seed, and tolerance $10^{-5}$. Random MoM uses $K=21$; random trimmed mean uses $k=15$ ($n-2k=970$). Objective gaps certify the finite direction bank. Each method was run once per seed, sequentially with one solver thread; runtimes include preprocessing but exclude shared data generation. Historical baseline runtimes used four concurrent workers. Per-seed estimates, metrics, bounds, settings and source hashes: [experiment22_random_j1000_results.json](experiment22_random_j1000_results.json). Median runtimes: 0.114101 s (MoM) and 0.118638 s (trimmed).

![L2 error histograms for Experiment 22](experiment22_error_histograms.png)

Per-seed results: [experiment22_results.json](experiment22_results.json).


## Experiments 23

Results for $(d, s, \delta, \epsilon, n) = (40, 5, 0.05, 0.02, 1000)$; $\nu=2.5$, $C=2$, $K=41$, $e_{\mathrm{tol}}=10^{-5}$:

Centered skew-t with $S=0.5I_d+0.5\mathbf{1}\mathbf{1}^\top$, $b=10\mathbf{1}/\sqrt d$, attack strength 100, $\overline\lambda=914.098772$. Random coordinate partition: $r=10$, $J=d/r=4$. Seeds 0–99; Algorithm 1, projected max and brute-force skipped; four one-thread workers. Limit: 1 hour/seed for projected LS.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery | Completed seeds |
| --- | ---: | ---: | ---: | ---: |
| Projected Algorithm 1 (LS) | 2.232842 | 72.856595 | 0.748000 | 100/100 |
| Coordinate-wise MoM | 1.682787 | 0.000353 | 0.890000 | 100/100 |
| Geometric MoM + HT | 2.577158 | 0.001276 | 0.654000 | 100/100 |
| Random MoM (tilde J=1000) | 2.591692 | 1.542800 | 0.722000 | 100/100 |
| Random trimmed mean (tilde J=1000) | 1.340091 | 0.453111 | 0.958000 | 100/100 |
| Sample mean | 6.484384 | 0.000038 | — | 100/100 |

All four methods completed all 100 seeds.

Random MoM and random trimmed mean: both completed seeds 0–99 using the identical historical Experiment 23 samples (verified hashes and true means), with $\widetilde J=1000$ random sparse directions plus 40 coordinate directions (1040 total), direction seed equal to data seed, and tolerance $10^{-5}$. Random MoM uses $K=41$; random trimmed mean uses $k=30$ ($n-2k=940$). Objective gaps certify the finite direction bank. Each method was run once per seed, sequentially with one solver thread; runtimes include preprocessing but exclude shared data generation. Historical baseline runtimes used four concurrent workers. Per-seed estimates, metrics, bounds, settings and source hashes: [experiment23_random_j1000_results.json](experiment23_random_j1000_results.json). Median runtimes: 0.428595 s (MoM) and 0.155313 s (trimmed).

![L2 error histograms for Experiment 23](experiment23_error_histograms.png)

Per-seed results: [experiment23_results.json](experiment23_results.json).


## Experiments 24

Results for $(d, s, \delta, \epsilon, n) = (50, 5, 0.05, 0.01, 1000)$; $\nu=2.5$, $C=2$, $K=25$ (baselines), $K=21$ (projections), $e_{\mathrm{tol}}=10^{-5}$:

Centered skew-t with $S=0.5I_d+0.5\mathbf{1}\mathbf{1}^\top$, $b=10\mathbf{1}/\sqrt d$, attack strength 100, $\overline\lambda=964.098772$. Random coordinate partition: $r=10$, $J=d/r=5$. Seeds 0–99; Algorithm 1, projected max and brute-force skipped; four one-thread workers. Limit: 1 hour/seed for projected LS.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery | Completed seeds |
| --- | ---: | ---: | ---: | ---: |
| Projected Algorithm 1 (LS) | 1.221435 | 5.467179 | 0.970000 | 100/100 |
| Coordinate-wise MoM | 0.920005 | 0.000245 | 0.998000 | 100/100 |
| Geometric MoM + HT | 1.074571 | 0.000786 | 0.998000 | 100/100 |
| Random MoM (tilde J=1000) | 1.481336 | 0.161116 | 0.964000 | 100/100 |
| Random trimmed mean (tilde J=1000) | 0.900542 | 0.121058 | 0.998000 | 100/100 |
| Sample mean | 3.419444 | 0.000031 | — | 100/100 |

All four methods completed all 100 seeds.

Random MoM and random trimmed mean: both completed seeds 0–99 using the identical historical Experiment 24 samples (verified hashes and true means), with $\widetilde J=1000$ random sparse directions plus 50 coordinate directions (1050 total), direction seed equal to data seed, and tolerance $10^{-5}$. Random MoM uses $K=25$; random trimmed mean uses $k=15$ ($n-2k=970$). Objective gaps certify the finite direction bank. Each method was run once per seed, sequentially with one solver thread; runtimes include preprocessing but exclude shared data generation. Historical baseline runtimes used four concurrent workers. Per-seed estimates, metrics, bounds, settings and source hashes: [experiment24_random_j1000_results.json](experiment24_random_j1000_results.json). Median runtimes: 0.094003 s (MoM) and 0.110511 s (trimmed).

![L2 error histograms for Experiment 24](experiment24_error_histograms.png)

Per-seed results: [experiment24_results.json](experiment24_results.json).


## Experiments 25

Results for $(d, s, \delta, \epsilon, n) = (100, 5, 0.05, 0.01, 1000)$; $\nu=2.5$, $C=2$, $K=31$ (baselines), $K=21$ (projections), $e_{\mathrm{tol}}=10^{-5}$:

Centered skew-t with $S=0.5I_d+0.5\mathbf{1}\mathbf{1}^\top$, $b=10\mathbf{1}/\sqrt d$, attack strength 100, $\overline\lambda=1214.098772$. Random coordinate partitions: $r=10$, $J=10$ and $r=20$, $J=5$. Seeds 0–99; Algorithm 1, projected max and brute-force skipped; four one-thread workers. Limit: 1 hour/seed for projected LS.

| Method | Mean L2 error | Mean runtime (s) | Mean support recovery | Completed seeds |
| --- | ---: | ---: | ---: | ---: |
| Projected Algorithm 1 (LS, r=10) | 1.210730 | 25.511930 | 0.952000 | 100/100 |
| Projected Algorithm 1 (LS, r=20) | 1.246798 | 422.312890 | 0.956098 | 41/100 |
| Coordinate-wise MoM | 0.714421 | 0.000401 | 1.000000 | 100/100 |
| Geometric MoM + HT | 1.017777 | 0.001079 | 0.994000 | 100/100 |
| Random MoM (tilde J=1000) | 1.044994 | 0.086810 | 0.990000 | 100/100 |
| Random trimmed mean (tilde J=1000) | 0.625352 | 0.110320 | 1.000000 | 100/100 |
| Sample mean | 3.086153 | 0.000041 | — | 100/100 |

Haar-projection LS and max ($r=5$, $J=10$): **stopped by user** before all 100 seeds were processed. Partial results and per-seed statuses are preserved in the [checkpoint](experiment25_haar_r5_J10_progress.json); these runs have not been added to the final table or histogram.

Haar-projection LS and max ($r=5$, $J=20$): **stopped by user** before all 100 seeds were processed. Partial results and per-seed statuses are preserved in the [checkpoint](experiment25_haar_progress.json); these runs have not been added to the final table or histogram.

Projected LS r=20 incomplete seeds: [{'seed': 31, 'status': 'failed'}, {'seed': 32, 'status': 'failed'}, {'seed': 33, 'status': 'failed'}, {'seed': 34, 'status': 'failed'}, {'seed': 35, 'status': 'failed'}, {'seed': 36, 'status': 'failed'}, {'seed': 37, 'status': 'failed'}, {'seed': 38, 'status': 'failed'}, {'seed': 39, 'status': 'failed'}, {'seed': 40, 'status': 'failed'}, {'seed': 41, 'status': 'failed'}, {'seed': 42, 'status': 'failed'}, {'seed': 44, 'status': 'failed'}, {'seed': 45, 'status': 'failed'}, {'seed': 46, 'status': 'failed'}, {'seed': 47, 'status': 'failed'}, {'seed': 48, 'status': 'failed'}, {'seed': 49, 'status': 'failed'}, {'seed': 50, 'status': 'failed'}, {'seed': 51, 'status': 'failed'}, {'seed': 52, 'status': 'failed'}, {'seed': 53, 'status': 'failed'}, {'seed': 54, 'status': 'failed'}, {'seed': 55, 'status': 'failed'}, {'seed': 56, 'status': 'failed'}, {'seed': 57, 'status': 'failed'}, {'seed': 58, 'status': 'failed'}, {'seed': 59, 'status': 'failed'}, {'seed': 60, 'status': 'failed'}, {'seed': 61, 'status': 'failed'}, {'seed': 62, 'status': 'failed'}, {'seed': 63, 'status': 'failed'}, {'seed': 64, 'status': 'failed'}, {'seed': 65, 'status': 'failed'}, {'seed': 66, 'status': 'failed'}, {'seed': 67, 'status': 'failed'}, {'seed': 68, 'status': 'failed'}, {'seed': 78, 'status': 'failed'}, {'seed': 79, 'status': 'failed'}, {'seed': 80, 'status': 'failed'}, {'seed': 81, 'status': 'failed'}, {'seed': 82, 'status': 'failed'}, {'seed': 83, 'status': 'failed'}, {'seed': 84, 'status': 'failed'}, {'seed': 85, 'status': 'failed'}, {'seed': 86, 'status': 'failed'}, {'seed': 87, 'status': 'failed'}, {'seed': 88, 'status': 'failed'}, {'seed': 89, 'status': 'failed'}, {'seed': 90, 'status': 'failed'}, {'seed': 91, 'status': 'failed'}, {'seed': 92, 'status': 'failed'}, {'seed': 93, 'status': 'failed'}, {'seed': 94, 'status': 'failed'}, {'seed': 95, 'status': 'failed'}, {'seed': 96, 'status': 'failed'}, {'seed': 97, 'status': 'failed'}, {'seed': 98, 'status': 'failed'}, {'seed': 99, 'status': 'failed'}]. Its averages and histogram use converged returned estimates only; other methods use all 100 seeds.

Random MoM and random trimmed mean: both completed seeds 0–99 using the identical historical Experiment 25 samples (verified hashes and true means), with $\widetilde J=1000$ random sparse directions plus 100 coordinate directions (1100 total), direction seed equal to data seed, and tolerance $10^{-5}$. Random MoM uses $K=31$; random trimmed mean uses $k=15$ ($n-2k=970$). Objective gaps certify the finite direction bank. Each method was run once per seed, sequentially with one solver thread; runtimes include preprocessing but exclude shared data generation. Historical baseline runtimes used four concurrent workers. Per-seed estimates, metrics, bounds, settings and source hashes: [experiment25_random_j1000_results.json](experiment25_random_j1000_results.json). Median runtimes: 0.081184 s (MoM) and 0.109385 s (trimmed).

![L2 error histograms for Experiment 25](experiment25_error_histograms.png)

Per-seed results: [experiment25_results.json](experiment25_results.json).

## Experiments 26

**Experiment 26 is running.** Experiment 19 setup with $\epsilon=0.05$: $(n,d,s,\nu,\delta)=(1000,20,5,2.5,0.05)$, $C=2$, attack strength 100, tolerance $10^{-5}$. Seeds 0–99; four workers. Algorithm 1 and brute-force excluded.

The 13 methods are partition LS/max ($r=10,J=2$), Haar LS/max at $(r,J)=(10,10),(5,10),(5,20)$, coordinate MoM, geometric MoM + HT, sample mean, random MoM, and random trimmed mean ($\widetilde J=1000$). Existing iteration limits are retained; no wall-clock cutoff.

[Live checkpoint with per-method counts, active seeds, completed estimates and metrics](experiment26_progress.json). Partial means in this checkpoint refer only to converged runs. The final table and histogram will be added after all requested runs finish.
