# Experiments

Results of experiments run with the refactored code (October 2026 onward). Earlier experiments are
kept locally in `previous_experiments/` and are not tracked by git.

## Experiment 1

Support confusion under a far shrink/inflate attack, $s \ll d$. Results for $(d, s, \delta, \epsilon, n) = (16, 4, 0.05, 0.02, 1000)$; $C=3$, $K=61$, block subsampling $K_{\mathrm{sub}}=29$; default tolerance $e_{\mathrm{tol}} = 10^{-3}\,U_{\det}^{(0)}$; SDP screen off. Seeds 0–99; eight one-thread workers.

Data (`results/experiment1_support_confusion/run.py`, `generate`). A random permutation of $[d]$ gives disjoint $S_{\mathrm{true}}, S_{\mathrm{adversarial}} \subset [d]$ with $|S_{\mathrm{true}}| = |S_{\mathrm{adversarial}}| = s$ ($S_{\mathrm{adversarial}}$: the coordinates the attack pushes up). Inliers:

$$X_i = \mu^* + \Sigma^{1/2} g_i, \quad g_i \overset{\mathrm{iid}}{\sim} N(0, I_d), \quad i = 1, \dots, n,$$

$$\mu^* = \mathrm{loc}\cdot\mathbf 1_{S_{\mathrm{true}}}, \qquad \Sigma = \mathrm{diag}(\sigma_1^2, \dots, \sigma_d^2), \qquad \sigma_j^2 = \begin{cases} 1, & j \in S_{\mathrm{true}} \cup S_{\mathrm{adversarial}}, \\ 0.1, & \text{otherwise}, \end{cases}$$

$$\mathrm{loc} = 2 z \sigma_b, \qquad \sigma_b = \sqrt{K/n}, \qquad z = \Phi^{-1}\Big(\frac{0.5}{1-\beta}\Big), \qquad \beta = 1 - (1 - 1/K)^{\epsilon n},$$

These are design constants computed from $(n, K, \epsilon)$ and the noise variance before any data are drawn, so $\mu^*$ is well defined. Here $\sigma_b = \sqrt{\sigma_j^2/(n/K)}$ with $\sigma_j^2 = 1$ is the population standard deviation of a clean block mean on $S_{\mathrm{true}} \cup S_{\mathrm{adversarial}}$ (blocks have $\lfloor n/K \rfloor$ or $\lceil n/K \rceil$ rows), $\beta$ is the probability that a given block of the $K$-partition contains at least one of the $\epsilon n$ outliers, and $z$ is the shift, in units of $\sigma_b$, of the median of block means when a fraction $\beta$ of the block means is pushed to one side (derived below). Choosing $\mathrm{loc} = 2z\sigma_b$ puts $S_{\mathrm{true}}$ and $S_{\mathrm{adversarial}}$ at the same height after the attack. Numerically $\beta = 0.281$, $z = 0.512$, $z\sigma_b = 0.1266$, $\mathrm{loc} = 0.2532$ and $\|\mu^*\|_2 = 0.5064$.

Contamination: with $u = (-\mathbf 1_{S_{\mathrm{true}}} + \mathbf 1_{S_{\mathrm{adversarial}}})/\sqrt{2s}$, $\hat m$ the coordinate-wise median of the data and $p_i = (X_i - \hat m)^\top u$, let $O$ index the $\epsilon n = 20$ smallest $p_i$. For $i \in O$,

$$X_i \leftarrow X_i + \big(\mathrm{med}(p) + 1000\,\mathrm{MAD}(p) - p_i\big)\, u,$$

so $p_i$ becomes $\mathrm{med}(p) + 1000\,\mathrm{MAD}(p)$ and the component orthogonal to $u$ is kept. Each coordinate in $S_{\mathrm{true}}$ moves by about $-1000\,\mathrm{MAD}(p)/\sqrt{2s} \approx -230$ and each coordinate in $S_{\mathrm{adversarial}}$ by about $+230$.

Effect on MoM: a block of the $K$-partition contains an outlier with probability about $\beta$, and its block mean is then pushed far below (on $S_{\mathrm{true}}$) or above (on $S_{\mathrm{adversarial}}$). The clean block means are about $N(\mu^*_j, \sigma_b^2)$, so the median $x_j$ of block means solves $(1-\beta)\,\Phi\big((x_j - \mu^*_j)/\sigma_b\big) + \beta = 1/2$ for $j \in S_{\mathrm{true}}$, i.e. $\Phi\big((\mu^*_j - x_j)/\sigma_b\big) = 0.5/(1-\beta)$ and $x_j = \mu^*_j - z\sigma_b$; the mirrored equation for $j \in S_{\mathrm{adversarial}}$ gives $x_j = \mu^*_j + z\sigma_b$. Hence

$$x_j \approx \mathrm{loc} - z\sigma_b = z\sigma_b \quad (j \in S_{\mathrm{true}}), \qquad x_j \approx z\sigma_b \quad (j \in S_{\mathrm{adversarial}}),$$

so $S_{\mathrm{true}}$ and $S_{\mathrm{adversarial}}$ are tied after coordinate-wise MoM. The data seed, the MoM partition seed and the block-selection seed all equal the seed. All MoM baselines keep the $s$ largest absolute coordinates (HT).

| Method | Mean L2 error | Median L2 error | Mean support recovery | Exact support | Mean runtime (s) | Completed seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Algorithm 1 (block subsampling) | 0.486564 | 0.140398 | 0.840000 | 71/100 | 46.625721 | 100/100 |
| Coordinate-wise MoM + HT | 0.494029 | 0.516954 | 0.382500 | 2/100 | 0.000713 | 100/100 |
| Geometric MoM + HT | 0.439187 | 0.446076 | 0.495000 | 3/100 | 0.001485 | 100/100 |
| Sample mean + HT | 9.576332 | 9.562359 | 0.000000 | 0/100 | 0.000042 | 100/100 |

Per seed, Algorithm 1 (block subsampling) has a smaller L2 error than coordinate-wise MoM + HT on 88/100 seeds (median ratio 3.33), than geometric MoM + HT on 84/100 (median ratio 2.82), and than both on 83/100.

Algorithm 1 (block subsampling): on seed 23, 15 of the 29 sampled blocks contain outliers, a majority of $(29+1)/2 = 15$, so the median breaks down and the estimate is about 13.5 on four coordinates (L2 error 27.19). This single seed sets its mean; without seed 23 its mean L2 error is 0.216808, against 0.493500 for coordinate-wise MoM + HT and 0.438889 for geometric MoM + HT on the same 99 seeds. The number of contaminated sampled blocks ranged from 2 to 15 over the 100 seeds (full partition: about 17 of 61). Runtime: median 41.0 s, max 128.4 s; all runs converged. These runtimes are from the rerun after the warm-start fix in `separation_oracle` (October 15); the first run took median 27.7 s, but timing the old and new code side by side on seeds 0–3 gives the same runtimes, so the difference is machine state, not the fix. The estimates match the first run to within $1.8\times 10^{-2}$ per coordinate (62/100 seeds within $10^{-6}$), with the same exact-support count.

![Experiment 1 summary](results/experiment1_support_confusion/experiment1_summary.png)

Per-seed records (estimates, errors, support, runtimes): [records.json](results/experiment1_support_confusion/records.json). Plot: `results/experiment1_support_confusion/plot.py`.

### Full Algorithm 1 (30-minute run, 32 seeds)

Algorithm 1 on all $K = 61$ blocks (no subsampling), same data and seeds as above, run for a fixed 30-minute wall-clock budget with eight one-thread workers; seeds were submitted in order 0, 1, … and the run was stopped at the budget. **32 seeds completed**: 0–25, 27–31 and 34; seeds still running at the budget (including 26, 32 and 33) were discarded. The other methods below are the main-run records restricted to the same 32 seeds.

| Method | Mean L2 error | Median L2 error | Mean support recovery | Exact support | Mean runtime (s) | Completed seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Algorithm 1 (full) | 0.109967 | 0.105808 | 0.984375 | 30/32 | 395.819936 | 32/32 |
| Algorithm 1 (block subsampling) | 1.060637 | 0.152262 | 0.828125 | 21/32 | 45.861229 | 32/32 |
| Coordinate-wise MoM + HT | 0.498508 | 0.518152 | 0.375000 | 0/32 | 0.000714 | 32/32 |
| Geometric MoM + HT | 0.437665 | 0.440953 | 0.492188 | 0/32 | 0.001688 | 32/32 |
| Sample mean + HT | 9.468042 | 9.433400 | 0.000000 | 0/32 | 0.000053 | 32/32 |

Per seed, full Algorithm 1 has a smaller L2 error than block subsampling on 25/32 seeds (median ratio 1.45), and than coordinate-wise and geometric MoM + HT on all 32 (median ratios 4.83 and 4.20). On seed 23, where subsampling breaks down (15 of 29 sampled blocks contaminated), the full partition has 19 of 61 contaminated blocks, below the majority 31, and full Algorithm 1 recovers the exact support (L2 error 0.130). Without seed 23 the means are 0.109321 (full) and 0.217677 (subsampling). Full Algorithm 1 misses the exact support only on seeds 1 and 8, where subsampling also misses it. Runtime: median 359.4 s, min 91.6 s, max 972.6 s, a median of 8.0 times the subsampling runtime on the same seed; all runs converged. These seeds were run after the warm-start fix, with `IP_algorithm.py` as on disk at 12:31 on October 15.

Per-seed records: [records_full.json](results/experiment1_support_confusion/records_full.json). Runner: `results/experiment1_support_confusion/run_full.py` (usage `run_full.py BUDGET_S [WORKERS]`).

### Runtime breakdown (100 seeds)

A separate profiled run of Algorithm 1 used the same Gaussian instance, generator and seeds 0-99 as Experiment 1: $(n,d,s,\epsilon,\delta,C)=(1000,16,4,0.02,0.05,3)$, $K=61$, block subsampling $K_{\mathrm{sub}}=29$, and the default $e_{\mathrm{tol}}=10^{-3}U_{\det}^{(0)}$. Both the SDP and geometric screens were off. All 100 seeds converged within their reported optimization tolerance, and their subsampled block indices were checked against the seeded selection rule.

The timing run used up to eight one-thread workers. After an interruption, 27 completed eight-worker records were retained and the remaining 73 seeds used isolated processes with the same concurrency limit. Failed attempts and a separate two-worker trial are excluded from this analysis. The mean total estimator wall time was **81.161773 s** (median 72.743335 s). These are fresh profiling measurements; they do not replace the runtimes in the main comparison table.

| Phase | Mean time (s) | Mean per-seed share | Share of pooled time |
| --- | ---: | ---: | ---: |
| Support cutting plane | 3.557244 | 5.95% | 4.38% |
| Inner minimization (restricted SOCP) | 25.128491 | 33.61% | 30.96% |
| Separation MIP | 51.106653 | 58.20% | 62.97% |
| Other (heuristic, bounds, preprocessing and overhead) | 1.369384 | 2.24% | 1.69% |

For phase $p$ and seed $a$, let $t_{p,a}$ denote phase time and $T_a$ total estimator time. The mean per-seed share is $\frac{1}{100}\sum_{a=0}^{99}t_{p,a}/T_a$, giving every seed equal weight; the pooled share is $\sum_a t_{p,a}/\sum_a T_a$, giving longer runs more weight. Seed 23 is included in both summaries.

Phase times include model construction, optimization, solution/dual extraction and disposal. The support cutting-plane time excludes the entire inner support callback, so it does not double-count restricted SOCP or separation time. The "Other" row includes heuristic separation (including its small fixed-pair convex solves), deterministic bounds, block preprocessing and remaining overhead. Imports, data generation and result serialization are outside estimator timing. The phase times sum to the total time for every seed and agree with the estimator's built-in restricted/oracle timers.

Within the restricted SOCP phase, calls to `optimize` averaged **8.844536 s**, or 35.2% of that phase; the remaining 64.8% covers model construction, extraction, disposal and other overhead. Within the separation MIP phase, `optimize` averaged **49.669590 s**, or 97.2%. Separation MIP optimization is the largest cost; support cutting planes take a much smaller share.

![Experiment 1 runtime breakdown](results/experiment1_support_confusion/runtime_profile_100seeds/runtime_breakdown.png)

Profiling artifacts: [summary.json](results/experiment1_support_confusion/runtime_profile_100seeds/summary.json), [per-seed times (CSV)](results/experiment1_support_confusion/runtime_profile_100seeds/per_seed_times.csv), [full records](results/experiment1_support_confusion/runtime_profile_100seeds/records.json), and [metadata and source hashes](results/experiment1_support_confusion/runtime_profile_100seeds/metadata.json). Runners: `results/experiment1_support_confusion/runtime_profile_100seeds/profile_seeds.py` and `analyze.py`; frozen inputs and source files are in its `source/` directory. Like the other experiment artifacts, these files remain local under the git-ignored `results/` directory.

## Experiment 2

Support confusion as in Experiment 1, with heavy-tailed noise. Results for $(d, s, \delta, \epsilon, n) = (16, 4, 0.05, 0.02, 1000)$ and $\nu \in \{3, 5\}$; $C=3$, $K=61$, block subsampling $K_{\mathrm{sub}}=29$; default tolerance $e_{\mathrm{tol}} = 10^{-3}\,U_{\det}^{(0)}$; SDP screen off. Seeds 0–99; $\nu = 3$ and $\nu = 5$ ran at the same time with five one-thread workers each.

Data (`results/experiment2_t_noise/run.py`, `generate`; usage `run.py NU [WORKERS]`). $S_{\mathrm{true}}$, $S_{\mathrm{adversarial}}$, $\mu^*$, $\Sigma$, $\mathrm{loc}$ and the contamination are as in Experiment 1; only the law of $g_i$ changes:

$$X_i = \mu^* + \Sigma^{1/2} g_i, \qquad g_{ij} \overset{\mathrm{iid}}{\sim} \sqrt{\frac{\nu-2}{\nu}}\; t_\nu, \quad j = 1, \dots, d,$$

so $\mathbb E g_{ij} = 0$ and $\mathrm{Var}(g_{ij}) = 1$. The design constants $\sigma_b$, $\beta$, $z$ and $\mathrm{loc}$ depend on the noise only through its variance, so they are unchanged ($\mathrm{loc} = 0.2532$). The calibration $x_j \approx z\sigma_b$ for both $S_{\mathrm{true}}$ and $S_{\mathrm{adversarial}}$ uses the normal approximation of a clean block mean of about $n/K \approx 16$ rows, which still holds near the center for $\nu \in \{3, 5\}$. The data seed, the MoM partition seed and the block-selection seed all equal the seed. All MoM baselines keep the $s$ largest absolute coordinates (HT).

$\nu = 3$:

| Method | Mean L2 error | Median L2 error | Mean support recovery | Exact support | Mean runtime (s) | Completed seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Algorithm 1 (block subsampling) | 0.238676 | 0.151873 | 0.820000 | 66/100 | 71.197510 | 100/100 |
| Coordinate-wise MoM + HT | 0.433760 | 0.445740 | 0.510000 | 5/100 | 0.000720 | 100/100 |
| Geometric MoM + HT | 0.401860 | 0.429127 | 0.577500 | 9/100 | 0.001584 | 100/100 |
| Sample mean + HT | 8.109423 | 8.120835 | 0.000000 | 0/100 | 0.000044 | 100/100 |

$\nu = 5$:

| Method | Mean L2 error | Median L2 error | Mean support recovery | Exact support | Mean runtime (s) | Completed seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Algorithm 1 (block subsampling) | 0.257154 | 0.163086 | 0.802500 | 65/100 | 69.886178 | 100/100 |
| Coordinate-wise MoM + HT | 0.500418 | 0.513519 | 0.360000 | 0/100 | 0.000766 | 100/100 |
| Geometric MoM + HT | 0.443987 | 0.447244 | 0.480000 | 4/100 | 0.001632 | 100/100 |
| Sample mean + HT | 9.211436 | 9.201297 | 0.000000 | 0/100 | 0.000038 | 100/100 |

Per seed, Algorithm 1 (block subsampling) has a smaller L2 error than coordinate-wise MoM + HT on 85/100 seeds for $\nu = 3$ (median ratio 2.44) and 84/100 for $\nu = 5$ (median ratio 2.73), than geometric MoM + HT on 82/100 (2.15) and 79/100 (2.34), and than both on 82/100 and 79/100.

Algorithm 1 (block subsampling): no seed breaks down; the largest L2 error is 0.82 for both $\nu$. The number of contaminated sampled blocks ranged from 3 to 13 ($\nu = 3$) and from 4 to 13 ($\nu = 5$), always below the majority 15 of the 29 sampled blocks (full partition: about 17 of 61). Its mean L2 error is therefore below the Gaussian mean 0.4866 of Experiment 1, which seed 23 alone inflates, and close to the Gaussian mean without seed 23 (0.2168). The support confusion persists for the MoM baselines (0–9 exact supports out of 100). Runtime: median 64.3 s ($\nu = 3$) and 61.8 s ($\nu = 5$), max 212 s; all runs converged. The machine was under extra load during these runs, so runtimes are not comparable with Experiment 1.

Before the warm-start fix in `separation_oracle`, seed 51 ($\nu = 3$) and seed 87 ($\nu = 5$) failed with a spurious Gurobi error, "Q matrix is not positive semi-definite": the oracle passed a warm start that its own threshold screening had removed, and Gurobi 13.0.3 fails while repairing such a start. All 100 seeds above were run after the fix.

![Experiment 2 summary](results/experiment2_t_noise/experiment2_summary.png)

Per-seed records (estimates, errors, support, runtimes): [records_nu3.json](results/experiment2_t_noise/records_nu3.json), [records_nu5.json](results/experiment2_t_noise/records_nu5.json). Plot: `results/experiment2_t_noise/plot.py`.
