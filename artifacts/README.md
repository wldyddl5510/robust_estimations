# Experiment artifacts

This directory holds saved experiment results and figures. Existing values, datasets, and frozen estimator source snapshots are preserved.

| Location | Contents |
| --- | --- |
| `experiment*_results.json`, `experiment*_errors.json`, `*.png` | Original single-file results and histograms |
| `experiment9_dense_comparison/` through `experiment14_dense_d10_block_scale/` | Dense experiments, per-seed records, datasets, figures, configurations, and runners |
| `sanity_check/` | Pilot experiments and their supporting records |
| `benchmarks/` | Runtime and optimization comparisons |
| `archive/` | Removed experiments and retained historical evidence |

Read the [main results](../results.md) or [sanity check results](../sanity_check/sanity_check_results.md) for comparisons. Each bundle retains its original internal paths, including `source/` snapshots and hash manifests. Paths inside historical JSON records are relative to this artifact directory when they previously referred to a repository-level experiment bundle.

Run a current wrapper from the repository root, for example:

```sh
python artifacts/experiment14_dense_d10_block_scale/run.py --help
python plot_histograms.py artifacts/experiment1_errors.json
```

Current wrappers locate the repository from their own file location and read/write supporting files inside `artifacts/`. Logs and live execution state stay excluded from Git. Archived wrapper copies retain their historical contents.
