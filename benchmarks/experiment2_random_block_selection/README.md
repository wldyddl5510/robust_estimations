# Experiment 2: random block selection without replacement

Seeds 0–99 reproduce the saved Experiment 2 data (`experiment_reference.json`). The estimator selects 23 of the original 41 blocks uniformly without replacement. Coefficient 2, separate selection RNG stream `(2,)`, tolerance `1e-5`, and the usual Algorithm 1 heuristics apply. Four concurrent processes use one solver/BLAS thread each. There is no wall-time cutoff; outer and inner iteration limits are 1,000.

- `source/`: the exact estimator and data-generation code used.
- `preflight.json`: hashes and selected block indices for all 100 seeds; optimization is not run during preflight.
- `seed_N.json`: fresh optimization result, including estimates, metrics, selected blocks, convergence status and solver bounds.
- `progress.json`, `run.log`: live progress and execution log.
- `block_contamination_diagnostics.json`: number of original/selected blocks containing any replaced observation, using the known simulated data.
- `results.json`: aggregate plus all seed results after execution ends. Main averages include converged fits only; `all_attempts` includes runtimes of unconverged fits.
- `run.py`: reproducible checkpoint-aware runner. Run with the robust_ip_estimation Python environment on this machine.
- `publish.py`: validates all 100 attempts and updates the root JSON files, Experiment 2 table and histogram. Run after `run.py` finishes.

Experiment 1 was cancelled at the user's request. The checkpointed background Experiment 8 is resumed when this benchmark exits. The main table's older methods retain their historical runtimes and worker configurations.
