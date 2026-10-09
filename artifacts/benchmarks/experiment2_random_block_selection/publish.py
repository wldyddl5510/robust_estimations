"""Validate and publish the 100-seed nonreplacement Experiment 2 benchmark."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl_experiment2_random_block_selection')
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

REPO = next(parent for parent in Path(__file__).resolve().parents if (parent / 'IP_algorithm.py').is_file())
ARTIFACTS = REPO / 'artifacts'
OUT = ARTIFACTS / 'benchmarks/experiment2_random_block_selection'
NAME = 'Random block selection (WOR)'
sys.path.insert(0, str(REPO))
from plot_histograms import plot_histograms, COLORS

def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)

def main():
    result = json.loads((OUT / 'results.json').read_text())
    reference = json.loads((OUT / 'experiment_reference.json').read_text())
    assert result['status'] == 'finished' and result['attempted_seeds'] == 100
    assert sorted(r['seed'] for r in result['runs']) == list(range(100))
    assert result['reference_sha256'] == hashlib.sha256((OUT / 'experiment_reference.json').read_bytes()).hexdigest()
    for name, digest in result['source_sha256'].items():
        assert hashlib.sha256((OUT / 'source' / name).read_bytes()).hexdigest() == digest
    refs = {r['seed']: r for r in reference['runs']}
    for run in result['runs']:
        seed = run['seed']
        assert run['data_sha256'] == refs[seed]['data_sha256'] and run['truth'] == refs[seed]['truth']
        if run['status'] == 'failed':
            continue
        info = run['info']
        selected = np.sort(np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(2,))).permutation(41)[:23])
        assert info['K'] == 41 and info['K_sub'] == info['K_sub_requested'] == info['unique_selected_blocks'] == 23
        assert not info['sampling_with_replacement'] and info['selected_block_indices'] == selected.tolist()
        assert info['tol'] == 1e-5 and info['runtime'] == run['runtime'] and run['runtime'] > 0
        estimate, truth = np.array(run['estimate']), np.array(run['truth'])
        assert np.isclose(run['error'], np.linalg.norm(estimate - truth), rtol=0, atol=1e-12)
        assert run['support_recovery'] == float(np.sum((np.abs(estimate) > 1e-8) & (truth != 0)) / 5)
        assert run['status'] == ('completed' if info['converged'] else 'unconverged')
        if info['converged']:
            assert info['gap'] <= info['gap_tolerance'] == 1e-5
    good = [r for r in result['runs'] if r['status'] == 'completed']
    summary = result['summary']
    assert summary['completed'] == len(good)
    for key in ('runtime', 'error', 'support_recovery'):
        assert np.isclose(summary[key], np.mean([r[key] for r in good]), rtol=0, atol=1e-12)
    summary.update(runtime_median=float(np.median([r['runtime'] for r in good])),
                   runtime_max=max(r['runtime'] for r in good),
                   full_support_seeds=sum(r['support_recovery'] == 1 for r in good))
    timed = [r['runtime'] if 'runtime' in r else r['elapsed'] for r in result['runs']]
    result['all_attempts'] = dict(count=100, runtime_mean=float(np.mean(timed)), runtime_total=float(np.sum(timed)),
                                  note='Includes unconverged fits and elapsed time of failed attempts; main summary includes only converged fits.')
    result['wall_time'] = max(r['finished_epoch'] for r in result['runs']) - min(r['started_epoch'] for r in result['runs'])
    result['validation'] = dict(all_100_data_hashes_match=True, all_subsets_reproduced=True,
                                metrics_recomputed=True, converged_gaps_verified=True)
    diagnostics = json.loads((OUT / 'block_contamination_diagnostics.json').read_text())
    majority_seeds = diagnostics['selected_contaminated_majority_seeds']
    result['block_contamination_diagnostics'] = dict(
        definition='A contaminated block contains at least one replaced observation.',
        selected_contaminated_majority_seeds=majority_seeds,
        source='benchmarks/experiment2_random_block_selection/block_contamination_diagnostics.json')
    for label, majority in [('majority', True), ('no_majority', False)]:
        group = [r for r in good if (r['seed'] in majority_seeds) == majority]
        result['block_contamination_diagnostics'][label] = dict(
            converged_count=len(group),
            error=float(np.mean([r['error'] for r in group])) if group else None,
            support_recovery=float(np.mean([r['support_recovery'] for r in group])) if group else None)
    random = json.loads((ARTIFACTS / 'experiment2_random_j1000_results.json').read_text())
    assert sorted(r['seed'] for r in random['runs']) == list(range(100))
    for run in random['runs']:
        assert run['data_sha256'] == refs[run['seed']]['data_sha256'] and run['truth'] == refs[run['seed']]['truth']
    result['random_direction_reference_sha256'] = hashlib.sha256((ARTIFACTS / 'experiment2_random_j1000_results.json').read_bytes()).hexdigest()
    baseline_runs = {r['seed']: r for r in reference['runs']}
    result['matched_comparisons'] = {}
    for name in reference['summary']:
        pairs = [(r, baseline_runs[r['seed']]['methods'][name]) for r in good
                 if name in baseline_runs[r['seed']]['methods']]
        if pairs:
            result['matched_comparisons'][name] = dict(seeds=[r['seed'] for r, _ in pairs],
                random_block_selection={key: float(np.mean([r[key] for r, _ in pairs])) for key in ('error', 'runtime', 'support_recovery')},
                baseline={key: float(np.mean([b[key] for _, b in pairs])) for key in ('error', 'runtime', 'support_recovery')})
    for key, name in [('random_mom', 'Random MoM (tilde J=1000)'), ('random_trimmed', 'Random trimmed mean (tilde J=1000)')]:
        lookup = {r['seed']: r for r in random['runs']}
        pairs = [(r, lookup[r['seed']]['methods'][key]) for r in good]
        result['matched_comparisons'][name] = dict(seeds=[r['seed'] for r, _ in pairs],
            random_block_selection={metric: float(np.mean([r[metric] for r, _ in pairs])) for metric in ('error', 'runtime', 'support_recovery')},
            baseline={metric: float(np.mean([b[metric] for _, b in pairs])) for metric in ('error', 'runtime', 'support_recovery')})
    save(OUT / 'results.json', result)
    save(ARTIFACTS / 'experiment2_random_block_selection_results.json', result)

    # Preserve every historical method and add the independently measured method.
    combined = json.loads((ARTIFACTS / 'experiment2_results.json').read_text())
    combined_before = copy.deepcopy(combined)
    lookup = {r['seed']: r for r in result['runs']}
    for run in combined['runs']:
        new = lookup[run['seed']]
        assert new['data_sha256'] == run['data_sha256'] and new['truth'] == run['truth']
        run['methods'][NAME] = {k: v for k, v in new.items() if k not in ('seed', 'loc', 'truth', 'data_sha256')}
    combined['summary'][NAME] = summary
    combined['random_block_selection_settings'] = result['settings']
    combined['random_block_selection_source_sha256'] = result['source_sha256']
    combined['random_block_selection_results'] = 'experiment2_random_block_selection_results.json'
    for old, new in zip(combined_before['runs'], combined['runs']):
        for name, method in old['methods'].items():
            if name != NAME:
                assert new['methods'][name] == method
    save(ARTIFACTS / 'experiment2_results.json', combined)

    errors = json.loads((ARTIFACTS / 'experiment2_errors.json').read_text())
    new_errors = [lookup[s]['error'] if lookup[s]['status'] == 'completed' else None for s in errors['seeds']]
    errors['errors'] = {NAME: new_errors, **{k: v for k, v in errors['errors'].items() if k != NAME}}
    save(ARTIFACTS / 'experiment2_errors.json', errors)
    COLORS[NAME] = '#117766'
    plot_histograms(errors['title'], errors['seeds'], errors['errors'], ARTIFACTS / 'experiment2_error_histograms.png', columns=4)

    markdown = (REPO / 'results.md').read_text()
    start = markdown.index('## Experiments 2\n')
    end = markdown.index('## Experiments 3\n', start)
    section = markdown[start:end]
    row = f"| {NAME} | {summary['error']:.6f} | {summary['runtime']:.6f} | {summary['support_recovery']:.6f} | {len(good)}/100 |"
    lines = [line for line in section.splitlines() if not line.startswith(f'| {NAME} |')]
    index = next(i for i, line in enumerate(lines) if line.startswith('| --- ')) + 1
    lines.insert(index, row)
    section = '\n'.join(lines) + '\n\n'
    section = section.replace('All five methods completed all 100 seeds.', 'All five original baseline methods completed all 100 seeds.')
    failures = [r for r in result['runs'] if r['status'] != 'completed']
    status = f"All {len(good)} seeds converged." if len(good) == 100 else f"{len(good)}/100 seeds converged; unconverged or failed fits are excluded from the averages and histogram."
    notes = [f"- Seed {r['seed']}: {r['status']}; " + (f"{r['info']['iterations']} outer iterations, gap {r['info']['gap']:.8g}, tolerance {r['info']['tol']:.8g}, runtime {r['runtime']:.2f} s." if 'info' in r else r['exception']) for r in failures]
    paragraph = ("<!-- experiment2-random-block-selection -->\n"
        "Random block selection (WOR = without replacement): `random_block_selection` uses coefficient 2, "
        "$K_{\\mathrm{sub}}=\\min\\{K,\\operatorname{oddceil}(2[s\\log(d/s)+\\log(4/\\delta)])\\}=23$ of the original $K=41$ blocks. "
        "Each seed chooses a uniform 23-block subset without replacement using a separate `SeedSequence(seed, spawn_key=(2,))` stream; selected rows retain original block order. "
        "The subset is chosen once before Algorithm 1 optimization. Seeds 0–99 reproduce the historical data and true means exactly (all hashes verified).\n\n"
        "These are fresh optimization runs, with four concurrent workers and one solver/BLAS thread per worker, tolerance $10^{-5}$, hybrid/indicator oracle, "
        "and outer/inner iteration limits of 1,000; no wall-time cutoff. Runtime includes block construction, selection and optimization, excluding data generation. "
        "Background Experiment 8 was stopped during measurement and resumed from checkpoints afterward.\n\n"
        f"{status} Mean runtime is computed over converged fits; median {summary['runtime_median']:.6f} s, maximum {summary['runtime_max']:.6f} s. "
        f"Across all 100 attempts, including unconverged or failed attempts, mean runtime is {result['all_attempts']['runtime_mean']:.6f} s. "
        "Runtime comparisons use separate measurement batches; random-direction baselines used one sequential worker. "
        "Full Algorithm 1 was not run for Experiment 2.\n\n"
        f"A data diagnostic found that {len(majority_seeds)}/100 subsets contain at least 12 contaminated blocks out of 23 "
        "(a block is counted as contaminated if it contains any replaced observation); this diagnostic does not determine solver convergence. "
        "Per-seed estimates, metrics, selected block indices, bounds, settings and source hashes: "
        "[experiment2_random_block_selection_results.json](artifacts/experiment2_random_block_selection_results.json). "
        "Reproduction and checkpoints: [benchmarks/experiment2_random_block_selection](artifacts/benchmarks/experiment2_random_block_selection).\n"
        + ('\n' + '\n'.join(notes) + '\n' if notes else '')
        + "<!-- /experiment2-random-block-selection -->\n\n")
    a = section.find('<!-- experiment2-random-block-selection -->')
    if a != -1:
        b = section.index('<!-- /experiment2-random-block-selection -->', a) + len('<!-- /experiment2-random-block-selection -->')
        section = section[:a] + section[b:].lstrip('\n')
    marker = '![L2 error histograms for Experiment 2]'
    section = section.replace(marker, paragraph + marker, 1)
    (REPO / 'results.md').write_text(markdown[:start] + section + markdown[end:])
    shutil.copy2(__file__, OUT / 'publish.py') if Path(__file__).resolve() != (OUT / 'publish.py').resolve() else None
    print(json.dumps(dict(summary=summary, failures=[r['seed'] for r in failures], validation=result['validation']), indent=2))

if __name__ == '__main__':
    main()
