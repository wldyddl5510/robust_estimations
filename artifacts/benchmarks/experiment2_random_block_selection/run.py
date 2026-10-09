"""Reproduce Experiment 2 with nonreplacement random block selection, seeds 0--99."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import concurrent.futures as cf
import contextlib
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
import numpy as np

REPO = next(parent for parent in Path(__file__).resolve().parents if (parent / 'IP_algorithm.py').is_file())
ARTIFACTS = REPO / 'artifacts'
OUT = ARTIFACTS / 'benchmarks/experiment2_random_block_selection'
SOURCE = OUT / 'source'
BACKGROUND = Path('/tmp/experiment26_eps005')
sys.path.insert(0, str(SOURCE))
from utils import sparse_skew_t_dist_data_generation, adversarial_sparse_contamination, mom_initialization
import random_block_selection as rbs
REFERENCE = json.loads((OUT / 'experiment_reference.json').read_text())
LOOKUP = {r['seed']: r for r in REFERENCE['runs']}
SETTINGS = {k: REFERENCE['settings'][k] for k in ('n', 'd', 's', 'epsilon', 'nu', 'delta', 'C', 'strength', 'shape', 'skew', 'lambda_upper', 'tol')}
SETTINGS.update(seeds=list(range(100)), workers=4, solver_threads=1, blas_threads=1,
                max_iter=1000, max_inner_iter=1000, wall_time_limit=None,
                subsample_multiplier=2, sampling_with_replacement=False,
                block_selection_seed='equal to data seed', block_selection_stream=[2],
                oracle_strategy='hybrid', oracle_formulation='indicator', perspective=False,
                scheduling='seeds 0--99; one fresh fit per seed',
                runtime='Includes block construction, sampling and optimization; excludes shared data generation.')
background_stopped = (OUT / 'background_resume_pending.json').exists()

def save(path, payload):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)

def generate(seed):
    p = SETTINGS
    np.random.seed(seed)
    loc = np.random.uniform(1, 3)
    data, truth = sparse_skew_t_dist_data_generation(p['n'], loc, p['nu'], p['d'], p['s'],
        scale=np.array(p['shape']), skew=np.array(p['skew']))
    data = adversarial_sparse_contamination(data, p['epsilon'], p['s'], strength=p['strength'])
    digest = hashlib.sha256(data.tobytes()).hexdigest()
    ref = LOOKUP[seed]
    assert digest == ref['data_sha256'] and truth.tolist() == ref['truth'] and loc == ref['loc']
    return data, truth, float(loc), digest

def fit(data, seed):
    p = SETTINGS
    return rbs.random_block_selection(data, p['s'], p['epsilon'], p['lambda_upper'], p['delta'],
        tol=p['tol'], seed=seed, C=p['C'], block_selection_seed=seed,
        subsample_multiplier=2, replace=False, max_iter=1000, max_inner_iter=1000)

def preflight():
    checks = []
    for seed in range(100):
        data, truth, loc, digest = generate(seed)
        p = SETTINGS
        original = mom_initialization(data, p['s'], p['epsilon'], p['lambda_upper'], p['delta'],
                                     p['tol'], seed, C=p['C'])[0]
        expected = np.sort(np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(2,))).permutation(41)[:23])
        def capture(means, s, **kwargs):
            assert np.array_equal(means, original[expected]) and means.shape == (23, 20)
            return np.zeros(p['d']), {}
        with patch.object(rbs, 'ip_estimation_from_block_means', side_effect=capture) as optimizer:
            _, info = fit(data, seed)
            assert optimizer.call_count == 1
        assert info['K'] == 41 and info['K_sub'] == info['unique_selected_blocks'] == info['K_sub_requested'] == 23
        assert info['selected_block_indices'] == expected.tolist() and not info['sampling_with_replacement']
        checks.append(dict(seed=seed, data_sha256=digest, original_block_means_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
                           selected_block_means_sha256=hashlib.sha256(original[expected].tobytes()).hexdigest(),
                           selected_block_indices=info['selected_block_indices'], K=41, K_sub_requested=23, K_sub=23))
    save(OUT / 'preflight.json', dict(all_100_data_hashes_match=True, all_100_selected_means_match_expected_subsets=True,
        note='Preflight captures optimizer inputs only; fresh optimization results are recorded separately.', checks=checks))
    print('PREFLIGHT: all 100 data hashes and uniform 23-of-41 subsets verified', flush=True)


def stop_background():
    global background_stopped
    pidfile = BACKGROUND / 'runner_pid.json'
    if not pidfile.exists():
        return
    root = json.loads(pidfile.read_text())['pid']
    output = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,command='], text=True)
    processes = {int(f[0]): (int(f[1]), f[2]) for line in output.splitlines() if len(f := line.strip().split(None, 2)) == 3}
    if root not in processes or str(BACKGROUND / 'resume_ht.py') not in processes[root][1]:
        return
    ids = {root}
    while True:
        new = ids | {pid for pid, (parent, cmd) in processes.items() if parent in ids}
        if new == ids:
            break
        ids = new
    for sig in (signal.SIGSTOP, signal.SIGKILL):
        for pid in sorted(ids):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
    background_stopped = True
    save(OUT / 'background_interruption.json', dict(root_pid=root, processes=sorted(ids), epoch=time.time(),
        reason='Isolated runtime measurement; completed checkpoints preserved; unfinished fits restart afterward.'))
    for path in BACKGROUND.glob('seed_*.status.json'):
        status = json.loads(path.read_text())
        if status['state'] == 'running':
            status['state'] = 'interrupted_for_experiment2_random_block_selection'
            save(path, status)
    print('Background experiment 8 checkpointed; active fits will restart afterward', flush=True)

def resume_background():
    if not background_stopped:
        return
    with (BACKGROUND / 'resume_after_experiment1_random_block.log').open('a') as log:
        proc = subprocess.Popen([sys.executable, '-u', str(BACKGROUND / 'resume_ht.py')], stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(BACKGROUND / 'runner_pid.json', dict(pid=proc.pid, started_epoch=time.time(), script='resume_ht.py', experiment_number=8,
        reason='Resume unfinished fits after Experiment 2 nonreplacement random block selection.'))
    save(OUT / 'background_resumed.json', dict(pid=proc.pid, epoch=time.time()))
    print('Resumed background experiment 8', proc.pid, flush=True)

def run_seed(seed):
    data, truth, loc, digest = generate(seed)
    started = time.time()
    save(OUT / f'seed_{seed}.status.json', dict(seed=seed, state='running', pid=os.getpid(), started_epoch=started))
    print('START', seed, flush=True)
    start = time.perf_counter()
    try:
        with (OUT / f'seed_{seed}.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            estimate, info = fit(data, seed)
        assert info['K'] == 41 and info['K_sub'] == info['unique_selected_blocks'] == info['K_sub_requested'] == 23
        assert not info['sampling_with_replacement'] and len(set(info['selected_block_indices'])) == 23
        result = dict(status='completed' if info['converged'] else 'unconverged', runtime=info['runtime'],
                      error=float(np.linalg.norm(estimate - truth)),
                      support_recovery=float(np.sum((np.abs(estimate) > 1e-8) & (truth != 0)) / SETTINGS['s']),
                      estimate=estimate.tolist(), info=info)
        historical = LOOKUP[seed]['methods'].get('Algorithm 1')
        if historical:
            result['full_algorithm1_comparison'] = dict(
                estimate_max_abs_difference=float(np.max(np.abs(estimate - np.array(historical['estimate'])))),
                error_difference=result['error'] - historical['error'],
                support_recovery_difference=result['support_recovery'] - historical['support_recovery'],
                historical_runtime=historical['runtime'],
                objective_difference=info['objective'] - historical['info']['objective'])
    except Exception as exc:
        result = dict(status='failed', elapsed=time.perf_counter() - start, exception=repr(exc), traceback=traceback.format_exc())
    record = dict(seed=seed, loc=loc, truth=truth.tolist(), data_sha256=digest,
                  started_epoch=started, finished_epoch=time.time(), **result)
    save(OUT / f'seed_{seed}.json', record)
    save(OUT / f'seed_{seed}.status.json', dict(seed=seed, state='finished', result_status=result['status'], pid=os.getpid()))
    print('DONE', seed, result['status'], result.get('runtime', result.get('elapsed')), flush=True)
    return seed

def snapshot(finished=False):
    runs = [json.loads(p.read_text()) for p in OUT.glob('seed_*.json') if not p.name.endswith('.status.json')]
    runs.sort(key=lambda r: r['seed'])
    good = [r for r in runs if r['status'] == 'completed']
    counts = {state: sum(r['status'] == state for r in runs) for state in ('completed', 'unconverged', 'failed')}
    summary = dict(counts)
    if good:
        summary.update({key: float(np.mean([r[key] for r in good])) for key in ('runtime', 'error', 'support_recovery')})
        summary['runtime_median'] = float(np.median([r['runtime'] for r in good]))
    active = [json.loads(p.read_text()) for p in OUT.glob('seed_*.status.json') if json.loads(p.read_text())['state'] == 'running']
    payload = dict(title='Experiment 2: random block selection without replacement', experiment_number=2,
                   status='finished' if finished else 'running', runner_pid=os.getpid(), last_updated_epoch=time.time(),
                   settings=SETTINGS, summary=summary, attempted_seeds=len(runs), total_seeds=100, active=active, runs=runs,
                   source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('*.py')},
                   reference_sha256=hashlib.sha256((OUT / 'experiment_reference.json').read_bytes()).hexdigest(),
                   platform=platform.platform(),
                   block_selection='K=41, requested K_sub=23, effective K_sub=23; uniform subset without replacement, restored to original block order.',
                   measurement='Fresh independent optimization runs; no historical runtimes or estimates reused.')
    save(OUT / 'progress.json', payload)
    save(ARTIFACTS / 'experiment2_random_block_selection_progress.json', payload)
    if finished:
        save(OUT / 'results.json', payload)
        save(ARTIFACTS / 'experiment2_random_block_selection_results.json', payload)
    print('PROGRESS', len(runs), '/100', counts, 'active', [a['seed'] for a in active], flush=True)

def main():
    preflight()
    stop_background()
    try:
        order = list(range(100))
        save(OUT / 'execution_order.json', order)
        with cf.ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(run_seed, seed): seed for seed in order if not (OUT / f'seed_{seed}.json').exists()}
            snapshot()
            while futures:
                done, _ = cf.wait(futures, timeout=30, return_when=cf.FIRST_COMPLETED)
                for future in done:
                    seed = futures.pop(future)
                    try:
                        future.result()
                    except Exception:
                        print('WORKER FAILURE', seed, traceback.format_exc(), flush=True)
                        raise
                snapshot()
        snapshot(finished=True)
    finally:
        resume_background()

if __name__ == '__main__':
    main()
