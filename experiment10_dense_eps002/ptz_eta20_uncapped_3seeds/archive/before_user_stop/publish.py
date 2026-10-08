"""Publish live elapsed time separately from completed DL/PTZ runtime."""
from pathlib import Path
import hashlib,json,math,re,time
import numpy as np
from run import OUT,PARENT,CFG,save
assert CFG['seeds']==[0,1,2] and CFG['ptz_decision_max_iter'] is None
assert CFG['ptz_eta_divisor']==20 and CFG['literature_K']==41
manifest=json.loads((OUT/'source_hashes.json').read_text())
for name,sha in manifest.items():assert hashlib.sha256((OUT/'source'/name).read_bytes()).hexdigest()==sha
digest=hashlib.sha256((OUT/'source_hashes.json').read_bytes()).hexdigest()
runs=[];live=[]
for seed in CFG['seeds']:
    f=OUT/f'seed_{seed}_dl_ptz.json'
    if f.exists():
        v=json.loads(f.read_text());runs.append(v)
        data=np.load(PARENT/f'data_seed_{seed}.npz')
        assert v['data_sha256']==hashlib.sha256(data['data'].tobytes()).hexdigest()
        assert v['source_manifest_sha256']==digest
        np.testing.assert_array_equal(v['truth'],data['truth'])
        if 'info' in v:
            info=v['info'];assert info['sdp_max_iter'] is None
            for h in info['history']:
                for c in h.get('calls',[]):
                    assert c['iteration_limit'] is None and c['internal_epsilon']==5e-6
            if v['status']=='completed':
                assert all(h['converged'] and h['relative_gap']<=1.0001e-4 for h in info['history'])
                assert math.isclose(v['error'],np.linalg.norm(np.array(v['estimate'])-data['truth']),rel_tol=1e-12)
        hist=v.get('info',{}).get('history',[])
        last=hist[-1].get('calls',[])[-1] if hist and hist[-1].get('calls') else {}
        live.append(dict(seed=seed,state=v['status'],elapsed_seconds=v['runtime'],
            converged_runtime=v['runtime'] if v['status']=='completed' else None,
            decision_iterations=last.get('iterations'),decision_gap=last.get('relative_gap'),
            error=v.get('error') if v['status']=='completed' else None))
    else:
        sf=OUT/f'seed_{seed}_dl_ptz.status.json'
        v=json.loads(sf.read_text()) if sf.exists() else dict(seed=seed,state='pending')
        q=v.get('progress',{})
        live.append(dict(seed=seed,state=v['state'],elapsed_seconds=v.get('elapsed_seconds'),
            converged_runtime=None,decision_iterations=q.get('iteration'),decision_gap=q.get('relative_gap'),
            decision_call_index=v.get('decision_call_index'),updated_epoch=v.get('updated_epoch')))
state=json.loads((OUT/'launch_status.json').read_text())['state'] if (OUT/'launch_status.json').exists() else 'prepared'
if len(runs)==3:state='finished'
summary={k:sum(v['status']==k for v in runs) for k in ('completed','unconverged','failed')}
summary['pending_or_running']=3-len(runs)
payload=dict(status=state,settings=CFG,finished_attempts=len(runs),total_attempts=3,summary=summary,
             runs=runs,live=live,updated_epoch=time.time())
save(OUT/'progress.json',payload)
if state=='finished':
    save(OUT/'results.json',payload)
    if (OUT/'runner_pid.json').exists():
        v=json.loads((OUT/'runner_pid.json').read_text());v['status']='finished';save(OUT/'runner_pid.json',v)
rel='experiment10_dense_eps002/ptz_eta20_uncapped_3seeds'
lines=['<!-- experiment10-ptz-uncapped:start -->','### Experiment 10: uncapped DL/PTZ diagnostic (3 seeds)','',
 'Same saved Experiment 10 data for seeds 0, 1, 2: n=1000, d=s=10, epsilon=.02, K=41. SDP eta=1e-4, internal eta/20=5e-6, tol=1e-5. No SDP iteration limit and no wall-clock timeout. The theoretical PTZ iteration bound is recorded but not imposed; the covering-SDP decision-call cap is also removed in this mode. The DL outer algorithm and its own iteration/bisection rules are unchanged.',
 '', 'Three simultaneous fresh workers, one BLAS/solver thread each. Estimator runtime excludes imports/data loading. Experiment 8 checkpoints are preserved and its unfinished fits are paused for timing; it resumes when this batch finishes. CFB remains cancelled. Prior 100-seed batches used four workers, so their wall times are not an exact matched-concurrency comparison.',
 '', '| Seed | Status | Current/last decision iterations | Decision gap U/L-1 | Elapsed seconds | Completed DL/PTZ runtime (s) |',
 '| ---: | --- | ---: | ---: | ---: | ---: |']
for v in live:
    n=str(v['decision_iterations']) if v.get('decision_iterations') is not None else '—'
    gap=f"{v['decision_gap']:.10g}" if v.get('decision_gap') is not None else '—'
    elapsed=f"{v['elapsed_seconds']:.1f}" if v.get('elapsed_seconds') is not None else '—'
    runtime=f"{v['converged_runtime']:.6f}" if v.get('converged_runtime') is not None else '—'
    lines.append(f"| {v['seed']} | {v['state']} | {n} | {gap} | {elapsed} | {runtime} |")
lines+=['',f"Batch status: **{state}**. Completed estimates: {summary['completed']}/3; unsuccessful terminal attempts: {summary['failed']+summary['unconverged']}/3. Live status is refreshed every 30 seconds. Elapsed time on running rows is not time to convergence. The decision gap is the currently running decision problem's feasible-bound gap; the enclosing SDP must meet eta=1e-4 before DL can proceed.",
 '',f'[Live records]({rel}/progress.json). [Settings]({rel}/config.json). [Frozen sources]({rel}/source_hashes.json). [Runner]({rel}/run.py). [Numerical tests]({rel}/test_validation.json).',
 '<!-- experiment10-ptz-uncapped:end -->']
block='\n'.join(lines)+'\n'
report=PARENT.parent/'results.md';text=report.read_text()
if '<!-- experiment10-ptz-uncapped:start -->' in text:
    text=re.sub(r'<!-- experiment10-ptz-uncapped:start -->.*?<!-- experiment10-ptz-uncapped:end -->\n?',lambda _:block,text,flags=re.S)
else:text=text.rstrip()+'\n\n'+block
report.write_text(text)
print(state,[(v['seed'],v['state'],v.get('elapsed_seconds'),v.get('decision_iterations'),v.get('decision_gap')) for v in live],flush=True)

