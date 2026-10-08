"""Validate the PTZ retry and publish a separate paired comparison."""
from pathlib import Path
import hashlib,json,math,re,sys
import numpy as np
from run import OUT,PARENT,CFG,save
manifest=json.loads((OUT/'source_hashes.json').read_text())
digest=hashlib.sha256((OUT/'source_hashes.json').read_bytes()).hexdigest()
for name,expected in manifest.items():
    assert hashlib.sha256((OUT/'source'/name).read_bytes()).hexdigest()==expected,name
assert CFG['ptz_eta_divisor']==20 and CFG['ptz_decision_max_iter']==200000
assert CFG['epsilon']==.02 and CFG['literature_K']==41 and CFG['seeds']==list(range(100))
records=[];old_records=[];pairs=[]
tau=CFG['ptz_eta']/20
alpha=tau*tau/((1+math.log(41))*(1+10*tau))
for seed in CFG['seeds']:
    old=json.loads((PARENT/f'seed_{seed}_dl_ptz.json').read_text());old_records.append(old)
    f=OUT/f'seed_{seed}_dl_ptz.json'
    if not f.exists():continue
    v=json.loads(f.read_text());records.append(v)
    arrays=np.load(PARENT/f'data_seed_{seed}.npz')
    assert v['source_manifest_sha256']==digest and v['seed']==seed
    assert v['data_sha256']==old['data_sha256']==hashlib.sha256(arrays['data'].tobytes()).hexdigest()
    np.testing.assert_array_equal(v['truth'],arrays['truth'])
    if 'estimate' in v:
        estimate=np.array(v['estimate']);assert np.isfinite(estimate).all()
        assert math.isclose(v['error'],np.linalg.norm(estimate-arrays['truth']),rel_tol=1e-12)
        assert v['support_recovery']==float(np.mean(np.abs(estimate)>1e-8))
    if 'info' in v:
        info=v['info']
        assert info['backend']=='ptz' and info['K']==41
        assert info['sdp_max_iter']==200000 and info['sdp_eta']==1e-4
        for h in info['history']:
            for c in h.get('calls',[]):
                assert c['internal_epsilon']==tau and math.isclose(c['step_alpha'],alpha,rel_tol=1e-14)
                assert c['iterations']<=200000
        if v['status']=='completed':
            assert all(h['converged'] and h['relative_gap']<=1.0001e-4 for h in info['history'])
        refs=json.loads((PARENT/f'seed_{seed}_dl_clarabel.json').read_text())['info']['history']
        if info['history']:
            h=info['history'][0]
            matching=[q for q in refs if q['outer']==h['outer']==0 and math.isclose(q['rho'],h['rho'],rel_tol=1e-12)]
            assert matching
            for q in matching:assert h['lower']<=q['upper']+1e-7 and h['upper']>=q['lower']-1e-7
    pairs.append(dict(seed=seed,old_status=old['status'],new_status=v['status'],
        old_runtime=old['runtime'],new_runtime=v['runtime'],
        old_first_gap=(old.get('info',{}).get('history') or [{}])[0].get('relative_gap'),
        new_first_gap=(v.get('info',{}).get('history') or [{}])[0].get('relative_gap')))
def summarize(runs):
    good=[v for v in runs if v['status']=='completed']
    s={k:sum(v['status']==k for v in runs) for k in ('completed','unconverged','failed')}
    s['attempts']=len(runs)
    if runs:s['mean_attempt_runtime']=float(np.mean([v['runtime'] for v in runs]))
    if good:
        for k in ('runtime','error','support_recovery'):s['mean_'+k]=float(np.mean([v[k] for v in good]))
    return s
old,new=summarize(old_records),summarize(records)
state=json.loads((OUT/'launch_status.json').read_text()).get('state') if (OUT/'launch_status.json').exists() else 'prepared'
state='finished' if len(records)==100 else state
checks=['Frozen source hashes verified.',
        'Recorded retry data and truth exactly match the original Experiment 10 seeds.',
        'Recorded internal accuracy eta/20, alpha and 200000-update limit checked.',
        'Initial SDP bounds cross-checked with existing Clarabel solutions; completed estimates require the original eta=1e-4 gap.']
payload=dict(status=state,settings=CFG,finished_attempts=len(records),total_attempts=100,
    original_summary=old,summary=new,paired_comparison=pairs,runs=records,validation=checks)
save(OUT/'progress.json',payload)
save(OUT/'validation.json',dict(checks=checks,verified_records=len(records),step_alpha=alpha))
if state=='finished':
    save(OUT/'results.json',payload)
    if (OUT/'runner_pid.json').exists():
        pid=json.loads((OUT/'runner_pid.json').read_text());pid['status']='finished';save(OUT/'runner_pid.json',pid)
rel='experiment10_dense_eps002/ptz_eta20_200k'
lines=['<!-- experiment10-ptz-eta20:start -->','### Experiment 10: DL/PTZ accuracy and iteration-limit comparison','',
 'User-requested rerun on the same 100 seeds and identical contaminated data: n=1000, d=s=10, epsilon=.02, K=41. The SDP accuracy eta=1e-4 is unchanged. Internal accuracy changes from eta/40 to eta/20; the per-decision update cap changes from 100,000 to 200,000. The acceptance certificates and all data parameters remain unchanged. Original sources and results are retained.',
 '',f'PTZ step alpha={alpha:.12e}, internal accuracy={tau:.12e}. Four fresh workers with one solver/BLAS thread each. Remaining CFB fits were cancelled by user request; the completed CFB estimates are preserved. This PTZ rerun proceeds without concurrent CFB jobs. Experiment 8 resumes after the PTZ rerun.',
 '', '| PTZ configuration | Attempts / target | Completed | Unconverged / failed | Mean attempt time (s) | Mean L2 error of completed estimates |',
 '| --- | ---: | ---: | ---: | ---: | ---: |']
for label,s in [('eta/40; max 100,000',old),('eta/20; max 200,000',new)]:
    runtime=f"{s['mean_attempt_runtime']:.6f}" if s['attempts'] else '—'
    error=f"{s['mean_error']:.6f}" if s['completed'] else '—'
    lines.append(f"| {label} | {s['attempts']}/100 | {s['completed']} | {s['unconverged']}/{s['failed']} | {runtime} | {error} |")
lines+=['',f"Retry status: **{state}**, {len(records)}/100 attempts recorded. Attempt time includes unconverged runs and is not the runtime to obtain a completed estimate.",'']
valid=[p for p in pairs if p['old_first_gap'] is not None and p['new_first_gap'] is not None]
if valid:
    first=valid[0]
    lines+=[f"Seed {first['seed']} first-SDP relative gap: {first['old_first_gap']:.12g} -> {first['new_first_gap']:.12g} (target <=1e-4).",'']
if len(records)==100:
    lines+=[f"Mean attempt runtime ratio, new/original: {new['mean_attempt_runtime']/old['mean_attempt_runtime']:.3f}.",'']
lines +=[f'[Retry records]({rel}/'+('results.json' if state=='finished' else 'progress.json')+f'). [Settings]({rel}/config.json). [Validation]({rel}/validation.json). [Frozen source hashes]({rel}/source_hashes.json). [Runner]({rel}/run.py).','<!-- experiment10-ptz-eta20:end -->']
block='\n'.join(lines)+'\n'
report=PARENT.parent/'results.md';text=report.read_text()
if '<!-- experiment10-ptz-eta20:start -->' in text:
    text=re.sub(r'<!-- experiment10-ptz-eta20:start -->.*?<!-- experiment10-ptz-eta20:end -->\n?',lambda _:block,text,flags=re.S)
else:text=text.rstrip()+'\n\n'+block
report.write_text(text)
print(state,len(records),'/100',new,flush=True)
