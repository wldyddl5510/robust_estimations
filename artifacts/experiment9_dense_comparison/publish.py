"""Validate and publish the 100-seed dense comparison as Experiment 9."""
from pathlib import Path
import hashlib,importlib.util,json,math,re,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator,FuncFormatter,NullFormatter

OUT=Path(__file__).resolve().parent
REPO = next(parent for parent in Path(__file__).resolve().parents if (parent / 'IP_algorithm.py').is_file())
ARTIFACTS = REPO / 'artifacts'
spec=importlib.util.spec_from_file_location('experiment9_runner',OUT/'run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
payload=r.snapshot();cfg=payload['settings'];methods=cfg['methods']
if (OUT/'cancellation.json').exists():payload['status']='cancelled_by_user'
NAMES=dict(sample_mean='Sample mean + HT',coordinate_mom='Coordinate-wise MoM',
           geometric_mom='Geometric MoM',algorithm_1='Algorithm 1',
           random_block_selection='Block subsampling (21/21)',
           dl_clarabel='DL / Clarabel',dl_ptz='DL / specialized PTZ',cfb='CFB / Clarabel (T=100)')
assert methods==list(NAMES)
assert cfg['n']==1000 and cfg['d']==cfg['s']==10 and cfg['delta']==.05 and cfg['epsilon']==.01
assert cfg['seeds']==list(range(100)) and cfg['literature_K']==cfg['K_standard']==21
manifest=json.loads((OUT/'source_hashes.json').read_text())
digest=hashlib.sha256((OUT/'source_hashes.json').read_bytes()).hexdigest()
for name,expected in manifest.items():
    assert hashlib.sha256((OUT/'source'/name).read_bytes()).hexdigest()==expected,name
keys=[(x['seed'],x['method']) for x in payload['runs']]
assert len(keys)==len(set(keys))
diagnostics=[];pairs=[]
for seed in cfg['seeds']:
    data,truth,clean,amplitude=r.generate(seed)
    saved=np.load(OUT/f'data_seed_{seed}.npz')
    for name,value in [('data',data),('truth',truth),('clean_data',clean)]:
        np.testing.assert_array_equal(saved[name],value)
    mask=np.any(data!=clean,axis=1)
    assert data.shape==(1000,10) and np.count_nonzero(truth)==10 and mask.sum()==10
    runs={v['method']:v for v in payload['runs'] if v['seed']==seed}
    for name,v in runs.items():
        assert name in NAMES and v['source_manifest_sha256']==digest
        assert v['data_sha256']==hashlib.sha256(data.tobytes()).hexdigest()
        np.testing.assert_array_equal(v['truth'],truth)
        if 'estimate' in v:
            estimate=np.array(v['estimate'])
            assert estimate.shape==(10,) and np.isfinite(estimate).all()
            assert math.isclose(v['error'],float(np.linalg.norm(estimate-truth)),rel_tol=1e-12,abs_tol=1e-12)
            assert v['support_recovery']==float(np.mean(np.abs(estimate)>1e-8))
        if 'info' in v and name!='sample_mean':assert v['info']['K']==21
        if name=='random_block_selection' and 'info' in v:
            assert v['info']['K_sub_requested']==29 and v['info']['K_sub']==21
            assert v['info']['selected_block_indices']==list(range(21))
        if name.startswith('dl_') and v['status']=='completed':
            assert v['info']['backend']==name.split('_')[1]
            assert all(h['converged'] and h['relative_gap']<=1.0001e-4 for h in v['info']['history'])
        if name=='dl_ptz' and 'info' in v:
            assert v['info']['backend']=='ptz'
            assert v['info']['sdp_max_iter']==100000 and v['info']['sdp_eta']==1e-4
            reference=runs.get('dl_clarabel',{}).get('info',{}).get('history',[])
            for h in v['info']['history']:
                if h['outer']!=0:continue
                same=[q for q in reference if q['outer']==0 and math.isclose(q['rho'],h['rho'],rel_tol=1e-12)]
                if h is v['info']['history'][0]:assert same, 'Missing matched initial Clarabel call'
                for q in same:
                    assert h['lower']<=q['upper']+1e-7 and h['upper']>=q['lower']-1e-7
        if name=='cfb' and v['status']=='completed':
            assert v['info']['sdp_max_residual']<=1e-5
            assert all(h['distance_upper']-h['distance']<=1.0001e-5 for h in v['info']['history'])
    groups=np.array_split(np.random.default_rng(seed).permutation(1000),21)
    diagnostics.append(dict(seed=seed,replaced_rows=10,corrupted_blocks=sum(bool(mask[g].any()) for g in groups)))
    if all(n in runs and runs[n]['status']=='completed' for n in ('algorithm_1','random_block_selection')):
        pairs.append(dict(seed=seed,max_coordinate_difference=float(np.max(np.abs(np.array(runs['algorithm_1']['estimate'])-runs['random_block_selection']['estimate'])))))
for name in methods:
    runs=[v for v in payload['runs'] if v['method']==name and v['status']=='completed']
    if runs:
        for key in ('error','runtime'):
            values=[v[key] for v in runs]
            payload['summary'][name].update({f'std_{key}':float(np.std(values,ddof=1)) if len(values)>1 else 0.,
                f'min_{key}':float(min(values)),f'max_{key}':float(max(values))})
checks=['All 100 datasets regenerate exactly; ten replaced rows per seed; shared truth/data hashes and metrics verified.',
        'Eight requested method/backend variants; source manifests match; Algorithm 2 excluded.',
        'K=21 for all block methods; K_sub=29 capped to all 21 without replacement.',
        'Completed DL SDP gaps and CFB residual/radius tolerances verified; PTZ initial-call bounds checked against Clarabel.']
payload['validation']=checks
r.save(OUT/'progress.json',payload)
if payload['status'] in ('finished','finished_with_cfb_cancelled'):
    r.save(OUT/'results.json',payload)
    pidfile=OUT/'runner_pid.json'
    if pidfile.exists():
        pidrecord=json.loads(pidfile.read_text());pidrecord['status']='finished';r.save(pidfile,pidrecord)
r.save(OUT/'summary_statistics.json',{k:payload[k] for k in ('settings','summary','status','finished_attempts','total_attempts')})
r.save(OUT/'validation.json',dict(checks=checks,algorithm1_subsampling_pairs=pairs))
r.save(OUT/'contamination_diagnostics.json',diagnostics)

lines=['<!-- experiment9-dense:start -->','## Experiments 9','',
       'Experiment 1 distribution with **n=1000, d=s=10, delta=0.05, epsilon=0.01**, seeds 0–99. Centered skew-t: nu=2.5, shape 0.5*I+0.5*11\', skew vector 10*1/sqrt(d), amplitude Uniform(1,3) on all coordinates, adaptive attack strength 100 replacing ten rows. C=2, e_tol=1e-5, covariance input 2*lambda_max(Sigma)='+f"{cfg['lambda_upper']:.6f}"+'.',
       '', 'The full-support complexity is d. All block methods use K=21 and the same seed partition. Block subsampling requests 29 blocks and selects all 21 without replacement, so it does not reduce the problem in this configuration. Hard thresholding is the identity at s=d.',
       '', 'Timing: four concurrent fresh worker processes, one BLAS/solver thread each, matching the concurrency of the historical Experiment 1 baselines. Runtime includes estimator preprocessing and excludes imports/data generation/metrics; process scheduling and contention are included. The unrelated Experiment 8 is paused during measurement. No wall-clock timeout is imposed. Algorithm 1 retains its existing 1000-iteration limits; CFB uses T=100 and step size 1/20. Failed/unconverged runs are excluded from successful error/runtime averages and are reported separately.',
       '', 'DL / Clarabel means the Depersin–Lecue covSDPofMeans outer procedure with its covering SDP solved through CVXPY/Clarabel. It is a generic-SDP implementation, not the specialized nearly-linear backend. DL/PTZ additionally uses the specialized positive-SDP updates directly, with eta=1e-4 and a 100,000-update cap per decision call. This is the small-d factorized-Taylor/identity-sketch reference kernel with dense certificates, not the full high-dimensional nearly-linear implementation. It never substitutes Clarabel on failure. Algorithm 2 remains excluded. CFB uses the SDP-based distance/gradient routines; its completion flag means the requested finite iterations completed, not an outer accuracy certificate.',
       '', '| Method | Mean L2 error | Mean runtime (s) | Support recovery | Completed / original target | Unconverged / failed | Cancelled |',
       '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for name in methods:
    s=payload['summary'][name]
    error=f"{s['mean_error']:.6f}" if s['completed'] else '—'
    runtime=f"{s['mean_runtime']:.6f}" if s['completed'] else (f"{s['mean_attempt_runtime']:.6f} (unsuccessful attempts)" if 'mean_attempt_runtime' in s else '—')
    support=f"{100*s['mean_support_recovery']:.1f}%" if s['completed'] else '—'
    lines.append(f"| {NAMES[name]} | {error} | {runtime} | {support} | {s['completed']}/100 | {s['unconverged']}/{s['failed']} | {s.get('cancelled',0)} |")
lines.extend(['',f"Batch status: **{payload['status']}**, {payload['finished_attempts']}/{payload['total_attempts']} attempts recorded; {payload.get('cancelled_attempts',0)} remaining CFB attempts cancelled. Averages use completed seeds; completion counts are shown for every method.",''])
if payload.get('cancelled_attempts'):
    cancelled=payload['cancelled_attempts']
    count=payload['summary']['cfb']['completed']
    lines.extend([f"CFB was stopped by user request: only {count} completed seeds are retained in the averages; {cancelled} unfinished/unstarted seeds are cancelled and will not resume. This is a partial CFB sample, while the other method attempts cover all 100 seeds. [Cancellation record](artifacts/"+OUT.name+"/cfb_cancellation.json).",''])
if pairs:
    lines.extend([f"Algorithm 1 and block subsampling use identical blocks. Maximum coordinate difference across {len(pairs)} completed pairs: {max(v['max_coordinate_difference'] for v in pairs):.3g}.",''])
ptz=[v for v in payload['runs'] if v['method']=='dl_ptz']
calls=[c for v in ptz if 'info' in v for h in v['info']['history'] for c in h.get('calls',[])]
if calls:
    lines.extend([f"DL/PTZ: {payload['summary']['dl_ptz']['completed']}/{len(ptz)} recorded attempts completed. PTZ step formula evaluated at internal epsilon=eta/40 gives alpha={calls[0]['step_alpha']:.3e}; internal epsilon={calls[0]['internal_epsilon']:.3e}. The factor 40 is a conservative implementation choice for the decision wrapper, not a paper-prescribed constant. Unconverged attempts retain their bounds, update counts, and elapsed time in the raw records; their initial/current estimates are excluded from successful error/support averages. Unsuccessful-attempt time is not a runtime for obtaining a completed estimate.",''])
zeros=[v['seed'] for v in payload['runs'] if v['method']=='cfb' and v['status']=='completed' and v['support_recovery']==0]
if zeros:lines.extend([f"CFB returned its zero initial iterate under the minimum-estimated-distance return rule for seeds {zeros}. Full-support recovery is a literal nonzero-coordinate metric here, not sparse variable selection.",''])
incomplete=[dict(seed=v['seed'],method=v['method'],status=v['status'],termination=v.get('info',{}).get('termination'),runtime=v['runtime'],exception=v.get('exception')) for v in payload['runs'] if v['status']!='completed']
r.save(OUT/'incomplete_runs.json',incomplete)
lines.extend(['The literature block counts are practical choices: DL\'s sufficient condition K>=300*|O|=3000 exceeds n=1000, and CFB\'s prescribed ceil(3200*log(1/delta))=9587 also exceeds n. CFB\'s iid guarantee does not directly cover this adaptive contamination. These results concern the recorded implementations and settings. [DL](https://arxiv.org/abs/1906.03058), [CFB](https://proceedings.mlr.press/v99/cherapanamjeri19b.html), [PTZ](https://arxiv.org/abs/1201.5135).','',
       '[Per-seed records](artifacts/experiment9_dense_comparison/'+('results.json' if payload['status'] in ('finished','finished_with_cfb_cancelled') else 'progress.json')+'). [Settings](artifacts/experiment9_dense_comparison/config.json). [Validation](artifacts/experiment9_dense_comparison/validation.json). [Incomplete runs](artifacts/experiment9_dense_comparison/incomplete_runs.json). [Source hashes](artifacts/experiment9_dense_comparison/source_hashes.json). [Runner](artifacts/experiment9_dense_comparison/run.py).','',
       '![Experiment 9 error and runtime comparison](artifacts/experiment9_dense_comparison/comparison.png)','<!-- experiment9-dense:end -->'])
report=REPO / 'results.md';text=report.read_text()
text=text.replace('All setups use centered skew-t data','Unless stated otherwise, setups use centered skew-t data')
block='\n'.join(lines)+'\n'
if '<!-- experiment9-dense:start -->' in text:
    text=re.sub(r'<!-- experiment9-dense:start -->.*?<!-- experiment9-dense:end -->\n?',lambda _:block,text,flags=re.S)
else:text=text.rstrip()+'\n\n'+block
report.write_text(text)
if '--skip-plot' not in sys.argv:
    fig,axes=plt.subplots(1,2,figsize=(14,6.4),sharey=True,gridspec_kw={'wspace':.16})
    jitter=np.random.default_rng(41)
    for i,name in enumerate(methods):
        attempts=[v for v in payload['runs'] if v['method']==name]
        good=[v for v in attempts if v['status']=='completed']
        color='#B36B20' if name in ('dl_clarabel','dl_ptz','cfb') else '#126A83'
        for ax,key in zip(axes,('error','runtime')):
            if good:
                values=[v[key] for v in good]
                ax.barh(i,np.mean(values),height=.63,color=color,alpha=.72)
                ax.scatter(values,i+jitter.uniform(-.16,.16,len(good)),s=10,c='black',alpha=.4,zorder=3)
            elif attempts and key=='runtime':
                ax.barh(i,np.mean([v['runtime'] for v in attempts]),height=.63,color='none',edgecolor='#A33F37',hatch='///')
            else:ax.text(.03,i,'No completed estimate' if attempts else 'Pending',transform=ax.get_yaxis_transform(),va='center',color='#777')
    axes[0].set_yticks(np.arange(len(methods)),[f"{NAMES[n]} [{payload['summary'][n]['completed']}/100]" for n in methods]);axes[0].invert_yaxis()
    for ax,label in zip(axes,['L2 error (log scale)','Seconds (log scale)']):
        ax.set_xlabel(label);ax.set_xscale('log');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    axes[0].xaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    fig.suptitle('Experiment 9: n=1000, d=s=10, epsilon=0.01 (100 seeds)',fontsize=14,y=.98)
    fig.text(.26,.035,'Dots: completed seeds. Bars: means. Hatched: unsuccessful attempts. Four workers, one thread each.',fontsize=9)
    fig.subplots_adjust(left=.25,bottom=.13,top=.90,right=.98)
    fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
print(payload['status'],payload['finished_attempts'],'/',payload['total_attempts'],checks,flush=True)

