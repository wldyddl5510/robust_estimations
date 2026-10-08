"""Validate one dimension batch and publish its results and the d=5,10,20 comparison."""
import fcntl, hashlib, importlib.util, json, math, os, re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
REPO=OUT.parent
publication_lock=(REPO/'.dense_dimension_publication.lock').open('a')
fcntl.flock(publication_lock,fcntl.LOCK_EX)
spec=importlib.util.spec_from_file_location('dense_dimension_runner',OUT/'run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
def save_publication(path,value):
    # The running estimator scheduler uses its own temporary files.
    temp=path.with_name(path.name+f'.publication.{os.getpid()}.tmp')
    temp.write_text(json.dumps(r.clean(value),indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)
r.save=save_publication
payload=r.snapshot();cfg=payload['settings'];methods=cfg['methods']
d=cfg['d'];K=cfg['K_standard'];sub=cfg['K_sub_actual'];number=cfg['experiment_number']
names=dict(sample_mean='Sample mean + HT',coordinate_mom='Coordinate-wise MoM',
           geometric_mom='Geometric MoM',algorithm_1='Algorithm 1',
           random_block_selection=f'Block subsampling ({sub}/{K})',dl_clarabel='DL / Clarabel')
all_names=names.copy()
names={name:names[name] for name in methods}
assert set(methods)<=set(all_names) and cfg['s']==d and d in (5,20)
assert ('random_block_selection' in methods)==(d==5)
assert (cfg['n'],cfg['delta'],cfg['epsilon'],cfg['nu'],cfg['C'],cfg['tol'],cfg['strength'])==(1000,.05,.01,2.5,2,1e-5,100)
assert cfg['seeds']==list(range(100)) and cfg['literature_K']==K
odd=lambda x: int(np.ceil(x))+(int(np.ceil(x))%2==0)
assert K==odd(cfg['C']*max(d,cfg['epsilon']*cfg['n'],np.log(1/cfg['delta'])))
assert cfg['K_sub_requested']==odd(2*(d+np.log(4/cfg['delta'])))
assert sub==min(K,cfg['K_sub_requested'])
np.testing.assert_array_equal(cfg['shape'],.5*np.eye(d)+.5*np.ones((d,d)))
np.testing.assert_array_equal(cfg['skew'],10*np.ones(d)/np.sqrt(d))
expected_lambda=2*np.linalg.eigvalsh(r.skew_t_covariance(cfg['nu'],cfg['skew'],scale=np.array(cfg['shape'])))[-1]
assert math.isclose(cfg['lambda_upper'],expected_lambda,rel_tol=1e-14)
manifest=json.loads((OUT/'source_hashes.json').read_text())
assert manifest==json.loads((REPO/'experiment9_dense_comparison/source_hashes.json').read_text())
digest=hashlib.sha256((OUT/'source_hashes.json').read_bytes()).hexdigest()
for filename,expected in manifest.items():
    assert hashlib.sha256((OUT/'source'/filename).read_bytes()).hexdigest()==expected,filename
keys=[(x['seed'],x['method']) for x in payload['runs']]
assert len(keys)==len(set(keys))
diagnostics=[];pairs=[]
for seed in cfg['seeds']:
    data,truth,clean,amplitude=r.generate(seed)
    with np.load(OUT/f'data_seed_{seed}.npz') as saved:
        for name,value in [('data',data),('truth',truth),('clean_data',clean)]:
            np.testing.assert_array_equal(saved[name],value)
    mask=np.any(data!=clean,axis=1)
    assert data.shape==(1000,d) and np.count_nonzero(truth)==d and mask.sum()==10
    runs={v['method']:v for v in payload['runs'] if v['seed']==seed}
    for name,v in runs.items():
        assert name in names and v['source_manifest_sha256']==digest
        assert v['data_sha256']==hashlib.sha256(data.tobytes()).hexdigest()
        assert v['clean_data_sha256']==hashlib.sha256(clean.tobytes()).hexdigest()
        np.testing.assert_array_equal(v['truth'],truth)
        assert math.isfinite(v['runtime']) and v['runtime']>=0
        if 'estimate' in v:
            estimate=np.array(v['estimate'])
            assert estimate.shape==(d,) and np.isfinite(estimate).all()
            assert math.isclose(v['error'],float(np.linalg.norm(estimate-truth)),rel_tol=1e-12,abs_tol=1e-12)
            assert v['support_recovery']==float(np.mean(np.abs(estimate)>1e-8))
            assert v['status']==('completed' if v['info']['converged'] else 'unconverged')
        if 'info' in v and name!='sample_mean':assert v['info']['K']==K
        if name=='random_block_selection' and 'info' in v:
            info=v['info']
            assert info['K_sub_requested']==cfg['K_sub_requested'] and info['K_sub']==sub
            assert not info['sampling_with_replacement'] and info['subsample_multiplier']==2
            expected=np.sort(np.random.default_rng(np.random.SeedSequence(seed,spawn_key=(2,))).permutation(K)[:sub])
            np.testing.assert_array_equal(info['selected_block_indices'],expected)
        if name=='dl_clarabel' and v['status']=='completed':
            assert v['info']['backend']=='clarabel'
            assert all(h['converged'] and h['relative_gap']<=1.0001e-4 for h in v['info']['history'])
    groups=np.array_split(np.random.default_rng(seed).permutation(1000),K)
    diagnostics.append(dict(seed=seed,replaced_rows=10,corrupted_blocks=sum(bool(mask[g].any()) for g in groups)))
    if all(n in runs and runs[n]['status']=='completed' for n in ('algorithm_1','random_block_selection')):
        difference=float(np.max(np.abs(np.array(runs['algorithm_1']['estimate'])-runs['random_block_selection']['estimate'])))
        pairs.append(dict(seed=seed,max_coordinate_difference=difference))
        if K==sub:assert difference<1e-9,(seed,difference)
for name in methods:
    good=[v for v in payload['runs'] if v['method']==name and v['status']=='completed']
    if good:
        for key in ('error','runtime','support_recovery'):
            values=[v[key] for v in good]
            payload['summary'][name].update({f'std_{key}':float(np.std(values,ddof=1)) if len(values)>1 else 0.,f'min_{key}':float(min(values)),f'max_{key}':float(max(values))})
checks=['All 100 saved datasets regenerate exactly, with ten replaced rows per seed.',
        f'Settings differ from Experiment 9 only in dimension/support and derived quantities; {len(methods)} selected methods.',
        'Frozen estimator sources are byte-identical to Experiment 9; per-seed data hashes and metrics verified.',
        (f'Full-support block rule: K={K}, K_sub={sub}; subset and random stream verified.' if 'random_block_selection' in methods else f'Full-support block rule: K={K}; subsampling excluded because it would use all blocks.'),
        'Completed DL covering-SDP certificates meet the original relative-gap tolerance.']
payload['validation']=checks
r.save(OUT/'progress.json',payload)
if payload['status']=='finished':r.save(OUT/'results.json',payload)
r.save(OUT/'summary_statistics.json',{k:payload[k] for k in ('settings','summary','status','finished_attempts','total_attempts')})
r.save(OUT/'validation.json',dict(checks=checks,algorithm1_subsampling_pairs=pairs,validated_attempts=len(keys)))
r.save(OUT/'contamination_diagnostics.json',diagnostics)
r.save(OUT/'incomplete_runs.json',[dict(seed=v['seed'],method=v['method'],status=v['status'],runtime=v['runtime'],termination=v.get('info',{}).get('termination'),exception=v.get('exception')) for v in payload['runs'] if v['status']!='completed'])

def render_plot():
    fig,axes=plt.subplots(1,2,figsize=(13,5.4),sharey=True,gridspec_kw={'wspace':.16})
    jitter=np.random.default_rng(41)
    for i,name in enumerate(methods):
        good=[v for v in payload['runs'] if v['method']==name and v['status']=='completed']
        for ax,key in zip(axes,('error','runtime')):
            if good:
                values=[v[key] for v in good]
                ax.barh(i,np.mean(values),height=.62,color='#B36B20' if name=='dl_clarabel' else '#126A83',alpha=.72)
                ax.scatter(values,i+jitter.uniform(-.15,.15,len(values)),s=9,c='black',alpha=.35,zorder=3)
            else:ax.text(.03,i,'Pending',transform=ax.get_yaxis_transform(),va='center',color='#777')
    axes[0].set_yticks(range(len(methods)),[f"{names[n]} [{payload['summary'][n]['completed']}/100]" for n in methods]);axes[0].invert_yaxis()
    for ax,label in zip(axes,('L2 error (log scale)','Seconds (log scale)')):
        ax.set_xlabel(label);ax.set_xscale('log');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'Experiment {number}: n=1000, d=s={d}, epsilon=0.01 (100 seeds)',fontsize=14)
    fig.text(.27,.025,'Dots: completed seeds. Bars: means. Four workers, one thread each.',fontsize=9)
    fig.subplots_adjust(left=.27,bottom=.14,top=.9,right=.98)
    fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    pixels=(OUT/'comparison.png').read_bytes()
    digest=hashlib.sha256(pixels).hexdigest()
    filename=f"comparison_{payload['finished_attempts']}_{digest[:10]}.png"
    (OUT/filename).write_bytes(pixels)
    r.save(OUT/'plot_snapshot.json',dict(
        filename=filename,canonical_filename='comparison.png',image_sha256=digest,
        finished_attempts=payload['finished_attempts'],total_attempts=payload['total_attempts'],
        completed_seeds={name:payload['summary'][name]['completed'] for name in methods},
        mean_metrics={name:{key:payload['summary'][name].get('mean_'+key) for key in ('error','runtime')} for name in methods}))
    return filename

plot_file=render_plot()

def update_section(text,marker,body):
    start=f'<!-- {marker}:start -->';end=f'<!-- {marker}:end -->'
    block=start+'\n'+body.rstrip()+'\n'+end+'\n'
    if start in text:
        text,count=re.subn(re.escape(start)+r'.*?'+re.escape(end)+r'\n?',lambda _:block,text,flags=re.S)
        assert count==1
        return text
    return text.rstrip()+'\n\n'+block

lines=[f'## Experiments {number}','',
    f'Experiment 9 setup with **d=s={d}**; n=1000, delta=0.05, epsilon=0.01, seeds 0–99. Centered skew-t with nu=2.5, shape 0.5*I+0.5*11\', skew 10*1/sqrt(d), amplitude Uniform(1,3), adaptive attack strength 100 replacing ten rows. C=2, e_tol=1e-5. The covariance input is recomputed by the same rule: 2*lambda_max(Sigma)={cfg["lambda_upper"]:.6f}.','',
    f'Full-support complexity is d; the unchanged block rule gives K={K} for Algorithm 1, MoM and DL. '+(f'Block subsampling requests {cfg["K_sub_requested"]} blocks and selects {sub}/{K} without replacement. ' if 'random_block_selection' in methods else f'Block subsampling is excluded by user request because it would select all {K} blocks and give the same Algorithm 1 problem. ')+ 'Hard thresholding is the identity at s=d. Support recovery is the fraction of nonzero coordinates at threshold 1e-8, and is not informative about sparse variable selection here.','',
    f'The {len(methods)} selected methods use exactly the frozen Experiment 9 estimator sources. Algorithm 2, CFB and DL/PTZ are excluded by user request. DL uses the original generic Clarabel covering-SDP backend and eta=1e-4. Four concurrent fresh processes, one solver/BLAS thread each; runtime includes estimator preprocessing and excludes imports/data generation/metrics. Experiments 11 and 12 run sequentially; Experiment 8 was stopped and removed. No wall-clock timeout; Algorithm 1 retains its existing 1000-iteration limits.','',
    *(['Execution order changed by user request: DL/Clarabel is prioritized before remaining Algorithm 1 seeds. In-flight Algorithm 1 fits continue without pause; their freed worker slots run DL first. Existing completed fits are reused; at most four workers run concurrently.', ''] if cfg.get('execution_order') else []),
    '| Method | Mean L2 error | Mean runtime (s) | Support recovery | Completed / target | Unconverged / failed |',
    '| --- | ---: | ---: | ---: | ---: | ---: |']
for name in methods:
    s=payload['summary'][name]
    error=f"{s['mean_error']:.6f}" if s['completed'] else '—'
    runtime=f"{s['mean_runtime']:.6f}" if s['completed'] else '—'
    support=f"{100*s['mean_support_recovery']:.1f}%" if s['completed'] else '—'
    lines.append(f"| {names[name]} | {error} | {runtime} | {support} | {s['completed']}/100 | {s['unconverged']}/{s['failed']} |")
lines+=['',f"Batch status: **{payload['status']}**, {payload['finished_attempts']}/{payload['total_attempts']} attempts recorded. Metrics use completed estimates only; pending and failed attempts are excluded.",'']
if pairs and K==sub:
    lines+=[f'All {len(pairs)} completed Algorithm 1/subsampling pairs use identical blocks; maximum coordinate difference is {max(v["max_coordinate_difference"] for v in pairs):.3g}.','']
lines+=['The DL block count is the same practical comparison choice as Experiment 9; these measurements do not establish the paper’s nearly-linear runtime guarantee.','',
    f'[Per-seed records]({OUT.name}/'+('results.json' if payload['status']=='finished' else 'progress.json')+f'). [Settings]({OUT.name}/config.json). [Validation]({OUT.name}/validation.json). [Source hashes]({OUT.name}/source_hashes.json). [Runner]({OUT.name}/run.py).','']
lines+=[f'![Experiment {number} error and runtime comparison]({OUT.name}/{plot_file})','']
report=REPO/'results.md';text=update_section(report.read_text(),f'experiment{number}-dense','\n'.join(lines))

folders={5:'experiment11_dense_d5',10:'experiment9_dense_comparison',20:'experiment12_dense_d20'}
batches={}
for dimension,folder in folders.items():
    path=REPO/folder/'summary_statistics.json'
    if path.exists():batches[dimension]=json.loads(path.read_text())
compare=['### Dense dimension comparison: Experiments 11, 9 and 12','',
    'Common settings: n=1000, epsilon=.01, delta=.05, nu=2.5, C=2, tol=1e-5; seeds 0–99. Columns show **mean L2 error / mean runtime in seconds (completed seeds)**. The d=10 column reuses Experiment 9. Block counts are 21, 21 and 41 for d=5,10,20, so dimension and K both change at d=20. This is not a fixed-K dimension-only timing test.','',
    '| Method | d=s=5, K=21 | d=s=10, K=21 | d=s=20, K=41 |',
    '| --- | ---: | ---: | ---: |']
cross={}
for name in all_names:
    row=[];cross[name]={}
    for dimension in (5,10,20):
        summary=batches.get(dimension,{}).get('summary',{}).get(name,{})
        if name=='random_block_selection' and dimension in (10,20):
            dimension_cfg=json.loads((REPO/folders[dimension]/'config.json').read_text())
            block_count=dimension_cfg['K_standard']
            assert dimension_cfg['K_sub_actual']==block_count
            if dimension==20:
                summary={**batches.get(dimension,{}).get('summary',{}).get('algorithm_1',{}),
                         'source_method':'algorithm_1','reused_result':True}
            note=f'Same blocks as full Algorithm 1 ({block_count}/{block_count})'
            cross[name][str(dimension)]={**summary,'same_blocks_as':'algorithm_1','display_note':note}
            row.append(note)
            continue
        cross[name][str(dimension)]=summary
        row.append(f"{summary['mean_error']:.6f} / {summary['mean_runtime']:.6f} ({summary['completed']}/100)" if summary.get('completed') else 'Pending')
    label='Block subsampling' if name=='random_block_selection' else all_names[name]
    compare.append('| '+label+' | '+' | '.join(row)+' |')
text=re.sub(r'<!-- dense-dimension-comparison:start -->.*?<!-- dense-dimension-comparison:end -->\n?', '', text, flags=re.S)
text=update_section(text,'dense-dimension-comparison','\n'.join(compare))
report.write_text(text)
r.save(REPO/'dense_dimension_comparison.json',dict(experiments=folders,methods=cross,note='Reuse historical d=10 Experiment 9; new d=5 and d=20 use identical estimator sources. K changes with d.'))
print(payload['status'],payload['finished_attempts'],'/',payload['total_attempts'],flush=True)
