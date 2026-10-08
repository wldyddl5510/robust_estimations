"""Validate and publish the dense C sweep with the shared-C subsampling rule."""
import ast
import fcntl
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
REPO=OUT.parent
lock=(REPO/'.dense_dimension_publication.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX)
spec=importlib.util.spec_from_file_location('scale_runner',OUT/'run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
payload=r.snapshot();cfg=payload['settings'];methods=cfg['methods'];Cs=cfg['C_values']
base=REPO/cfg['baseline_directory']
names=dict(algorithm_1='Algorithm 1',random_block_selection='Block subsampling',dl_clarabel='DL / Clarabel',
           coordinate_mom='Coordinate-wise MoM',geometric_mom='Geometric MoM')
assert Cs==[1,2,3,4,5] and set(methods)==set(names)
assert (cfg['n'],cfg['d'],cfg['s'],cfg['epsilon'],cfg['delta'],cfg['nu'],cfg['strength'],cfg['tol'])==(1000,5,5,.01,.05,2.5,100,1e-5)
assert cfg['seeds']==list(range(100))
odd=lambda x:int(np.ceil(x))+(int(np.ceil(x))%2==0)
manifest=json.loads((OUT/'source_hashes.json').read_text())
baseline_manifest=json.loads((base/'source_hashes.json').read_text())
assert sorted(k for k in manifest if manifest[k]!=baseline_manifest[k])==cfg['changed_source_files']
for filename,digest in manifest.items():assert hashlib.sha256((OUT/'source'/filename).read_bytes()).hexdigest()==digest
# Reused MoM implementations are unchanged despite wrapper/CLI changes in their file.
for name in ['coordinate_mom_estimation','geometric_mom_estimation']:
    def function_text(path):
        return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
    assert function_text(OUT/'source/experiments.py')==function_text(base/'source/experiments.py')
digest=hashlib.sha256((OUT/'source_hashes.json').read_bytes()).hexdigest()
old_digest=hashlib.sha256((base/'source_hashes.json').read_bytes()).hexdigest()
reused=json.loads((OUT/'reused_records.json').read_text())
for relative,item in reused.items():
    assert hashlib.sha256((OUT/relative).read_bytes()).hexdigest()==item['sha256']
    assert (OUT/relative).read_bytes()==(REPO/item['source']).read_bytes()
by_key={(v['C'],v['seed'],v['method']):v for v in payload['runs']}
assert len(by_key)==len(payload['runs'])
diagnostics=[]
for seed in cfg['seeds']:
    data,truth,clean,amplitude=r.generate(seed)
    with np.load(OUT/f'data_seed_{seed}.npz') as saved:
        for name,value in [('data',data),('truth',truth),('clean_data',clean)]:np.testing.assert_array_equal(saved[name],value)
        assert float(saved['amplitude'])==amplitude
    assert (OUT/f'data_seed_{seed}.npz').read_bytes()==(base/f'data_seed_{seed}.npz').read_bytes()
    mask=np.any(data!=clean,axis=1)
    assert data.shape==(1000,5) and mask.sum()==10 and np.count_nonzero(truth)==5
    for C in Cs:
        case=cfg['cases'][str(C)];K=case['K_standard'];sub=case['K_sub_actual']
        assert K==odd(C*max(5,10,np.log(1/.05)))
        assert case['K_sub_requested']==odd(C*max(5,np.log(4/.05)))
        assert sub==min(K,case['K_sub_requested'])
        groups=np.array_split(np.random.default_rng(seed).permutation(1000),K)
        selected=np.sort(np.random.default_rng(np.random.SeedSequence(seed,spawn_key=(2,))).permutation(K)[:sub])
        corrupted=np.array([bool(mask[g].any()) for g in groups])
        diagnostics.append(dict(C=C,seed=seed,K=K,K_sub=sub,corrupted_blocks=int(corrupted.sum()),
            corrupted_selected_blocks=int(corrupted[selected].sum()),full_clean_majority=bool((~corrupted).sum()>K/2),
            subset_clean_majority=bool((~corrupted[selected]).sum()>sub/2)))
        for name in methods:
            v=by_key.get((C,seed,name))
            if v is None:continue
            is_reused=C==2 and name in cfg['reused_methods']
            assert v['source_manifest_sha256']==(old_digest if is_reused else digest)
            assert v['data_sha256']==hashlib.sha256(data.tobytes()).hexdigest()
            assert v['clean_data_sha256']==hashlib.sha256(clean.tobytes()).hexdigest()
            np.testing.assert_array_equal(v['truth'],truth)
            assert math.isfinite(v['runtime']) and v['runtime']>=0
            if 'estimate' not in v:continue
            estimate=np.array(v['estimate']);info=v['info']
            assert estimate.shape==(5,) and np.isfinite(estimate).all()
            assert math.isclose(v['error'],np.linalg.norm(estimate-truth),rel_tol=1e-12,abs_tol=1e-12)
            assert v['support_recovery']==float(np.mean(np.abs(estimate)>1e-8))
            assert v['status']==('completed' if info['converged'] else 'unconverged')
            assert info['K']==K
            if name!='dl_clarabel':assert info['C']==C
            if name=='random_block_selection':
                assert info['K_sub_requested']==case['K_sub_requested'] and info['K_sub']==sub
                assert info['subsample_multiplier']==C and info['subsample_complexity']==5
                assert info['subsampling_rule']=='shared_C_max_s_log_ed_over_s'
                assert not info['sampling_with_replacement']
                np.testing.assert_array_equal(info['selected_block_indices'],selected)
            if name in ('algorithm_1','random_block_selection') and v['status']=='completed':assert info['gap']<=cfg['tol']+1e-8
            if name=='dl_clarabel' and v['status']=='completed':
                assert info['backend']=='clarabel'
                assert all(h['converged'] and h['relative_gap']<=1.0001e-4 for h in info['history'])
diagnostic_summary={}
for C in Cs:
    values=[v for v in diagnostics if v['C']==C]
    diagnostic_summary[str(C)]=dict(mean_corrupted_blocks=float(np.mean([v['corrupted_blocks'] for v in values])),
        mean_corrupted_selected_blocks=float(np.mean([v['corrupted_selected_blocks'] for v in values])),
        full_clean_majority_seeds=sum(v['full_clean_majority'] for v in values),
        subset_clean_majority_seeds=sum(v['subset_clean_majority'] for v in values))
checks=['All saved datasets are byte-identical to Experiment 11 and regenerate exactly.',
    'Only the subsampling implementation and experiments wrapper/CLI changed; reused estimator functions are unchanged.',
    'C scales both full and subset counts: K=11,21,31,41,51; K_sub=5,11,15,21,25.',
    'Sampling is without replacement, capped at K, and independent of the original partition stream.',
    'C=2 reuses 400 unchanged records from Experiment 11; its subsampling is newly executed.',
    'Per-seed hashes, errors, support metrics, Algorithm 1 gaps and completed DL SDP certificates verified.']
payload['validation']=checks
r.save(OUT/'progress.json',payload)
r.save(OUT/'summary_statistics.json',{k:v for k,v in payload.items() if k!='runs'})
if payload['status']=='finished':r.save(OUT/'results.json',payload)
r.save(OUT/'validation.json',dict(checks=checks,validated_attempts=len(by_key)))
r.save(OUT/'contamination_diagnostics.json',dict(summary=diagnostic_summary,runs=diagnostics))
r.save(OUT/'incomplete_runs.json',[dict(C=v['C'],seed=v['seed'],method=v['method'],status=v['status'],exception=v.get('exception')) for v in payload['runs'] if v['status']!='completed'])

colors=['#126A83','#9467BD','#B36B20','#2CA02C','#D45151']
fig,axes=plt.subplots(1,2,figsize=(13,6))
jitter=np.random.default_rng(13)
for index,name in enumerate(methods):
    for ax,key in zip(axes,('error','runtime')):
        means=[]
        for C in Cs:
            values=[v[key] for v in payload['runs'] if v['C']==C and v['method']==name and v['status']=='completed']
            means.append(float(np.mean(values)) if values else np.nan)
            if values:ax.scatter(C+jitter.uniform(-.09,.09,len(values)),values,s=7,color=colors[index],alpha=.12)
        ax.plot(Cs,means,marker='o',lw=2,color=colors[index],label=names[name])
for ax,label in zip(axes,('L2 error','Runtime (seconds)')):
    ax.set_xticks(Cs);ax.set_xlabel('Block-count scale C');ax.set_ylabel(label);ax.set_yscale('log')
    ax.grid(alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Experiment 13: d=s=5, n=1000, epsilon=0.01; 100 seeds per C',fontsize=14)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,.045),frameon=False)
fig.text(.5,.015,'Lines: means. Dots: completed seeds. C=2 reuses four unchanged methods; subsampling uses the new rule.',ha='center',fontsize=9)
fig.subplots_adjust(bottom=.25,top=.88,wspace=.25,left=.08,right=.98)
fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
pixels=(OUT/'comparison.png').read_bytes();image_digest=hashlib.sha256(pixels).hexdigest()
plot_file=f"comparison_{payload['finished_attempts']}_{image_digest[:10]}.png"
(OUT/plot_file).write_bytes(pixels)
r.save(OUT/'plot_snapshot.json',dict(filename=plot_file,image_sha256=image_digest,finished_attempts=payload['finished_attempts'],
    completed_seeds={str(C):{name:payload['summary'][str(C)][name]['completed'] for name in methods} for C in Cs},
    mean_metrics={str(C):{name:{metric:payload['summary'][str(C)][name].get('mean_'+metric) for metric in ('error','runtime')} for name in methods} for C in Cs}))

lines=['## Experiments 13','',
    'Block-count scale comparison on the exact Experiment 11 datasets: d=s=5, n=1000, epsilon=.01, delta=.05, nu=2.5, adaptive attack strength 100, seeds 0–99. Centered skew-t with shape .5*I+.5*11\', skew 10*1/sqrt(5), amplitude Uniform(1,3), and covariance input 2*lambda_max(Sigma)=739.0987715913293. Algorithm 1 tolerance is 1e-5.','',
    'C=1,2,3,4,5 scales the unchanged full-block rule K=oddceil(C*max(d,epsilon*n,log(1/delta))). DL/Clarabel and both MoM baselines use this same K and seeded partition. Subsampling now requests oddceil(C*max(s*log(e*d/s),log(4/delta))), using the same C, and uniformly selects min(K,K_sub_requested) blocks without replacement. Its independent selection stream and original-order restoration are unchanged.','',
    '| C | Full K / DL K | Selected K_sub | Observations per full block |','| --- | ---: | ---: | ---: |']
for C in Cs:
    case=cfg['cases'][str(C)];K=case['K_standard'];low=cfg['n']//K;high=math.ceil(cfg['n']/K)
    lines.append(f"| {C} | {K} | {case['K_sub_actual']} | {low}–{high} |")
lines+=['',
    'Four fresh worker processes, one solver/BLAS thread each, no wall-clock timeout. Estimator runtime includes block construction and estimation and excludes imports/data loading/metrics; scheduling and contention are included. DL uses the original generic Clarabel covering-SDP backend with eta=1e-4. Existing Algorithm 1 iteration limits and heuristics are retained. C=2 reuses the 400 unchanged Algorithm 1, DL and MoM records from Experiment 11; subsampling is rerun because its count changes from 19 to 11. Algorithm 2, CFB, DL/PTZ and Haar are excluded. Frozen sources and per-record provenance distinguish old and new runs.','']
for metric,title in [('error','Mean L2 error'),('runtime','Mean runtime (seconds)')]:
    lines+=[f'**{title}**; each cell shows mean (completed seeds / 100).','',
        '| Method | C=1 | C=2 | C=3 | C=4 | C=5 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in methods:
        cells=[]
        for C in Cs:
            s=payload['summary'][str(C)][name]
            cells.append(f"{s['mean_'+metric]:.6f} ({s['completed']}/100)" if s['completed'] else 'Pending (0/100)')
        lines.append('| '+names[name]+' | '+' | '.join(cells)+' |')
    lines.append('')
lines+=['| C | Completed / target | Unconverged | Failed |','| --- | ---: | ---: | ---: |']
for C in Cs:
    values=payload['summary'][str(C)].values()
    lines.append(f"| {C} | {sum(v['completed'] for v in values)}/500 | {sum(v['unconverged'] for v in values)} | {sum(v['failed'] for v in values)} |")
lines+=['',f"Batch status: **{payload['status']}**, {payload['finished_attempts']}/{payload['total_attempts']} attempts recorded, including 400 reused records. Averages use completed estimates only. Per-seed support recovery is also recorded; at s=d, the fraction of nonzero coordinates is not informative about sparse variable selection.",'',
    '[Settings](experiment13_dense_block_scale/config.json). [Current records](experiment13_dense_block_scale/'+('results.json' if payload['status']=='finished' else 'progress.json')+'). [Validation](experiment13_dense_block_scale/validation.json). [Contaminated-block diagnostics](experiment13_dense_block_scale/contamination_diagnostics.json).','',
    f'![Experiment 13: error and runtime versus C](experiment13_dense_block_scale/{plot_file})','']
report=REPO/'results.md';text=report.read_text();start='<!-- experiment13-block-scale:start -->';end='<!-- experiment13-block-scale:end -->'
body=start+'\n'+'\n'.join(lines).rstrip()+'\n'+end+'\n'
if start in text:
    text,count=re.subn(re.escape(start)+r'.*?'+re.escape(end)+r'\n?',lambda _:body,text,flags=re.S);assert count==1
else:text=text.rstrip()+'\n\n'+body
report.write_text(text)
print(payload['status'],payload['finished_attempts'],'/',payload['total_attempts'],flush=True)
