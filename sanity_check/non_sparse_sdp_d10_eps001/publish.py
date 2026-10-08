"""Validate and publish sanity check 22 without changing earlier checks."""
from pathlib import Path
import hashlib,importlib.util,json,math,re,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter

OUT=Path(__file__).resolve().parent
REPO=OUT.parent.parent
spec=importlib.util.spec_from_file_location('dense_runner_22',OUT/'run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
payload=r.snapshot();cfg=payload['settings'];methods=cfg['methods']
if (OUT/'cancellation.json').exists(): payload['status']='cancelled_by_user'
NAMES=dict(sample_mean='Sample mean + HT',coordinate_mom='Coordinate-wise MoM + HT',
 geometric_mom='Geometric MoM + HT',random_mom='Random MoM (J=1000)',random_trimmed='Random trimmed (J=1000)',
 brute_force='Brute-force net',algorithm_1='Algorithm 1 (s=d)',dense_algorithm_1='Algorithm 1 (dense solver)',
 random_block_selection=f"Random block selection ({cfg['K_sub_actual']}/{cfg['K_standard']})",
 partition_ls='Coordinate partition LS (r=1)',partition_max='Coordinate partition max (r=1)',
 haar_ls='Haar LS (r=2, J=10)',haar_max='Haar max (r=2, J=10)',
 dl_clarabel='DL / Clarabel',dl_ptz='DL / specialized PTZ',cfb='CFB / Clarabel (T=100)')
STATUS=('completed','unconverged','failed','skipped')
expected_K={name:cfg['K_standard'] for name in ('coordinate_mom','geometric_mom','random_mom','brute_force','algorithm_1','random_block_selection','partition_ls','partition_max')}
expected_K.update(partition_ls=cfg['K_partition'],partition_max=cfg['K_partition'],dense_algorithm_1=cfg['K_dense'],haar_ls=cfg['K_haar'],haar_max=cfg['K_haar'],
                  dl_clarabel=cfg['literature_K'],dl_ptz=cfg['literature_K'],cfb=cfg['literature_K'])
checks=[]
assert 'algorithm_2' not in methods
expected=json.loads((OUT/'source_hashes.json').read_text())
for name,digest in expected.items():
    assert hashlib.sha256((OUT/'source'/name).read_bytes()).hexdigest()==digest,name
checks.append('Frozen source files match SHA256 manifest; Algorithm 2 excluded.')
cont=[];paired=[]
for seed in cfg['seeds']:
    saved=np.load(OUT/f'data_seed_{seed}.npz')
    data,truth,clean,amp=r.generate(seed)
    for key,value in [('data',data),('truth',truth),('clean_data',clean)]:
        np.testing.assert_array_equal(saved[key],value)
    assert data.shape==(cfg['n'],cfg['d']) and np.count_nonzero(truth)==cfg['s']
    mask=np.any(data!=clean,axis=1)
    assert mask.sum()==int(cfg['epsilon']*cfg['n'])
    matching={x['method']:x for x in payload['runs'] if x['seed']==seed}
    for name,run in matching.items():
        assert run['data_sha256']==hashlib.sha256(data.tobytes()).hexdigest()
        np.testing.assert_array_equal(run['truth'],truth)
        if 'estimate' in run:
            estimate=np.array(run['estimate'])
            assert estimate.shape==truth.shape and np.isfinite(estimate).all()
            assert math.isclose(run['error'],float(np.linalg.norm(estimate-truth)),rel_tol=1e-12,abs_tol=1e-12)
            assert run['support_recovery']==float(np.count_nonzero((np.abs(estimate)>1e-8)&(truth!=0))/cfg['s'])
        if 'info' in run and name in expected_K:
            assert run['info']['K']==expected_K[name],(name,run['info']['K'])
        if name.startswith('haar') and 'info' in run:
            assert run['info']['box_K']==cfg['K_haar_box']
        if name=='brute_force' and run['status']=='skipped':
            assert run['reason']=='existing_net_size_guard' and run['net_size']>run['max_net_points']
        if name.startswith('dl_') and run['status']=='completed':
            assert all(h['converged'] and h['relative_gap']<=1.0001*cfg['ptz_eta'] for h in run['info']['history'])
        if name=='dl_ptz' and 'info' in run and 'dl_clarabel' in matching and 'info' in matching['dl_clarabel']:
            for h in run['info']['history']:
                same=[x for x in matching['dl_clarabel']['info']['history'] if x['outer']==h['outer'] and math.isclose(x['rho'],h['rho'])]
                for ref in same:
                    assert h['lower']<=ref['upper']+1e-7 and h['upper']>=ref['lower']-1e-7
        if name=='cfb' and run['status']=='completed':
            assert run['info']['sdp_max_residual']<=1e-5
            assert all(h['distance_upper']-h['distance']<=1.0001*cfg['tol'] for h in run['info']['history'])
    item=dict(seed=seed,replaced_rows=int(mask.sum()),corrupted_blocks={})
    for k in sorted(set(expected_K.values())):
        groups=np.array_split(np.random.default_rng(seed).permutation(cfg['n']),k)
        item['corrupted_blocks'][str(k)]=sum(bool(mask[g].any()) for g in groups)
    cont.append(item)
    if all(name in matching and matching[name]['status']=='completed' for name in ('algorithm_1','random_block_selection')):
        subset=matching['random_block_selection']['info']
        assert subset['K_sub']==cfg['K_sub_actual'] and subset['K_sub_requested']==cfg['K_sub_requested']
        assert subset['selected_block_indices']==list(range(cfg['K_standard']))
        paired.append(dict(seed=seed,max_coordinate_difference=float(np.max(np.abs(np.array(matching['algorithm_1']['estimate'])-matching['random_block_selection']['estimate'])))))
checks.extend(['All datasets regenerate exactly; one replaced row per seed; shared data/truth and metrics verified.',
               'Actual block counts match each method configuration.',
               'Completed literature SDP gaps/residuals/radius brackets pass tolerances; PTZ bounds checked against general SDP at matching calls.'])
for name in methods:
    runs=[x for x in payload['runs'] if x['method']==name and x['status']=='completed']
    if runs:
        for key in ('error','runtime'):
            values=[x[key] for x in runs]
            payload['summary'][name].update({f'std_{key}':float(np.std(values,ddof=1)) if len(values)>1 else 0.,
                                           f'min_{key}':float(min(values)),f'max_{key}':float(max(values))})
        for key in ('iterations','sdp_calls'):
            values=[x['info'][key] for x in runs if key in x['info']]
            if values:payload['summary'][name]['mean_'+key]=float(np.mean(values))
payload['validation']=checks
r.save(OUT/'progress.json',payload)
if payload['status']=='finished':r.save(OUT/'results.json',payload)
r.save(OUT/'summary_statistics.json',{k:payload[k] for k in ('settings','summary','status','finished_attempts','total_attempts')})
r.save(OUT/'validation.json',dict(checks=checks,random_block_full_pairs=paired))
r.save(OUT/'contamination_diagnostics.json',cont)

lines=['<!-- non-sparse-sdp-d10-eps001:start -->',
       '## Sanity check 22: Non-sparse d=s=10, epsilon=0.01','',
       'Repeat of sanity check 21 with **d=s=10** and **epsilon=0.01**. Algorithm 2 is excluded throughout. Other parameters stay n=100, delta=0.05, nu=2.5, adversarial strength=100, C=2, e_tol=1e-5, and seeds 0–9. Centered skew-t uses shape 0.5*I+0.5*11\' and skew vector 10*1/sqrt(d); the seed-specific true amplitude is Uniform(1,3) on every coordinate. The attack replaces exactly one row per seed. The covariance input is 2*lambda_max(Sigma)='+f"{cfg['lambda_upper']:.6f}"+'. The data are shared exactly within a seed/method comparison; samples are not nested across the two dimensions.','',
       f"The corrected full-support complexity d is used: Algorithm 1 and ordinary MoM methods use K={cfg['K_standard']}; dense Algorithm 1 uses K={cfg['K_dense']} (complexity d); coordinate partitions r=1 use K={cfg['K_partition']}; Haar r=2,J=10 uses K={cfg['K_haar']} (delta/J), with original-data box K={cfg['K_haar_box']}. DL and CFB explicitly use K={cfg['literature_K']}. Random block selection requests {cfg['K_sub_requested']} blocks but uses all {cfg['K_sub_actual']} after min(K,K_sub), without replacement. Random directions use tilde_J=1000. Random trimmed uses raw data, trimming k=5 observations per tail. Hard thresholding is the identity at full support.",'',
       'At s=d, both Algorithm 1 implementations already optimize over all coordinates. Following the user correction, the common block rule uses complexity d at full support (s*log(d/s) remains the default for s<d). Both implementations now use the same K=21 and partition; the dense wrapper bypasses outer support selection. Random block selection also uses complexity d in K_sub. Earlier records affected by the zero-complexity full-support rule are preserved under archive/pre_full_support_rule_correction and excluded from this table. Unaffected raw-data methods, coordinate projections, the dense solver, and K=21 literature fits are retained with their original source manifest recorded.', '',
       'Timing: sequential fresh worker processes, one BLAS/solver thread, with the pre-existing experiment 8 stopped during measurement and resumed afterwards. Runtime includes estimator setup/compilation, excluding imports/data generation/metrics. Error and support means use completed runs only. CFB completion means its 100 requested iterations finished, not an outer error certificate; e_tol controls its distance search.','',
       '| Method | K | Mean L2 error | Mean runtime (s) | Support recovery | Completed / recorded |',
       '| --- | ---: | ---: | ---: | ---: | ---: |']
for name in methods:
    s=payload['summary'][name];count=sum(s.get(k,0) for k in STATUS)
    error=f"{s['mean_error']:.6f}" if s['completed'] else '—'
    rt=f"{s['mean_runtime']:.6f}" if s['completed'] else ('skipped (net-size guard)' if s.get('skipped')==count and count else (f"{s['mean_attempt_runtime']:.6f} (unsuccessful attempts)" if count else 'pending'))
    support=f"{100*s['mean_support_recovery']:.1f}%" if s['completed'] else '—'
    lines.append(f'| {NAMES[name]} | {expected_K.get(name,"raw data")} | {error} | {rt} | {support} | {s["completed"]}/{count} |')
lines.extend(['',f'Batch status: **{payload["status"]}**, {payload["finished_attempts"]}/{payload["total_attempts"]} scheduled records resolved. Algorithm 2 has no scheduled fits.',''])
skips=[x for x in payload['runs'] if x['status']=='skipped']
if skips:
    size=skips[0]['net_size']
    lines.extend([f'Brute-force: the default radius-1/4 net contains {size:,} points, above the unchanged 200,000-point guard (the float64 direction array alone would require {size*cfg["d"]*8/2**30:.1f} GiB). All such records are resource-guard skips, not successful fits or estimation runtimes.',''])
if paired:
    lines.extend([f'Algorithm 1 versus random block selection: all original blocks are selected; maximum coordinate difference in the {len(paired)} completed paired seeds is {max(x["max_coordinate_difference"] for x in paired):.3g}. Any runtime difference measures selection/bookkeeping and run variability.',''])
ptz=[x for x in payload['runs'] if x['method']=='dl_ptz']
calls=[c for x in ptz if 'info' in x for h in x['info']['history'] for c in h.get('calls',[])]
if calls:
    lines.extend([f'Specialized PTZ: eta=1e-4, at most 100,000 updates per decision call; paper step alpha={calls[0]["step_alpha"]:.3e}. {payload["summary"]["dl_ptz"]["completed"]}/{len(ptz)} retained attempts produced a completed estimate. This is the same small-d factorized-Taylor/identity-sketch implementation with dense d-by-d certificates as sanity check 21; the full high-dimensional nearly-linear kernel is not implemented. Unfinished calls are not successful-estimation runtimes.',''])
cfb=[x for x in payload['runs'] if x['method']=='cfb' and x['status']=='completed']
zeros=[x['seed'] for x in cfb if x['support_recovery']==0]
if zeros:lines.extend([f'CFB returns its zero initial iterate in seeds {zeros}, following the paper\'s minimum-estimated-distance return rule. Full-support recovery here is a literal nonzero-coordinate metric, not sparse feature selection.',''])
lines.extend(['The literature methods still use practical block counts: DL\'s sufficient condition K>=300*|O|=300 exceeds n=100, while CFB prescribes ceil(3200*log(1/delta))=9587 and its iid theorem does not cover this adaptive contamination. Results describe this implementation/configuration, not those guarantees. [DL](https://arxiv.org/abs/1906.03058), [CFB](https://proceedings.mlr.press/v99/cherapanamjeri19b.html), [PTZ](https://arxiv.org/abs/1201.5135).','',
       'Both d and epsilon change relative to sanity check 21, and several block counts change too. Cross-setting runtime ratios cannot isolate a dimension effect or establish nearly-linear scaling.',''])
if payload['status']=='finished':
    a=payload['summary']['dense_algorithm_1']
    for name in ('dl_clarabel','cfb'):
        s=payload['summary'][name]
        if a['completed']==10 and s['completed']==10:
            lines.append(f'{NAMES[name]} / Algorithm 1 dense solver mean runtime (both K={cfg["K_dense"]}): {s["mean_runtime"]/a["mean_runtime"]:.2f}x; mean L2 errors {s["mean_error"]:.6f} / {a["mean_error"]:.6f}.')
    lines.append('')
folder=OUT.name
lines.extend([f'[Per-seed results]({folder}/'+('results.json' if payload['status']=='finished' else 'progress.json')+f'). [Summary statistics]({folder}/summary_statistics.json). [Validation]({folder}/validation.json). [Settings]({folder}/config.json). [Frozen source hashes]({folder}/source_hashes.json). [Runner]({folder}/run.py). [Implementation details](../README_non_sparse.md).','',
              f'![Non-sparse d=10 comparison]({folder}/comparison.png)','<!-- non-sparse-sdp-d10-eps001:end -->'])
report=REPO/'sanity_check/sanity_check_results.md';text=report.read_text();block='\n'.join(lines)+'\n'
if '<!-- non-sparse-sdp-d10-eps001:start -->' in text:
    text=re.sub(r'<!-- non-sparse-sdp-d10-eps001:start -->.*?<!-- non-sparse-sdp-d10-eps001:end -->\n?',lambda _:block,text,flags=re.S)
else:text=text.rstrip()+'\n\n'+block
report.write_text(text)
if '--skip-plot' not in sys.argv:
    names=[n for n in methods if sum(payload['summary'][n].get(k,0) for k in STATUS)]
    fig,axes=plt.subplots(1,2,figsize=(14,8),sharey=True,gridspec_kw={'wspace':.14})
    jitter=np.random.default_rng(41)
    for i,name in enumerate(names):
        runs=[x for x in payload['runs'] if x['method']==name]
        good=[x for x in runs if x['status']=='completed']
        color='#126A83' if not name.startswith(('dl_','cfb')) else '#B36B20'
        if good:
            for ax,key in [(axes[0],'error'),(axes[1],'runtime')]:
                ax.barh(i,np.mean([x[key] for x in good]),height=.63,color=color,alpha=.72)
                ax.scatter([x[key] for x in good],i+jitter.uniform(-.13,.13,len(good)),s=12,c='black',alpha=.55,zorder=3)
        elif all(x['status']=='skipped' for x in runs):
            for ax in axes:ax.text(.01,i,'Skipped: net-size guard',transform=ax.get_yaxis_transform(),va='center',fontsize=9,color='#777777')
        else:
            axes[0].text(.01,i,'No completed estimate',transform=axes[0].get_yaxis_transform(),va='center',fontsize=9,color='#A33F37')
            axes[1].barh(i,np.mean([x['runtime'] for x in runs]),height=.63,color='none',edgecolor='#A33F37',hatch='///')
    axes[0].set_yticks(np.arange(len(names)),[NAMES[n] for n in names]);axes[0].invert_yaxis()
    for ax,label in zip(axes,['L2 error (log scale)','Seconds (log scale)']):
        ax.set_xlabel(label);ax.set_xscale('log');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    axes[0].xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    fig.suptitle('Centered skew-t: n=100, d=s=10, epsilon=0.01 (10 seeds)',fontsize=15,y=.98)
    fig.text(.32,.03,'Dots: seeds. Bars: means. Hatched runtime: unsuccessful attempts. Algorithm 2 excluded.',fontsize=9)
    fig.subplots_adjust(left=.30,bottom=.10,top=.93,right=.98);fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
print(payload['status'],payload['finished_attempts'],checks,flush=True)
