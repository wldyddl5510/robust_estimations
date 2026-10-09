"""Validate checkpoints and publish the dense sanity comparison."""
from pathlib import Path
import hashlib,importlib.util,json,math,platform,re,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
REPO = next(parent for parent in Path(__file__).resolve().parents if (parent / 'IP_algorithm.py').is_file())
ARTIFACTS = REPO / 'artifacts'
spec=importlib.util.spec_from_file_location('dense_runner',OUT/'run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
payload=r.snapshot()
NAMES=dict(sample_mean='Sample mean + HT',coordinate_mom='Coordinate-wise MoM + HT',
 geometric_mom='Geometric MoM + HT',random_mom='Random MoM (J=1000)',random_trimmed='Random trimmed (J=1000)',
 brute_force='Brute-force net',algorithm_1='Algorithm 1',dense_algorithm_1='Dense Algorithm 1',
 random_block_selection='Random block selection (9/21)',algorithm_2='Algorithm 2',
 partition_ls='Coordinate partition LS (r=1)',partition_max='Coordinate partition max (r=1)',
 haar_ls='Haar LS (r=2, J=10)',haar_max='Haar max (r=2, J=10)',
 dl_clarabel='DL / Clarabel',dl_ptz='DL / specialized PTZ',cfb='CFB / Clarabel (T=100)')

def validated():
    tests=[]
    expected=json.loads((OUT/'source_hashes.json').read_text())
    for name,digest in expected.items():
        assert hashlib.sha256((OUT/'source'/name).read_bytes()).hexdigest()==digest,name
    tests.append('Frozen source SHA256 matches manifest.')
    for seed in range(10):
        saved=np.load(OUT/f'data_seed_{seed}.npz')
        data,truth,clean,amp=r.generate(seed)
        np.testing.assert_array_equal(data,saved['data'])
        np.testing.assert_array_equal(truth,saved['truth'])
        np.testing.assert_array_equal(clean,saved['clean_data'])
        assert np.count_nonzero(np.any(data!=clean,axis=1))==10
        matching=[x for x in payload['runs'] if x['seed']==seed]
        for run in matching:
            assert run['data_sha256']==hashlib.sha256(data.tobytes()).hexdigest()
            np.testing.assert_array_equal(run['truth'],truth)
            if 'estimate' in run:
                assert math.isclose(run['error'],np.linalg.norm(np.array(run['estimate'])-truth),rel_tol=1e-12)
                assert run['support_recovery']==float(np.count_nonzero(np.abs(run['estimate'])>1e-8)/5)
            if run['method']=='dl_ptz' and 'info' in run:
                if run['status']=='completed':
                    assert all(x['converged'] and x['relative_gap']<=1.0001e-4 for x in run['info']['history'])
                reference=next((x for x in matching if x['method']=='dl_clarabel' and 'info' in x),None)
                if reference is not None:
                    for h in run['info']['history']:
                        same=[x for x in reference['info']['history'] if x['outer']==h['outer'] and math.isclose(x['rho'],h['rho'])]
                        for x in same:
                            assert h['lower']<=x['upper']+1e-7
                            assert h['upper']>=x['lower']-1e-7
            if run['method']=='dl_clarabel' and run['status']=='completed':
                assert all(x['converged'] and x['relative_gap']<=1.0001e-4 for x in run['info']['history'])
            if run['method']=='cfb' and run['status']=='completed':
                assert run['info']['sdp_max_residual']<=1e-5
                assert all(x['distance_upper']-x['distance']<=1.0001e-5 for x in run['info']['history'])
    tests.append('All data regenerated exactly; 10 rows replaced per seed; shared data/truth across methods; metrics recomputed.')
    tests.append('Completed literature SDP gaps/residuals/radius brackets satisfy recorded tolerances.')
    tests.append('PTZ bounds bracket the independent general-SDP bounds at matching centers/rho.')
    return tests

checks=validated()
for name in payload['settings']['methods']:
    runs=[x for x in payload['runs'] if x['method']==name and x['status']=='completed']
    if runs:
        for key in ('error','runtime'):
            values=[x[key] for x in runs]
            payload['summary'][name]['std_'+key]=float(np.std(values,ddof=1)) if len(values)>1 else 0.
            payload['summary'][name]['min_'+key]=float(min(values))
            payload['summary'][name]['max_'+key]=float(max(values))
        for key in ('sdp_calls','iterations'):
            values=[x['info'][key] for x in runs if key in x['info']]
            if values:payload['summary'][name]['mean_'+key]=float(np.mean(values))
r.save(OUT/'progress.json',payload)
if payload['status']=='finished':r.save(OUT/'results.json',payload)
r.save(OUT/'summary_statistics.json',{k:payload[k] for k in ('settings','summary','finished_attempts','total_attempts','status')})
ptz_calls=[c for x in payload['runs'] if x['method']=='dl_ptz' and 'info' in x
           for h in x['info']['history'] for c in h.get('calls',[])]
(OUT/'validation.json').write_text(json.dumps(dict(checks=checks,finished_attempts=payload['finished_attempts']),indent=2)+'\n')
# Extra seed-level contamination diagnostic, using the same balanced partition.
cont=[]
for seed in range(10):
    a=np.load(OUT/f'data_seed_{seed}.npz');mask=np.any(a['data']!=a['clean_data'],axis=1)
    blocks=np.array_split(np.random.default_rng(seed).permutation(100),21)
    contaminated=np.array([bool(mask[b].any()) for b in blocks])
    subset=json.loads((OUT/f'seed_{seed}_random_block_selection.json').read_text())
    selected=subset['info']['selected_block_indices']
    cont.append(dict(seed=seed,corrupted_blocks=int(contaminated.sum()),
        corrupted_selected_blocks=int(contaminated[selected].sum()),
        selected_blocks=len(selected),random_block_error=subset['error']))
(OUT/'contamination_diagnostics.json').write_text(json.dumps(cont,indent=2)+'\n')

lines=['<!-- non-sparse-sdp:start -->','## Sanity check 21: Non-sparse estimators and specialized SDP','',
'Parameters: $n=100$, $s=d=5$, $\\delta=0.05$, $\\epsilon=0.1$, $\\nu=2.5$, adversarial strength 100, $C=2$, and $e_{\\mathrm{tol}}=10^{-5}$. Seeds 0–9. Centered skew-t data use shape $0.5I+0.5\\mathbf{1}\\mathbf{1}^T$ and skew vector $10\\mathbf{1}/\\sqrt{5}$, as in the earlier 100-seed experiments. The true mean has amplitude Uniform(1,3) on all five coordinates. The repository attack replaces exactly ten rows; it is a heuristic, not a worst-case optimization. $\\overline\\lambda=2\\lambda_{\\max}(\\Sigma)='+f"{r.SETTINGS['lambda_upper']:.6f}"+'$.','',
'All methods use identical generated data within each seed, verified by SHA256. MoM/literature methods use balanced seeded blocks with $K=21$; random block selection uses $K_{\\mathrm{sub}}=9$ without replacement. Random directions use $\\widetilde J=1000$. Coordinate partition uses r=1, and Haar uses r=2,J=10. Hard thresholding is the identity at s=d.','',
'Runs are sequential, with one BLAS/solver thread on Apple M4. Other experiment workers are stopped during timing and restarted afterwards from their checkpoints. Runtime includes estimator initialization, compilation, and optimization, excluding imports, data generation, and metric calculation. Failed/unconverged attempts do not contribute error or support means. Their attempted runtime is displayed explicitly.','',
'| Method | Mean L2 error | Mean runtime (s) | Support recovery | Completed / attempted |','| --- | ---: | ---: | ---: | ---: |']
for name in payload['settings']['methods']:
    s=payload['summary'][name];attempted=sum(s.get(k,0) for k in ['completed','failed','unconverged'])
    error=f"{s['mean_error']:.6f}" if s.get('completed') else '—'
    rt=f"{s['mean_runtime']:.6f}" if s.get('completed') else (f"{s['mean_attempt_runtime']:.6f} (unsuccessful attempts)" if attempted else 'pending')
    sr=f"{100*s['mean_support_recovery']:.1f}%" if s.get('completed') else '—'
    lines.append(f'| {NAMES[name]} | {error} | {rt} | {sr} | {s["completed"]}/{attempted} |')
lines.extend(['',f'Batch status: **{payload["status"]}**, {payload["finished_attempts"]}/{payload["total_attempts"]} attempts saved. Each method is scheduled for ten seeds. Full support makes the support metric uninformative in this experiment.','',
'**What the literature implementations measure.** DL uses covSDPofMeans with a covering SDP backend. Clarabel is an explicitly labeled general-solver control; PTZ implements the specialized positive-SDP multiplicative updates and factorized Taylor exponential. The SDP relative accuracy is eta=1e-4; PTZ has a diagnostic cap of 100,000 updates per decision call. Its d=5 identity sketch and dense d-by-d certificates do not implement the full high-dimensional nearly-linear kernel. CFB uses the SDP Algorithms 4/5, step size 1/20 and 100 iterations. Its completion means the requested iterations finished, not an outer accuracy certificate; 1e-5 is its distance-search precision.','',
'**Theorem conditions.** Both literature methods use practical K=21. DL requires $K\\ge300|O|=3000$; CFB prescribes $K=\\lceil3200\\log(1/\\delta)\\rceil=9587$, exceeding n. CFB\'s iid theorem does not establish guarantees for this adaptive attack. Observed errors therefore describe this practical configuration. [DL paper](https://arxiv.org/abs/1906.03058), [CFB paper](https://proceedings.mlr.press/v99/cherapanamjeri19b.html), [PTZ solver](https://arxiv.org/abs/1201.5135).',''])
selection=OUT/'method_selection.json'
if selection.exists():
    excluded=json.loads(selection.read_text()).get('excluded_methods',[])
    if excluded:lines.extend(['Excluded methods: '+', '.join(NAMES[x] for x in excluded)+'. See [scope record](../artifacts/sanity_check/non_sparse_sdp_comparison/method_selection.json).',''])
interrupted=OUT/'interrupted_initial_attempt.json'
if interrupted.exists():
    attempt=json.loads(interrupted.read_text())
    lines.extend([f'An initial Algorithm 2 seed-0 attempt was interrupted after {attempt["elapsed"]:.1f}s to move that slow method to the end of the queue. It produced no estimate and is separate from the retained attempts in the table. [Interruption record](../artifacts/sanity_check/non_sparse_sdp_comparison/interrupted_initial_attempt.json).',''])
if ptz_calls:
    alpha=[c['step_alpha'] for c in ptz_calls]
    caps=[c['theoretical_iteration_bound'] for c in ptz_calls]
    gaps=[x['info']['history'][-1]['relative_gap'] for x in payload['runs'] if x['method']=='dl_ptz' and 'info' in x and x['info']['history']]
    lines.extend([f'PTZ diagnostics: alpha={min(alpha):.3e}; conservative decision-iteration bound {max(caps):.3e}. Actual retained decision calls used {sum(c["iterations"] for c in ptz_calls):,} updates total; final covering relative gaps range {min(gaps):.6f}–{max(gaps):.6f}. The bound is not an observed runtime or a lower bound on practical work: valid certificates may stop earlier.',''])
if all(sum(payload['summary'][name].get(k,0) for k in ('completed','failed','unconverged'))==10
       for name in ('algorithm_1','dl_clarabel','dl_ptz','cfb')):
    s=payload['summary'];a=s['algorithm_1'];d=s['dl_clarabel'];c=s['cfb'];p=s['dl_ptz']
    if a['completed'] and d['completed'] and c['completed']:
        lines.append(f'DL/Clarabel and CFB/Clarabel mean runtimes are respectively {d["mean_runtime"]/a["mean_runtime"]:.2f}× and {c["mean_runtime"]/a["mean_runtime"]:.2f}× Algorithm 1 in this setup. These ratios compare the actual implementations and stopping rules, not asymptotic complexity.')
    if p['completed']==0:
        lines.append('The specialized PTZ backend did not produce a certified final estimate in any seed. Its capped attempts are not successful-estimation runtimes; they show that this direct implementation at the paper accuracy is not practical under this update budget. They do not refute the nearly-linear asymptotic theorem. A scaling claim would require multiple n,d settings and the complete large-d kernel.')
    lines.append('')
zero_cfb=[x['seed'] for x in payload['runs'] if x['method']=='cfb' and x.get('support_recovery')==0]
if zero_cfb:
    support=100*payload['summary']['cfb']['mean_support_recovery']
    lines.extend([f'CFB returns its zero initial iterate for seeds {zero_cfb}, because it has the smallest estimated distance among that seed\'s 100 evaluated iterates. Consequently its literal nonzero-support recovery is {support:.1f}%. This follows the paper\'s best-iterate return rule; it is not a sparse feature-selection result.',''])
majority=[x for x in cont if 2*x['corrupted_selected_blocks']>=x['selected_blocks']]
lines.extend([f'Contamination diagnostic: the full K=21 partition has {min(x["corrupted_blocks"] for x in cont)}–{max(x["corrupted_blocks"] for x in cont)} contaminated blocks. The nine-block subset contains a contaminated majority in {len(majority)}/10 seeds. [Counts and paired errors](../artifacts/sanity_check/non_sparse_sdp_comparison/contamination_diagnostics.json) help interpret the random-block outliers; this is a diagnostic association, not a proof that contamination counts alone determine the error.',''])
lines.extend(['[Implementation details and deviations](../README_non_sparse.md). [Per-seed results and solver diagnostics](../artifacts/sanity_check/non_sparse_sdp_comparison/'+('results.json' if payload['status']=='finished' else 'progress.json')+'). [Summary statistics and ranges](../artifacts/sanity_check/non_sparse_sdp_comparison/summary_statistics.json). [Data/source verification](../artifacts/sanity_check/non_sparse_sdp_comparison/validation.json). [Environment](../artifacts/sanity_check/non_sparse_sdp_comparison/environment.json). [Reproducible runner](../artifacts/sanity_check/non_sparse_sdp_comparison/run.py).','',
'![Error and runtime by method](../artifacts/sanity_check/non_sparse_sdp_comparison/comparison.png)','<!-- non-sparse-sdp:end -->'])
report=REPO / 'sanity_check/sanity_check_results.md'
content=report.read_text();block='\n'.join(lines)+'\n'
if '<!-- non-sparse-sdp:start -->' in content:
    content=re.sub(r'<!-- non-sparse-sdp:start -->.*?<!-- non-sparse-sdp:end -->\n?',lambda _:block,content,flags=re.S)
else:content=content.rstrip()+'\n\n'+block
report.write_text(content)

if '--skip-plot' in sys.argv:
    print(payload['status'],payload['finished_attempts'],checks)
    sys.exit(0)

names=[n for n in payload['settings']['methods'] if sum(payload['summary'][n].get(k,0) for k in ['completed','failed','unconverged'])]
fig,axes=plt.subplots(1,2,figsize=(14,8),sharey=True,gridspec_kw={'wspace':.14})
y=np.arange(len(names));rng=np.random.default_rng(41)
for i,name in enumerate(names):
    runs=[x for x in payload['runs'] if x['method']==name]
    good=[x for x in runs if x['status']=='completed']
    color='#126A83' if not name.startswith(('dl_','cfb')) else '#B36B20'
    if good:
        for ax,key in [(axes[0],'error'),(axes[1],'runtime')]:
            ax.barh(i,np.mean([x[key] for x in good]),height=.63,color=color,alpha=.72)
            ax.scatter([x[key] for x in good],i+rng.uniform(-.13,.13,len(good)),s=12,c='black',alpha=.55,zorder=3)
    else:
        axes[0].text(.01,i,'No completed estimate',transform=axes[0].get_yaxis_transform(),va='center',fontsize=9,color='#A33F37')
        axes[1].barh(i,np.mean([x['runtime'] for x in runs]),height=.63,color='none',edgecolor='#A33F37',hatch='///')
axes[0].set_yticks(y,[NAMES[n] for n in names]);axes[0].invert_yaxis()
axes[0].set_xlabel('L2 error (log scale)');axes[0].set_xscale('log')
axes[1].set_xlabel('Seconds (log scale)');axes[1].set_xscale('log')
for ax in axes:
    ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Non-sparse centered skew-t sanity check: n=100, d=s=5',fontsize=15,y=.98)
fig.text(.32,.03,'Dots: seeds. Bars: means. Hatched runtime: unsuccessful attempts, not completed estimates.',fontsize=9)
fig.subplots_adjust(left=.30,bottom=.10,top=.93,right=.98)
fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
print(payload['status'],payload['finished_attempts'],checks)
