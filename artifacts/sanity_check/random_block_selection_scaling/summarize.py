import json,hashlib,shutil,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = next(parent for parent in Path(__file__).resolve().parents if (parent / 'IP_algorithm.py').is_file())
ARTIFACTS = R / 'artifacts';OUT=ARTIFACTS /'sanity_check/random_block_selection_scaling';doc=R / 'sanity_check/sanity_check_results.md'
raw=json.loads((OUT/'results.json').read_text());ref=json.loads((OUT/'full_algorithm1_reference.json').read_text());assert raw['status']=='finished' and len(raw['runs'])==50
configs=[(100,5),(100,10),(100,20),(200,10),(300,10)];reference={(r['n'],r['d'],r['seed']):r for r in ref['runs']};summary={};paired=[]
for r in raw['runs']:
 f=reference[r['n'],r['d'],r['seed']];assert r['data_sha256']==f['data_sha256'] and r['truth']==f['truth'];paired.append(dict(n=r['n'],d=r['d'],seed=r['seed'],data_sha256=r['data_sha256'],truth=r['truth'],full=f,random_block_selection=r))
for n,d in configs:
 records=[p for p in paired if (p['n'],p['d'])==(n,d)];assert sorted(p['seed'] for p in records)==list(range(10));good=[p for p in records if p['full']['status']==p['random_block_selection']['status']=='completed']
 def aggregate(method):
  records=[p[method] for p in good];times=np.array([r['runtime'] for r in records]);return dict(mean_runtime=float(times.mean()),median_runtime=float(np.median(times)),runtime_min=float(times.min()),runtime_max=float(times.max()),mean_error=float(np.mean([r['error'] for r in records])),median_error=float(np.median([r['error'] for r in records])),mean_support_recovery=float(np.mean([r['support_recovery'] for r in records])))
 a=aggregate('full');b=aggregate('random_block_selection');ratios=np.array([p['full']['runtime']/p['random_block_selection']['runtime'] for p in good]);delta_error=np.array([p['random_block_selection']['error']-p['full']['error'] for p in good]);delta_support=np.array([p['random_block_selection']['support_recovery']-p['full']['support_recovery'] for p in good])
 summary[f'n{n}_d{d}']=dict(n=n,d=d,K=good[0]['full']['K'],K_sub=good[0]['random_block_selection']['info']['K_sub'],mean_unique_selected_blocks=float(np.mean([p['random_block_selection']['info']['unique_selected_blocks'] for p in good])),matched_completed=len(good),full=a,random_block_selection=b,speedup_ratio_of_mean_runtimes=a['mean_runtime']/b['mean_runtime'],median_paired_speedup=float(np.median(ratios)),mean_error_difference=float(delta_error.mean()),mean_support_recovery_difference=float(delta_support.mean()),random_has_lower_error_seeds=sum(delta_error<0).item(),random_has_higher_support_recovery_seeds=sum(delta_support>0).item(),random_has_lower_support_recovery_seeds=sum(delta_support<0).item())
raw['comparison_summary']=summary;raw['paired_runs']=paired;raw['comparison_note']='Full Algorithm 1 results are reused from the earlier 50-fit benchmark; its optimization core was subsequently extracted unchanged into ip_estimation_from_block_means. Each comparison uses identical data hashes and truth. Both timings used one sequential worker and one solver/BLAS thread with other experiment workers stopped, but they were collected in separate batches. Same seed values, independent block-selection RNG; one selection per seed, no best-of-repeat selection. Statistics use paired converged fits.'
(OUT/'results.json').write_text(json.dumps(raw,indent=2)+'\n')
fig,axes=plt.subplots(2,3,figsize=(13.2,7.8));colors={'full':'#4477aa','random_block_selection':'#cc6677'};labels={'full':'Algorithm 1 (all blocks)','random_block_selection':'Random block selection'}
metrics=[('runtime','Runtime (seconds, log scale)'),('error','L2 error'),('support_recovery','Support recovery')]
for row,group,xlabel,title in [(0,[(100,5),(100,10),(100,20)],'Dimension d','Fixed n=100'),(1,[(100,10),(200,10),(300,10)],'Sample size n','Fixed d=10')]:
 for col,(metric,ylabel) in enumerate(metrics):
  ax=axes[row,col]
  for method,offset in [('full',-.08),('random_block_selection',.08)]:
   means=[]
   for j,(n,d) in enumerate(group):
    vals=[p[method][metric] for p in paired if (p['n'],p['d'])==(n,d) and p['full']['status']==p['random_block_selection']['status']=='completed'];means.append(np.mean(vals));jitter=np.linspace(-.035,.035,len(vals));ax.scatter(j+offset+jitter,vals,s=13,color=colors[method],alpha=.35,zorder=2)
   ax.plot(np.arange(3)+offset,means,'o-',color=colors[method],label=labels[method],linewidth=1.8,markersize=5,zorder=3)
  ax.set_xticks(range(3),[str(d if row==0 else n) for n,d in group]);ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.set_title(title);ax.grid(axis='y',alpha=.2)
  if metric=='runtime':ax.set_yscale('log')
  if metric=='support_recovery':ax.set_ylim(-.05,1.05)
for col in [0,1]:
 lo=min(ax.get_ylim()[0] for ax in axes[:,col]);hi=max(ax.get_ylim()[1] for ax in axes[:,col])
 if col==1:lo=0
 for ax in axes[:,col]:ax.set_ylim(lo,hi)
handles,legend_labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.5,.967),ncol=2,frameon=False)
fig.suptitle('Full Algorithm 1 vs random block selection | s=2, delta=0.05, epsilon=0.1',y=.995);fig.text(.5,.006,'Lines: 10-seed means. Faint points: individual seeds. Runtime includes all block construction and selection.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.025,1,.925));fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)

def table(group,x):
 text=f'| {x} | Method | Blocks used / draws | Mean runtime (s) | Mean L2 error | Mean support recovery | Matched seeds |\n| ---: | --- | ---: | ---: | ---: | ---: | ---: |\n'
 for n,d in group:
  q=summary[f'n{n}_d{d}']
  for method in ['full','random_block_selection']:
   a=q[method];k=q['K'] if method=='full' else q['K_sub'];text+=f"| {d if x=='d' else n} | {labels[method]} | {k} | {a['mean_runtime']:.6f} | {a['mean_error']:.6f} | {a['mean_support_recovery']:.6f} | {q['matched_completed']}/10 |\n"
 return text
common='''\n**Comparison with random block selection.** The original full Algorithm 1 fits above are reused. Random block selection was run on exactly the same data and seeds 0–9; contaminated-data SHA256 hashes and true means were checked for every fit. It first builds all original $K$ blocks, then draws $K_{\\mathrm{sub}}$ indices independently and uniformly with replacement using `block_selection_seed=seed` in a separate RNG stream. One selection per seed; duplicates retain their multiplicity. $K_{\\mathrm{sub}}$ is the smallest odd integer at least $2[s\\log(d/s)+\\log(4/\\delta)]$. Initialization and optimization use only those selected block means. Both methods use tolerance $10^{-5}$ and the same Algorithm 1 heuristics. Runtime includes all block construction and, for the random method, selection. Both batches ran sequentially with one solver/BLAS thread and other experiment workers stopped; timings come from separate batches. Means below use the same converged seed pairs.\n\n'''
a=common+table([(100,5),(100,10),(100,20)],'d')
a+='\nFull/random mean-runtime speedups at d=5,10,20: '+', '.join(f"{summary[f'n100_d{d}']['speedup_ratio_of_mean_runtimes']:.2f}×" for d in [5,10,20])+'. Mean distinct selected block counts: '+', '.join(f"{summary[f'n100_d{d}']['mean_unique_selected_blocks']:.1f}" for d in [5,10,20])+', respectively.\n'
b='\n**Comparison with random block selection, d=10 fixed.** The sampling and timing conditions are the same as in sanity check 19. $K_{\\mathrm{sub}}=17$ for every n, while the full method uses $K=21,41,61$.\n\n'+table([(100,10),(200,10),(300,10)],'n')
b+='\nFull/random mean-runtime speedups at n=100,200,300: '+', '.join(f"{summary[f'n{n}_d10']['speedup_ratio_of_mean_runtimes']:.2f}×" for n in [100,200,300])+'.\n'
b+='\n| n | d | Runtime speedup | L2 error change (random − full) | Support recovery change (percentage points) |\n| ---: | ---: | ---: | ---: | ---: |\n'
for n,d in configs:
 q=summary[f'n{n}_d{d}'];b+=f"| {n} | {d} | {q['speedup_ratio_of_mean_runtimes']:.2f}× | {q['mean_error_difference']:+.6f} | {100*q['mean_support_recovery_difference']:+.1f} |\n"
b+=f"\nIn all five tested settings, the random method is faster but has higher mean L2 error and lower mean support recovery. At d=10, its mean runtime is {summary['n100_d10']['random_block_selection']['mean_runtime']:.6f} s for n=100 and {summary['n300_d10']['random_block_selection']['mean_runtime']:.6f} s for n=300, with K_sub fixed at 17. The speed improvement therefore comes with an observed estimation-quality cost in this pilot.\n"
b+='\nPositive L2 error changes mean worse estimation error; positive support-recovery changes mean improvement. This is a 10-seed pilot with one block selection per seed, so the estimates include both data and selection variability. Algorithm 1 stopping gaps certify their respective full-block or sampled-block objectives; those objective values are not directly comparable.\n\n![Runtime, error, and support recovery comparison](../artifacts/sanity_check/random_block_selection_scaling/comparison.png)\n\n[Paired seed results, selected block indices, runtime medians, metrics, solver diagnostics, and source hashes](../artifacts/sanity_check/random_block_selection_scaling/results.json).\n'
text=doc.read_text()
for tag in ['random-block-dimension','random-block-sample-size']:
 text=re.sub(r'<!-- '+tag+r':start -->.*?<!-- '+tag+r':end -->\n?', '', text, flags=re.S)
boundary='\n## Sanity check 20: Algorithm 1 runtime versus sample size\n';assert text.count(boundary)==1;text=text.replace(boundary,'\n<!-- random-block-dimension:start -->\n'+a+'<!-- random-block-dimension:end -->\n'+boundary)
end='<!-- algorithm1-runtime-scaling:end -->';assert text.count(end)==1;text=text.replace(end,'<!-- random-block-sample-size:start -->\n'+b+'<!-- random-block-sample-size:end -->\n'+end)
doc.write_text(text);print(json.dumps(summary,indent=2))
