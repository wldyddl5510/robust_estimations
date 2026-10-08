import json,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/Users/jiyoungpark/postdoc/robust_estimation_IP/robust_estimations');OUT=R/'sanity_check/random_block_selection_without_replacement';DOC=R/'sanity_check/sanity_check_results.md'
raw=json.loads((OUT/'results.json').read_text());assert raw['status']=='finished' and len(raw['runs'])==50
sources={'full':json.loads((OUT/'full_algorithm1_reference.json').read_text()),'wr2':json.loads((OUT/'with_replacement_reference.json').read_text()),'wr3':json.loads((OUT/'multiplier3_reference.json').read_text()),'wor2':raw}
lookups={m:{(r['n'],r['d'],r['seed']):r for r in v['runs']} for m,v in sources.items()};configs=[(100,5),(100,10),(100,20),(200,10),(300,10)];paired=[];summary={}
for key in sorted(lookups['wor2']):
 methods={m:v[key] for m,v in lookups.items()};assert len({r['data_sha256'] for r in methods.values()})==1;assert all(r['truth']==methods['full']['truth'] for r in methods.values());paired.append(dict(n=key[0],d=key[1],seed=key[2],methods=methods))
for n,d in configs:
 good=[p for p in paired if (p['n'],p['d'])==(n,d) and all(v['status']=='completed' for v in p['methods'].values())];v=dict(n=n,d=d,K=good[0]['methods']['full']['K'],matched_completed=len(good),methods={})
 for m in sources:
  rows=[p['methods'][m] for p in good];v['methods'][m]={f'mean_{metric}':float(np.mean([r[metric] for r in rows])) for metric in ['runtime','error','support_recovery']};v['methods'][m].update(median_runtime=float(np.median([r['runtime'] for r in rows])),blocks_used=rows[0]['info'].get('K_sub',rows[0]['info']['K']),mean_unique_blocks=float(np.mean([r['info'].get('unique_selected_blocks',r['info']['K']) for r in rows])))
 v['comparisons']={}
 for m in ['full','wr2','wr3']:
  a=v['methods'][m];b=v['methods']['wor2'];v['comparisons'][m]=dict(reference_over_wor_runtime=a['mean_runtime']/b['mean_runtime'],wor_error_difference=b['mean_error']-a['mean_error'],wor_support_difference=b['mean_support_recovery']-a['mean_support_recovery'])
 summary[f'n{n}_d{d}']=v
raw['comparison_summary']=summary;raw['paired_runs']=paired;raw['comparison_note']='Data hashes and truth match across all four methods. Full and with-replacement timings reuse earlier isolated sequential single-thread batches. Without-replacement coefficient 2 is a new batch on seeds 0–9, using a separate SeedSequence(seed, spawn_key=(2,)); take the first min(K, requested count) indices of a uniform permutation and restore original block order. No replacement, no retry/best-of-repeat selection. Means use converged seed quadruplets. Runtime includes original block construction and selection.'
(OUT/'results.json').write_text(json.dumps(raw,indent=2)+'\n')
labels={'full':'Full Algorithm 1','wr2':'With replacement (c=2)','wr3':'With replacement (c=3)','wor2':'Without replacement (c=2)'};short={'full':'Full Algorithm 1','wr2':'WR, c=2','wr3':'WR, c=3','wor2':'WOR, c=2'};colors={'full':'#4477aa','wr2':'#cc6677','wr3':'#228833','wor2':'#8855aa'}
fig,axes=plt.subplots(2,3,figsize=(13.4,7.8))
for row,group,title,xlabel in [(0,[(100,5),(100,10),(100,20)],'Fixed n=100','Dimension d'),(1,[(100,10),(200,10),(300,10)],'Fixed d=10','Sample size n')]:
 for col,(metric,ylabel) in enumerate([('runtime','Runtime (s, log scale)'),('error','L2 error'),('support_recovery','Support recovery')]):
  ax=axes[row,col]
  for m,offset in zip(sources,[-.15,-.05,.05,.15]):
   means=[]
   for j,(n,d) in enumerate(group):
    vals=[p['methods'][m][metric] for p in paired if (p['n'],p['d'])==(n,d) and all(v['status']=='completed' for v in p['methods'].values())];means.append(np.mean(vals));ax.scatter(j+offset+np.linspace(-.02,.02,len(vals)),vals,color=colors[m],s=9,alpha=.18,zorder=2)
   ax.plot(np.arange(3)+offset,means,'o-',color=colors[m],label=short[m],linewidth=2 if m=='wor2' else 1.5,markersize=4,zorder=3)
  ax.set_xticks(range(3),[str(d if row==0 else n) for n,d in group]);ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.set_title(title);ax.grid(axis='y',alpha=.2)
  if metric=='runtime':ax.set_yscale('log')
  if metric=='support_recovery':ax.set_ylim(-.05,1.05)
for col in [0,1]:
 lo=min(ax.get_ylim()[0] for ax in axes[:,col]);hi=max(ax.get_ylim()[1] for ax in axes[:,col])
 for ax in axes[:,col]:ax.set_ylim(0 if col==1 else lo,hi)
handles,names=axes[0,0].get_legend_handles_labels();fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.964),ncol=4,frameon=False)
fig.suptitle('Random blocks: sampling without replacement | s=2, delta=0.05, epsilon=0.1',y=.995);fig.text(.5,.008,'WR: with replacement. WOR: without replacement. Lines: 10-seed means; faint points: individual seeds.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.028,1,.925));fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
def table(group,x):
 text=f'| {x} | Method | Blocks used / draws | Mean runtime (s) | Mean L2 error | Mean support recovery | Matched seeds |\n| ---: | --- | ---: | ---: | ---: | ---: | ---: |\n'
 for n,d in group:
  q=summary[f'n{n}_d{d}']
  for m in sources:
   a=q['methods'][m];text+=f"| {d if x=='d' else n} | {labels[m]} | {a['blocks_used']} | {a['mean_runtime']:.6f} | {a['mean_error']:.6f} | {a['mean_support_recovery']:.6f} | {q['matched_completed']}/10 |\n"
 return text
common='''
**Comparison of full Algorithm 1 and random-block variants.** Every method uses identical data and true means for each seed (verified SHA256 and vector equality). Full Algorithm 1 and with-replacement coefficients 2/3 reuse the earlier results. The new method resets the coefficient to 2 and samples **without replacement**, using
$K_{\\mathrm{sub}}=\\min(K,\\operatorname{oddceil}(2[s\\log(d/s)+\\log(4/\\delta)]))$.
Original K-block construction is unchanged. K and the requested count are odd, so the capped count is odd. The subset is uniform; every selected block is used once. A separate RNG stream from `block_selection_seed=seed` generates a uniform permutation; its prefix selects the subset and original block order is restored for optimization. If all K blocks are selected, optimization matches full Algorithm 1. Historical with-replacement methods retain duplicate draws and do not cap K_sub. Each coefficient 3 draw list extends the corresponding coefficient 2 list, but there is no prefix relationship across sampling modes.

All variants use tolerance $10^{-5}$ and the existing Algorithm 1 heuristics. One fit per setting and seed 0–9, with no best-of-repeat selection. Statistics below use converged seed quadruplets. Timings include original block construction, selection, and optimization; they come from separate batches, each sequential with one solver/BLAS thread and other experiment workers stopped.

'''
a=common+table([(100,5),(100,10),(100,20)],'d')
a+='\nAt d=5,10,20, without-replacement coefficient 2 selects exactly 13,17,19 distinct blocks from K=21. The cap is not active in these five benchmark setups; the K_sub > K edge case is covered by solver tests.\n'
b='\n**Comparison at fixed d=10.** The full method uses K=21,41,61. Coefficient 2 uses K_sub=17 for both sampling modes; without replacement these are always 17 distinct blocks. With-replacement coefficient 3 uses 23 draws.\n\n'+table([(100,10),(200,10),(300,10)],'n')
b+='\nEffect of replacing coefficient-2 sampling with replacement (WR2) by sampling without replacement (WOR2):\n\n| n | d | WR2 / WOR2 mean runtime | L2 error change (WOR2 − WR2) | Support change (percentage points) | Full / WOR2 mean runtime |\n| ---: | ---: | ---: | ---: | ---: | ---: |\n'
for n,d in configs:
 q=summary[f'n{n}_d{d}'];v=q['comparisons']['wr2'];b+=f"| {n} | {d} | {v['reference_over_wor_runtime']:.2f}× | {v['wor_error_difference']:+.6f} | {100*v['wor_support_difference']:+.1f} | {q['comparisons']['full']['reference_over_wor_runtime']:.2f}× |\n"
err=sum(q['comparisons']['wr2']['wor_error_difference']<0 for q in summary.values());sup=sum(q['comparisons']['wr2']['wor_support_difference']>1e-12 for q in summary.values());worse=sum(q['comparisons']['wr2']['wor_support_difference']<-1e-12 for q in summary.values());fast=sum(q['comparisons']['full']['reference_over_wor_runtime']>1 for q in summary.values())
b+=f'\nAgainst WR2, WOR2 has lower mean L2 error in {err}/5 settings. Mean support recovery improves in {sup}, worsens in {worse}, and is unchanged in {5-sup-worse}. It has lower mean runtime than full Algorithm 1 in {fast}/5 settings. This remains a ten-seed pilot with one block selection per seed; timing ratios compare separate batches and do not establish an asymptotic complexity law. Objective gaps concern each selected-block collection.\n'
b+='\n![Full and random-block methods: with and without replacement](random_block_selection_without_replacement/comparison.png)\n\n[Four-way paired results, selected indices, requested/actual block counts, diagnostics, and source hashes](random_block_selection_without_replacement/results.json). Historical comparisons: [WR coefficient 2](random_block_selection_scaling/results.json), [WR coefficient 3](random_block_selection_multiplier3/results.json).\n'
text=DOC.read_text()
for tag,content in [('random-block-dimension',a),('random-block-sample-size',b)]:
 pattern=r'<!-- '+tag+r':start -->.*?<!-- '+tag+r':end -->';text,count=re.subn(pattern,lambda m:f'<!-- {tag}:start -->\n{content}<!-- {tag}:end -->',text,flags=re.S);assert count==1,(tag,count)
DOC.write_text(text);print(json.dumps(summary,indent=2))
