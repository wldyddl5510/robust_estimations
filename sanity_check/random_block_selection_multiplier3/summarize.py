import json,hashlib,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/Users/jiyoungpark/postdoc/robust_estimation_IP/robust_estimations');OUT=R/'sanity_check/random_block_selection_multiplier3';doc=R/'sanity_check/sanity_check_results.md'
raw=json.loads((OUT/'results.json').read_text());assert raw['status']=='finished' and len(raw['runs'])==50
full=json.loads((OUT/'full_algorithm1_reference.json').read_text());c2=json.loads((OUT/'multiplier2_reference.json').read_text());configs=[(100,5),(100,10),(100,20),(200,10),(300,10)]
lookups={name:{(r['n'],r['d'],r['seed']):r for r in data['runs']} for name,data in [('full',full),('c2',c2),('c3',raw)]};summary={};paired=[]
for key in sorted(lookups['c3']):
 records={m:v[key] for m,v in lookups.items()};assert len({r['data_sha256'] for r in records.values()})==1;assert records['full']['truth']==records['c2']['truth']==records['c3']['truth'];assert records['c3']['info']['selected_block_indices'][:records['c2']['info']['K_sub']]==records['c2']['info']['selected_block_indices'];paired.append(dict(n=key[0],d=key[1],seed=key[2],methods=records))
for n,d in configs:
 good=[p for p in paired if (p['n'],p['d'])==(n,d) and all(v['status']=='completed' for v in p['methods'].values())];v=dict(n=n,d=d,K=good[0]['methods']['full']['K'],matched_completed=len(good),methods={})
 for method in ['full','c2','c3']:
  records=[p['methods'][method] for p in good];v['methods'][method]={f'mean_{metric}':float(np.mean([r[metric] for r in records])) for metric in ['runtime','error','support_recovery']};v['methods'][method]['median_runtime']=float(np.median([r['runtime'] for r in records]));v['methods'][method]['blocks_used']=records[0]['info'].get('K_sub',records[0]['info']['K']);v['methods'][method]['mean_unique_blocks']=float(np.mean([r['info'].get('unique_selected_blocks',r['info']['K']) for r in records]))
 a=v['methods']['full'];b=v['methods']['c2'];c=v['methods']['c3'];v['c3_vs_c2']=dict(runtime_ratio=c['mean_runtime']/b['mean_runtime'],error_difference=c['mean_error']-b['mean_error'],support_recovery_difference=c['mean_support_recovery']-b['mean_support_recovery']);v['full_vs_c3_speedup']=a['mean_runtime']/c['mean_runtime'];summary[f'n{n}_d{d}']=v
raw['comparison_summary']=summary;raw['paired_runs']=paired;raw['comparison_note']='All methods use identical data/truth. Coefficient 3 draws extend the coefficient 2 draw prefix with the same selection seed. Full and coefficient 2 timings are reused from their earlier isolated sequential single-thread batches; coefficient 3 is a new batch. One fit per seed, no retries/best-of-repeat selection. Original block construction and all preprocessing included in runtime. Means use converged pairs shared by all three methods.'
(OUT/'results.json').write_text(json.dumps(raw,indent=2)+'\n')
labels={'full':'Algorithm 1 (all blocks)','c2':'Random blocks, c=2','c3':'Random blocks, c=3'};colors={'full':'#4477aa','c2':'#cc6677','c3':'#228833'};fig,axes=plt.subplots(2,3,figsize=(13.4,7.8))
for row,group,title,xlabel in [(0,[(100,5),(100,10),(100,20)],'Fixed n=100','Dimension d'),(1,[(100,10),(200,10),(300,10)],'Fixed d=10','Sample size n')]:
 for col,(metric,ylabel) in enumerate([('runtime','Runtime (s, log scale)'),('error','L2 error'),('support_recovery','Support recovery')]):
  ax=axes[row,col]
  for method,offset in [('full',-.10),('c2',0),('c3',.10)]:
   means=[]
   for j,(n,d) in enumerate(group):
    vals=[p['methods'][method][metric] for p in paired if (p['n'],p['d'])==(n,d) and all(v['status']=='completed' for v in p['methods'].values())];means.append(np.mean(vals));ax.scatter(j+offset+np.linspace(-.025,.025,len(vals)),vals,color=colors[method],s=11,alpha=.25,zorder=2)
   ax.plot(np.arange(3)+offset,means,'o-',color=colors[method],label=labels[method],linewidth=1.7,markersize=4,zorder=3)
  ax.set_xticks(range(3),[str(d if row==0 else n) for n,d in group]);ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.set_title(title);ax.grid(axis='y',alpha=.2)
  if metric=='runtime':ax.set_yscale('log')
  if metric=='support_recovery':ax.set_ylim(-.05,1.05)
for col in [0,1]:
 lo=min(ax.get_ylim()[0] for ax in axes[:,col]);hi=max(ax.get_ylim()[1] for ax in axes[:,col])
 for ax in axes[:,col]:ax.set_ylim(0 if col==1 else lo,hi)
handles,names=axes[0,0].get_legend_handles_labels();fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.965),ncol=3,frameon=False)
fig.suptitle('Random block selection: coefficient 2 vs 3 | s=2, delta=0.05, epsilon=0.1',y=.995);fig.text(.5,.006,'Lines: 10-seed means. Faint points: individual seeds. All runtimes include original block construction.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.025,1,.925));fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)

def table(group,x):
 text=f'| {x} | Method | Blocks used / draws | Mean runtime (s) | Mean L2 error | Mean support recovery | Matched seeds |\n| ---: | --- | ---: | ---: | ---: | ---: | ---: |\n'
 for n,d in group:
  q=summary[f'n{n}_d{d}']
  for method in ['full','c2','c3']:
   v=q['methods'][method];text+=f"| {d if x=='d' else n} | {labels[method]} | {v['blocks_used']} | {v['mean_runtime']:.6f} | {v['mean_error']:.6f} | {v['mean_support_recovery']:.6f} | {q['matched_completed']}/10 |\n"
 return text
common='''
**Comparison with random block selection, coefficients 2 and 3.** All methods use the same contaminated data and truth for each seed, verified by SHA256 and vector equality. The full Algorithm 1 and coefficient 2 results are retained from earlier batches; coefficient 3 is a new 50-fit batch. Original $K$-block construction is unchanged. For coefficient $c$, $K_{\\mathrm{sub}}$ is the smallest odd integer at least $c[s\\log(d/s)+\\log(4/\\delta)]$. Sampling is independent uniform **with replacement**, retaining duplicates, and is performed once before optimization. `block_selection_seed=seed` uses a separate RNG stream. Each coefficient 3 draw sequence starts with all coefficient 2 draws, then appends more. All tolerances remain $10^{-5}$. Means below use the same converged seed triplets (0–9); one fit per method/setting/seed, with no best-of-repeat selection. Runtime includes construction of all original blocks, sampling, and optimization. Batches were measured separately, sequentially with one solver/BLAS thread and other experiment workers stopped.

'''
a=common+table([(100,5),(100,10),(100,20)],'d')
a+='\nAt d=5,10,20, coefficient 2 uses K_sub=13,17,19 and coefficient 3 uses K_sub=19,23,27, respectively. With replacement, K_sub can exceed the original K=21; it does not mean all original blocks are used. Average distinct selected blocks (c=2 → c=3): '+', '.join(f"d={d}: {summary[f'n100_d{d}']['methods']['c2']['mean_unique_blocks']:.1f} → {summary[f'n100_d{d}']['methods']['c3']['mean_unique_blocks']:.1f}" for d in [5,10,20])+'.\n'
b='\n**Comparison with random block selection, d=10 fixed.** The conditions are the same as in sanity check 19. The full method uses K=21,41,61. The random methods use K_sub=17 for c=2 and K_sub=23 for c=3, regardless of n.\n\n'+table([(100,10),(200,10),(300,10)],'n')
b+='\nChanges from coefficient 2 to 3:\n\n| n | d | Runtime ratio (c=3 / c=2) | L2 error change (c=3 − c=2) | Support change (percentage points) | Full / c=3 runtime ratio |\n| ---: | ---: | ---: | ---: | ---: | ---: |\n'
for n,d in configs:
 q=summary[f'n{n}_d{d}'];v=q['c3_vs_c2'];b+=f"| {n} | {d} | {v['runtime_ratio']:.2f}× | {v['error_difference']:+.6f} | {100*v['support_recovery_difference']:+.1f} | {q['full_vs_c3_speedup']:.2f}× |\n"
error_better=sum(q['c3_vs_c2']['error_difference']<0 for q in summary.values());support_better=sum(q['c3_vs_c2']['support_recovery_difference']>1e-12 for q in summary.values());support_worse=sum(q['c3_vs_c2']['support_recovery_difference']<-1e-12 for q in summary.values());time_more=sum(q['c3_vs_c2']['runtime_ratio']>1 for q in summary.values())
b+=f'\nCoefficient 3 has lower mean L2 error in {error_better}/5 settings than coefficient 2; mean support recovery improves in {support_better}, worsens in {support_worse}, and is unchanged in {5-support_better-support_worse}. Mean runtime increases in {time_more}/5 settings. This is a 10-seed pilot: increasing the number of draws does not guarantee monotone accuracy or runtime for each realized subset. Runtime ratios are ratios of means from separate batches, not complexity-law estimates. Optimization gaps certify each method’s own full- or sampled-block objective.\n'
b+='\n![Full Algorithm 1 and random block coefficients 2 and 3](random_block_selection_multiplier3/comparison.png)\n\n[Three-way paired results, selected indices, metrics, source hashes, and comparisons](random_block_selection_multiplier3/results.json). [Original coefficient 2 comparison](random_block_selection_scaling/results.json).\n'
text=doc.read_text()
for tag,content in [('random-block-dimension',a),('random-block-sample-size',b)]:
 pattern=r'<!-- '+tag+r':start -->.*?<!-- '+tag+r':end -->';text,count=re.subn(pattern,lambda m:f'<!-- {tag}:start -->\n{content}<!-- {tag}:end -->',text,flags=re.S);assert count==1,(tag,count)
doc.write_text(text);print(json.dumps(summary,indent=2))
