import sys,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/Users/jiyoungpark/postdoc/robust_estimation_IP/robust_estimations');OUT=R/'sanity_check/algorithm1_runtime_scaling';doc=R/'sanity_check/sanity_check_results.md'
raw=json.loads((OUT/'results.json').read_text());assert raw['status']=='finished' and len(raw['runs'])==50
configs=[(100,5),(100,10),(100,20),(200,10),(300,10)];summary={};by={}
for n,d in configs:
 records=[r for r in raw['runs'] if (r['n'],r['d'])==(n,d)];assert sorted(r['seed'] for r in records)==list(range(10))
 good=[r for r in records if r['status']=='completed'];times=np.array([r['runtime'] for r in good]);by[n,d]={r['seed']:r for r in good}
 summary[f'n{n}_d{d}']=dict(n=n,d=d,K=records[0]['K'],mean_outer_iterations=float(np.mean([r['info']['iterations'] for r in good])) if good else None,mean_oracle_calls=float(np.mean([r['info']['oracle_calls'] for r in good])) if good else None,mean_inner_iterations=float(np.mean([r['info']['inner_iterations'] for r in good])) if good else None,completed=len(good),failed=sum(r['status']=='failed' for r in records),unconverged=sum(r['status']=='unconverged' for r in records),mean_runtime=float(times.mean()) if times.size else None,median_runtime=float(np.median(times)) if times.size else None,min_runtime=float(times.min()) if times.size else None,max_runtime=float(times.max()) if times.size else None,std_runtime=float(times.std(ddof=1)) if times.size>1 else None,mean_error=float(np.mean([r['error'] for r in good])) if good else None,mean_support_recovery=float(np.mean([r['support_recovery'] for r in good])) if good else None)
def comparison(a,b):
 seeds=sorted(set(by[a])&set(by[b]));ratios=np.array([by[b][s]['runtime']/by[a][s]['runtime'] for s in seeds]);logs=np.log(ratios);rng=np.random.default_rng(20261006);bootstrap=np.exp(np.mean(rng.choice(logs,size=(10000,len(logs)),replace=True),axis=1))
 return dict(from_setting=a,to_setting=b,matched_seeds=seeds,ratio_of_mean_runtimes=summary[f'n{b[0]}_d{b[1]}']['mean_runtime']/summary[f'n{a[0]}_d{a[1]}']['mean_runtime'],median_paired_ratio=float(np.median(ratios)),geometric_mean_paired_ratio=float(np.exp(logs.mean())),bootstrap_95_percent_interval=np.quantile(bootstrap,[.025,.975]).tolist(),per_seed_ratios=ratios.tolist())
comparisons={label:comparison(a,b) for label,a,b in [('d_5_to_20',(100,5),(100,20)),('n_100_to_300',(100,10),(300,10)),('d_10_to_20',(100,10),(100,20)),('n_100_to_200',(100,10),(200,10))]}
shared=sorted(set(by[100,10])&set(by[100,20])&set(by[200,10]));relative=np.array([by[100,20][s]['runtime']/by[200,10][s]['runtime'] for s in shared]);logs=np.log(relative);rng=np.random.default_rng(20261006);boot=np.exp(rng.choice(logs,size=(10000,len(logs)),replace=True).mean(axis=1));comparisons['dimension_vs_n_doubling']=dict(matched_seeds=shared,definition='Ratio of the runtime multiplier for d:10->20 to that for n:100->200, sharing the (100,10) baseline.',geometric_mean_ratio=float(np.exp(logs.mean())),bootstrap_95_percent_interval=np.quantile(boot,[.025,.975]).tolist())
raw['runtime_summary']=summary;raw['runtime_comparisons']=comparisons;raw['analysis_note']='Single fit per setting and seed. Paired-seed descriptive/bootstrap comparisons reflect runtime and data difficulty; not a complexity-law estimate. Increasing n also increases K via the default block rule. Other experiment workers were paused; unrelated system workload was not controlled.'
(OUT/'results.json').write_text(json.dumps(raw,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(10,4.4),sharey=True)
for ax,group,title,xlabel in [(axes[0],[(100,5),(100,10),(100,20)],'Fixed n=100','Dimension d'),(axes[1],[(100,10),(200,10),(300,10)],'Fixed d=10','Sample size n')]:
 values=[[r['runtime'] for r in by[c].values()] for c in group];labels=[str(c[1] if xlabel=='Dimension d' else c[0]) for c in group]
 ax.boxplot(values,positions=np.arange(3),widths=.45,showfliers=False,patch_artist=True,boxprops=dict(facecolor='#dce8f3'),medianprops=dict(color='#222222'))
 for j,(c,v) in enumerate(zip(group,values)):
  for seed,r in by[c].items():ax.scatter(j+(seed-4.5)*.022,r['runtime'],s=18,color='#4477aa',alpha=.7)
 ax.plot(np.arange(3),[np.mean(v) for v in values],'D--',color='#cc6677',markersize=5,label='Mean')
 ax.set_xticks(range(3),labels);ax.set_xlabel(xlabel);ax.set_ylabel('Algorithm 1 runtime (s, log scale)');ax.set_yscale('log');ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.legend()
fig.suptitle('Algorithm 1: s=2, delta=0.05, epsilon=0.1; 10 seeds per setting');fig.tight_layout();fig.savefig(OUT/'runtime_comparison.png',dpi=180);plt.close(fig)
common='''Both comparisons fix $s=2$, $\\delta=0.05$, $\\epsilon=0.1$. Additional settings: multivariate $t_3$ data with shape $I_d$, sparse signal amplitude Uniform$(1,3)$ per seed, $\\overline\\lambda=6$, attack strength 30, $C=2$, and fixed tolerance $10^{-5}$. Algorithm 1 uses the current hybrid indicator oracle and additional heuristics, with its default outer/inner iteration limits of 1000. Seeds 0–9; one fit per setting and seed, run sequentially with one solver thread and BLAS thread. Other experiment workers were stopped during timing; unfinished fits restarted from saved checkpoints afterwards. Runtime includes estimator initialization and optimization, excluding data generation and metric calculation. The shared $(n,d)=(100,10)$ fits are reused in both comparisons (50 unique fits total). Data share the distribution and seed convention; samples across differing dimensions or sample sizes are not nested.\n\n'''
def table(group):
 text='| n | d | K | Mean runtime (s) | Median runtime (s) | Min–max runtime (s) | Completed seeds |\n| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n'
 for n,d in group:
  v=summary[f'n{n}_d{d}'];text+=f"| {n} | {d} | {v['K']} | {v['mean_runtime']:.6f} | {v['median_runtime']:.6f} | {v['min_runtime']:.6f}–{v['max_runtime']:.6f} | {v['completed']}/10 |\n"
 return text
section='\n<!-- algorithm1-runtime-scaling:start -->\n## Sanity check 19: Algorithm 1 runtime versus dimension\n\n'+common+table([(100,5),(100,10),(100,20)])
x=comparisons['d_5_to_20'];section+=f"\nDimension 5 → 20 (4×) changes mean runtime by {x['ratio_of_mean_runtimes']:.3f}×. The block count stays at $K=21$.\n"
section+='\n## Sanity check 20: Algorithm 1 runtime versus sample size\n\n'+table([(100,10),(200,10),(300,10)])
x=comparisons['n_100_to_300'];section+=f"\nSample size 100 → 300 (3×) changes mean runtime by {x['ratio_of_mean_runtimes']:.3f}×. The default block rule gives $K=21,41,61$: $K$ is the smallest odd integer at least $2\\max\\{{s\\log(d/s),\\epsilon n,\\log(1/\\delta)\\}}$. Here $\\epsilon n$ dominates, so the n sweep also increases the integer oracle's block count rather than only data preprocessing.\n"
a=comparisons['d_10_to_20'];b=comparisons['n_100_to_200'];relative=comparisons['dimension_vs_n_doubling'];lo,hi=relative['bootstrap_95_percent_interval'];effect='dimension' if relative['geometric_mean_ratio']>1 else 'sample size (and its induced block count)'
section+=f"\nFor equal 2× input changes, mean runtime ratios are {a['ratio_of_mean_runtimes']:.3f}× for d:10 → 20 and {b['ratio_of_mean_runtimes']:.3f}× for n:100 → 200. Across the same seeds, the geometric mean ratio of these runtime multipliers is {relative['geometric_mean_ratio']:.3f} (paired-seed bootstrap 95% interval {lo:.3f}–{hi:.3f}); values above 1 favor a larger dimension effect. The observed average multiplier is larger for {effect} in this setup. "
section+=('The interval spans 1, so these ten seeds do not clearly distinguish the two effects. ' if lo<=1<=hi else 'The interval lies entirely '+('above' if lo>1 else 'below')+' 1 in this pilot comparison. ')
section+='This is a runtime comparison over these settings and seeds, not evidence of an asymptotic complexity law; seed difficulty and solve iteration counts can strongly affect the timings.\n\n![Algorithm 1 runtime versus dimension and sample size](algorithm1_runtime_scaling/runtime_comparison.png)\n\n[Detailed settings, seed-level metrics, solver diagnostics, and comparisons](algorithm1_runtime_scaling/results.json).\n<!-- algorithm1-runtime-scaling:end -->\n'
text=doc.read_text();start=text.find('<!-- algorithm1-runtime-scaling:start -->');end=text.find('<!-- algorithm1-runtime-scaling:end -->')
if start>=0:
 assert end>start;text=text[:start].rstrip()+section+text[end+len('<!-- algorithm1-runtime-scaling:end -->'):]
else:text=text.rstrip()+section
doc.write_text(text)
print(json.dumps(dict(summary=summary,comparisons=comparisons),indent=2))
