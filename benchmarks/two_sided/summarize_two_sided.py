"""Summarize paired benchmarks and draw a static comparison figure."""
import argparse
import json
from pathlib import Path
import math
import statistics as st
import numpy as np

def summarize(records,keys):
 groups={}
 for row in records:
  key=tuple(row[k] for k in keys)
  groups.setdefault(key,[]).append(row)
 output=[]
 for key,rows in groups.items():
  seed_rows=[]
  for seed in sorted(set(r['seed'] for r in rows)):
   both={side:[r for r in rows if r['seed']==seed and r['two_sided']==side] for side in (False,True)}
   timed_out=any(r.get('status')==9 or ('error' in r) or not r.get('converged',r.get('certified',False)) for r in both[False]+both[True])
   item={'seed':seed,'complete':not timed_out}
   for side,label in ((False,'one'),(True,'two')):
    rr=both[side]
    if not rr:continue
    item[label+'_seconds']=st.median(r['seconds'] for r in rr)
    if 'nodes' in rr[0]:item[label+'_nodes']=st.median(r['nodes'] for r in rr)
    if 'work' in rr[0]:item[label+'_work']=st.median(r['work'] for r in rr)
    if 'info' in rr[0]:item[label+'_calls']=rr[0]['info']['oracle_calls']
   item['speedup']=item['one_seconds']/item['two_seconds']
   seed_rows.append(item)
  good=[r for r in seed_rows if r['complete']]
  item=dict(zip(keys,key));item.update(seeds=seed_rows,complete_seeds=len(good),censored_seeds=len(seed_rows)-len(good))
  if good:
   one=st.mean(r['one_seconds'] for r in good);two=st.mean(r['two_seconds'] for r in good)
   item.update(one_seconds=one,two_seconds=two,speedup=one/two,
    paired_geomean=math.exp(st.mean(math.log(r['speedup']) for r in good)),
    min_seed_speedup=min(r['speedup'] for r in good),max_seed_speedup=max(r['speedup'] for r in good),
    seeds_faster=sum(r['speedup']>1 for r in good),seeds_faster_10pct=sum(r['speedup']>1.1 for r in good),
    seeds_slower_10pct=sum(r['speedup']<1/1.1 for r in good))
   for metric in ('nodes','work','calls'):
    if 'one_'+metric in good[0]:
     item['one_'+metric]=st.mean(r['one_'+metric] for r in good)
     item['two_'+metric]=st.mean(r['two_'+metric] for r in good)
  output.append(item)
 return output

def main():
 p=argparse.ArgumentParser();p.add_argument('--oracle',type=Path,required=True);p.add_argument('--e2e',type=Path);p.add_argument('--outdir',type=Path,required=True)
 a=p.parse_args();o=json.loads(a.oracle.read_text());e=json.loads(a.e2e.read_text()) if a.e2e else {'e2e_records':[]}
 oracle=summarize(o['oracle_records'],('case','formulation','perspective','mode'))
 e2e=summarize(e['e2e_records'],('case','formulation'))
 result=dict(oracle=oracle,e2e=e2e,oracle_count=len(o['oracle_records']),e2e_count=len(e['e2e_records']))
 (a.outdir/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 for title,rows in [('EXACT INDICATOR',[r for r in oracle if r['formulation']=='indicator' and r['mode']=='exact']),('E2E',e2e)]:
  print(title)
  for r in rows:
   print(r['case'],f"{r.get('one_seconds',0)*1000:.2f} -> {r.get('two_seconds',0)*1000:.2f} ms, ratio {r.get('speedup',0):.2f}, wins {r.get('seeds_faster',0)}/{r['complete_seeds']}, censored {r['censored_seeds']}")
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 labels={
 'dense_gaussian':'Dense Gaussian (d2, s2, K5)', 'sparse_gaussian':'Sparse Gaussian (d6, s1, K7)',
 'sparse_t5':'Student t5 (d8, s2, K9)', 'wide_sparse':'Wide sparse (d12, s1, K9)',
 'less_sparse':'Less sparse (d8, s3, K9)','more_blocks':'More blocks (d6, s2, K15)',
 'dense_many_blocks':'Dense (d3, s3, K25)','correlated':'Correlation 0.9 (d8, s2, K9)',
 'heavy_tail':'Student t2.5 (d8, s2, K9)','outlier_20pct':'20% block outliers (d8, s2, K15)',
 'outlier_40pct':'40% block outliers (d6, s2, K15)','displaced_center':'Displaced center (d8, s2, K9)',
 'skewed':'Skewed noise (d6, s2, K9)','larger_sparse':'Larger sparse (d12, s2, K21)',
 'large_K':'Large K (d8, s2, K31)','wide_d20':'Wide (d20, s2, K15)',
 'dense_d6_large_K':'Dense (d6, s6, K31)','ties_near_zero':'Ties / zero optimum (d6, s2, K9)'}
 fig,axes=plt.subplots(1,2,figsize=(15,10),gridspec_kw={'width_ratios':[1.3,1]})
 names=[c['name'] for c in o['oracle_cases'] if any(r['case']==c['name'] for r in oracle)]
 ax=axes[0]
 for form,color,offset in (('indicator','#2563eb',-.13),('bigm','#b45309',.13)):
  for i,name in enumerate(names):
   r=next((r for r in oracle if r['case']==name and r['formulation']==form and not r['perspective'] and r['mode']=='exact'),None)
   if not r or not r['complete_seeds']:continue
   # Small dots are seed medians; large markers are ratio of mean times.
   ax.scatter([x['speedup'] for x in r['seeds'] if x['complete']],[i+offset]*r['complete_seeds'],s=14,alpha=.3,color=color)
   ax.scatter(r['speedup'],i+offset,color=color,s=40,label=form if i==0 else None)
 ax.axvline(1,color='#444444',lw=1);ax.set_xscale('log');ax.set_yticks(range(len(names)),[labels.get(n,n) for n in names]);ax.invert_yaxis()
 ax.set_title('Standalone exact oracle',weight='bold',pad=14);ax.legend(loc='lower left',frameon=False)
 ax.set_xlabel('Speedup from the second implication  (one-sided / two-sided)')
 ax.set_xticks([.4,.6,.8,1,1.2,1.5], ['0.4','0.6','0.8','1.0','1.2','1.5']);ax.minorticks_off()
 ax.grid(axis='x',alpha=.18)
 ax=axes[1]
 if e2e:
  for i,r in enumerate(e2e):
   good=[x for x in r['seeds'] if x['complete']]
   ax.scatter([x['speedup'] for x in good],[i]*len(good),s=22,alpha=.4,color='#2563eb')
   if good:ax.scatter(r['speedup'],i,color='#2563eb',s=65)
  ax.set_yticks(range(len(e2e)),[r['case'].replace('_',' ') for r in e2e]);ax.invert_yaxis()
 ax.axvline(1,color='#444444',lw=1);ax.set_xscale('log');ax.set_title('End-to-end hybrid algorithm (indicator)',weight='bold',pad=14)
 ax.set_xlabel('Speedup from the second implication  (one-sided / two-sided)');ax.grid(axis='x',alpha=.18)
 ax.set_xticks([.6,.8,1,1.2,1.4], ['0.6','0.8','1.0','1.2','1.4']);ax.minorticks_off()
 fig.suptitle('Does adding b = 0 imply projection <= G improve runtime?',fontsize=17,weight='bold',y=.99)
 fig.text(.5,.035,'Left of 1: adding the constraint is slower. Right of 1: faster.\nLarge dots: ratio of mean per-seed medians. Small dots: individual seeds. Five seeds, three serial repeats, one solver thread.',ha='center',fontsize=10,color='#555555')
 fig.tight_layout(rect=(0,.075,1,.965),w_pad=3)
 fig.savefig(a.outdir/'comparison.png',dpi=180,facecolor='white');plt.close(fig)

if __name__=='__main__':main()
