"""Paired one- vs two-sided Eq. (10) benchmarks; production source is not edited.

Run with the repo's Gurobi-enabled Python:
  python benchmark_two_sided.py --repo /path/to/repo --output /tmp/comparison.json

Only the b=0 implication differs within each paired comparison. An identical
callback records work/nodes/bounds and imposes a per-oracle time limit in both
models. Instances that hit the limit remain in the results as censored runs.
"""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import gurobipy as gp
import numpy as np

ORACLE_CASES = [
 dict(name='dense_gaussian',d=2,s=2,K=5,law='normal',rho=0.,outliers=0.,shift=0.),
 dict(name='sparse_gaussian',d=6,s=1,K=7,law='normal',rho=0.,outliers=0.,shift=0.),
 dict(name='sparse_t5',d=8,s=2,K=9,law='t5',rho=0.,outliers=0.,shift=0.),
 dict(name='wide_sparse',d=12,s=1,K=9,law='normal',rho=0.,outliers=0.,shift=0.),
 dict(name='less_sparse',d=8,s=3,K=9,law='t5',rho=0.,outliers=0.,shift=0.),
 dict(name='more_blocks',d=6,s=2,K=15,law='normal',rho=0.,outliers=0.,shift=0.),
 dict(name='dense_many_blocks',d=3,s=3,K=25,law='normal',rho=0.,outliers=0.,shift=0.),
 dict(name='correlated',d=8,s=2,K=9,law='normal',rho=.9,outliers=0.,shift=0.),
 dict(name='heavy_tail',d=8,s=2,K=9,law='t2.5',rho=.5,outliers=0.,shift=0.),
 dict(name='outlier_20pct',d=8,s=2,K=15,law='t5',rho=0.,outliers=.2,shift=0.),
 dict(name='outlier_40pct',d=6,s=2,K=15,law='t5',rho=0.,outliers=.4,shift=0.),
 dict(name='displaced_center',d=8,s=2,K=9,law='t5',rho=0.,outliers=.2,shift=2.),
 dict(name='skewed',d=6,s=2,K=9,law='skew',rho=.5,outliers=.1,shift=0.),
 dict(name='larger_sparse',d=12,s=2,K=21,law='t5',rho=0.,outliers=.1,shift=0.),
 dict(name='large_K',d=8,s=2,K=31,law='normal',rho=0.,outliers=0.,shift=0.),
 dict(name='wide_d20',d=20,s=2,K=15,law='normal',rho=.3,outliers=0.,shift=0.),
 dict(name='dense_d6_large_K',d=6,s=6,K=31,law='normal',rho=0.,outliers=0.,shift=0.),
 dict(name='ties_near_zero',d=6,s=2,K=9,law='ties',rho=0.,outliers=0.,shift=0.),
]
E2E_CASES = [
 dict(name='ip_normal',method='ip',d=6,s=1,n=160,law='normal',rho=0.,epsilon=0.,delta=.05,tol=.01),
 dict(name='ip_t5',method='ip',d=8,s=2,n=160,law='t5',rho=0.,epsilon=.02,delta=.05,tol=.01),
 dict(name='ip_correlated',method='ip',d=8,s=2,n=160,law='normal',rho=.8,epsilon=.02,delta=.05,tol=.01),
 dict(name='ip_heavy_tail',method='ip',d=6,s=2,n=160,law='t2.5',rho=0.,epsilon=.03,delta=.05,tol=.01),
 dict(name='ip_tight_tol',method='ip',d=6,s=2,n=160,law='t5',rho=0.,epsilon=.02,delta=.05,tol=1e-5),
 dict(name='projection_r2',method='projected',d=8,s=2,r=2,n=160,law='t5',rho=0.,epsilon=.03,delta=.05,tol=.01),
 dict(name='projection_r4',method='projected',d=8,s=2,r=4,n=160,law='t5',rho=.8,epsilon=.03,delta=.05,tol=.01),
 dict(name='projection_max',method='projected',d=8,s=2,r=2,n=160,law='skew',rho=0.,epsilon=.03,delta=.05,tol=.01,objective='max'),
]

class OracleTimeLimit(RuntimeError):
 pass

def finite(x):
 x=float(x)
 return x if math.isfinite(x) and abs(x)<1e90 else None

def noise(rng,n,d,law,rho):
 cov=(1-rho)*np.eye(d)+rho*np.ones((d,d))
 x=rng.standard_normal((n,d))@np.linalg.cholesky(cov).T
 if law.startswith('t') and law!='ties':
  nu=float(law[1:]);x*=np.sqrt((nu-2)/rng.chisquare(nu,size=(n,1)))
 elif law=='skew':
  x=(x+.8*(rng.exponential(size=(n,1))-1))/np.sqrt(1+.8**2)
 elif law=='ties':
  x=np.round(x,0); x[:(n+1)//2]=0
 return x

def oracle_input(case,seed):
 rng=np.random.default_rng(51000+seed)
 x=noise(rng,case['K'],case['d'],case['law'],case['rho'])
 count=int(case['K']*case['outliers'])
 if count:
  ids=rng.choice(case['K'],count,replace=False)
  direction=np.zeros(case['d']);direction[:min(2*case['s'],case['d'])]=1
  direction/=np.linalg.norm(direction)
  x[ids]+=12*direction
 mu=np.zeros(case['d']);mu[:case['s']]=case['shift']
 return x,mu

def e2e_input(case,seed):
 rng=np.random.default_rng(61000+seed)
 truth=np.zeros(case['d']);truth[:case['s']]=1.5*np.where(np.arange(case['s'])%2,-1,1)
 x=noise(rng,case['n'],case['d'],case['law'],case['rho'])+truth
 count=int(case['n']*case['epsilon'])
 if count:
  ids=rng.choice(case['n'],count,replace=False)
  x[ids]=6+rng.standard_normal((count,case['d']))
 return x,truth

def make_oracle(ip,two_sided,limit):
 source=inspect.getsource(ip.separation_oracle)
 def replace(old,new):
  nonlocal source
  assert source.count(old)==1,(old,source.count(old))
  source=source.replace(old,new)
 replace('model.Params.OutputFlag = 0','model.Params.OutputFlag = 0\n        model.Params.TimeLimit = BENCH_LIMIT')
 if two_sided:
  replace('model.addConstr(residuals[i] @ v >= level - big_m * (1 - b[i]))',
   'model.addConstr(residuals[i] @ v >= level - big_m * (1 - b[i]))\n                model.addConstr(residuals[i] @ v <= level + row_bounds[i] * b[i])')
  replace('model.addConstr((b[i] == 1) >> (residuals[i] @ v >= level))',
   'model.addConstr((b[i] == 1) >> (residuals[i] @ v >= level))\n                model.addConstr((b[i] == 0) >> (residuals[i] @ v <= level))')
 replace('        stopped = False', '''        stopped = False
        BENCH_AUDIT.clear()
        def benchmark_callback(m, where):
            if where == gp.GRB.Callback.MIP:
                nodes = m.cbGet(gp.GRB.Callback.MIP_NODCNT)
                bound = m.cbGet(gp.GRB.Callback.MIP_OBJBND)
                if nodes < .5 and abs(bound) < 1e90:
                    BENCH_AUDIT['root_bound'] = float(bound)
            if decision_threshold is not None:
                stop_when_decided(m, where)
''')
 replace('model.optimize()','model.optimize(benchmark_callback)')
 replace('model.optimize(stop_when_decided)','model.optimize(benchmark_callback)')
 replace('        if model.Status != gp.GRB.OPTIMAL and not (stopped and model.Status == gp.GRB.INTERRUPTED):', '''        BENCH_AUDIT.update(status=int(model.Status), solver_seconds=float(model.Runtime),
                           nodes=float(model.NodeCount), work=float(model.Work),
                           raw_bound=finite(model.ObjBound), solutions=int(model.SolCount))
        if model.Status not in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT) and not (stopped and model.Status == gp.GRB.INTERRUPTED):''')
 replace('elif upper - lower > tol:', 'elif model.Status != gp.GRB.TIME_LIMIT and upper - lower > tol:')
 namespace=dict(vars(ip));namespace.update(BENCH_LIMIT=limit,BENCH_AUDIT={},finite=finite)
 exec(compile(source,'<instrumented separation_oracle>','exec'),namespace)
 return namespace['separation_oracle'],namespace['BENCH_AUDIT']

def dump(path,result):
 path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

def check_pair(rows,tol):
 if len(rows)!=2:return
 if all(r.get('lower') is not None for r in rows):
  overlap=max(r['lower'] for r in rows)<=min(r['upper'] for r in rows)+max(2e-6,tol*.01)
  for r in rows:r['optimum_intervals_overlap']=overlap
  if not overlap:raise AssertionError(rows)

def run_oracles(args,ip,result):
 records=result['oracle_records']
 with gp.Env(empty=True) as env:
  env.setParam('OutputFlag',0);env.setParam('Threads',1);env.start()
  # Same warm-up for both sides, outside timing.
  for side in (False,True):
   fn,_=make_oracle(ip,side,args.oracle_limit)
   fn(np.array([[1.,0.],[0.,1.],[-1.,-1.]]),np.zeros(2),1,1e-5,env=env)
  cases=[c for c in ORACLE_CASES if not args.cases or c['name'] in args.cases]
  for case in cases:
   for seed in args.seeds:
    x,mu=oracle_input(case,seed)
    for form,perspective in [('indicator',False),('bigm',False)]+([('bigm',True)] if args.perspective and case['name'] in ('sparse_t5','wide_sparse','less_sparse','outlier_20pct') else []):
     for mode in args.oracle_modes:
      # Decisions use the same deterministic, feasible warm witness in each side.
      warm=None;threshold=None
      if mode=='decision':
       witness,_,warm=ip.heuristic_separation(x,mu,case['s'],env=env)
       threshold=witness+max(.005,.05*witness)
      any_timeout=False
      for repeat in range(args.repeats):
       pair=[]
       sides=(False,True) if (repeat+seed)%2==0 else (True,False)
       for side in sides:
        fn,audit=make_oracle(ip,side,args.oracle_limit)
        rec=dict(case=case['name'],d=case['d'],s=case['s'],K=case['K'],seed=seed,
         formulation=form,perspective=perspective,mode=mode,two_sided=side,repeat=repeat,threshold=threshold)
        start=time.perf_counter()
        try:
         lower,upper,_=fn(x,mu,case['s'],args.oracle_tol,env=env,formulation=form,
          perspective=perspective,decision_threshold=threshold,warm_start=warm)
         rec.update(lower=lower,upper=upper,gap=upper-lower,
          certified=(upper-lower<=args.oracle_tol if threshold is None else lower>threshold or upper<=threshold))
        except (RuntimeError,gp.GurobiError) as error:
         if 'Could not resolve host' in str(error) or 'server authorization' in str(error):
          raise
         rec.update(error=str(error),certified=False)
        rec.update(seconds=time.perf_counter()-start,**audit)
        any_timeout |= audit.get('status')==gp.GRB.TIME_LIMIT
        records.append(rec);pair.append(rec)
       check_pair(pair,args.oracle_tol)
       # Censored instances retained; do not spend repetitions timing a timeout.
       if any_timeout:break
    current=[r for r in records if r['case']==case['name'] and r['seed']==seed]
    dump(args.output,result)
    print('oracle',case['name'],'seed',seed,'runs',len(current),'timeouts',sum(r.get('status')==9 for r in current),flush=True)

def certify_candidate(ip,x,mu,s,reported,tolerance,env,limit):
 fun,audit=make_oracle(ip,False,limit)
 lower,upper,_=fun(x,mu,s,min(1e-6,tolerance/100),env=env,formulation='indicator')
 allowance=2e-6*max(1,abs(upper))
 return dict(lower=lower,upper=upper,reported_lower=reported['lower_bound'],reported_upper=reported['objective'],
  upper_valid=lower<=reported['objective']+allowance,
  gap_valid=upper-reported['lower_bound']<=tolerance+allowance,
  gap=max(0.,upper-reported['lower_bound']),status=audit['status'])

def run_e2e(args,ip,projected,utils,result):
 original=ip.separation_oracle
 with gp.Env(empty=True) as warm_env:
  warm_env.setParam('OutputFlag',0);warm_env.setParam('Threads',1);warm_env.start()
 records=result['e2e_records']
 cases=[c for c in E2E_CASES if not args.cases or c['name'] in args.cases]
 for case in cases:
  for seed in args.seeds:
   x,truth=e2e_input(case,seed)
   for form in args.e2e_formulations:
    any_timeout=False
    for repeat in range(args.repeats):
     for side in ((False,True) if (repeat+seed)%2==0 else (True,False)):
      # Build from the unchanged function, then patch only the module binding.
      ip.separation_oracle=original
      fn,audit=make_oracle(ip,side,args.oracle_limit)
      calls=[];dense_runs=[]
      def wrapped(*a,**kw):
       outcome=fn(*a,**kw)
       calls.append(dict(audit))
       if audit.get('status')==gp.GRB.TIME_LIMIT:raise OracleTimeLimit('Per-oracle time limit reached')
       if time.perf_counter()-start>args.e2e_limit:raise OracleTimeLimit('Estimator wall time budget reached')
       return outcome
      original_dense=projected.dense_estimation
      def capture(means,center,s,tol,**kw):
       answer,info=original_dense(means,center,s,tol,**kw)
       dense_runs.append((means,answer,s,tol,info));return answer,info
      ip.separation_oracle=wrapped;projected.dense_estimation=capture
      rec=dict(case=case['name'],seed=seed,repeat=repeat,two_sided=side,formulation=form)
      start=time.perf_counter()
      try:
       kw=dict(s=case['s'],epsilon=case['epsilon'],lambda_upper=case['d'],delta=case['delta'],
        tol=case['tol'],seed=71000+seed,oracle_strategy='hybrid',oracle_formulation=form,
        max_iter=100,max_inner_iter=100)
       if case['method']=='ip':mu,info=ip.ip_estimation(x,**kw)
       else:mu,info=projected.projected_estimation(x,r=case['r'],objective=case.get('objective','least_squares'),**kw)
       rec.update(info=info,estimate=mu.tolist(),l2_error=float(np.linalg.norm(mu-truth)),converged=info['converged'])
      except (RuntimeError,gp.GurobiError) as error:
       if 'Could not resolve host' in str(error):
        raise
       rec.update(error=str(error),converged=False)
       any_timeout |= isinstance(error,OracleTimeLimit)
      finally:
       rec.update(seconds=time.perf_counter()-start,oracle_metrics=calls)
       ip.separation_oracle=original;projected.dense_estimation=original_dense
      # Expensive independent certificate is only needed once for deterministic repeats.
      if repeat==0 and rec['converged']:
       with gp.Env(empty=True) as env:
        env.setParam('OutputFlag',0);env.setParam('Threads',1);env.start()
        if case['method']=='ip':
         means,*_=utils.mom_initialization(x,case['s'],case['epsilon'],case['d'],case['delta'],case['tol'],71000+seed)
         dense_runs=[(means,mu,case['s'],case['tol'],info)]
        certs=[]
        for means,candidate,s,tol,local in dense_runs:
         certs.append(certify_candidate(ip,means,candidate,s,local,tol if case['method']=='ip' else tol/4,env,args.oracle_limit))
        rec['certificates']=certs
        if not all(c['upper_valid'] and c['gap_valid'] for c in certs):
         raise AssertionError(rec)
      records.append(rec);dump(args.output,result)
     if any_timeout:break
   print('e2e',case['name'],'seed',seed,'done',flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--stage',choices=['oracle','e2e','both'],default='both')
 p.add_argument('--cases',nargs='+');p.add_argument('--seeds',nargs='+',type=int,default=list(range(5)))
 p.add_argument('--repeats',type=int,default=3);p.add_argument('--oracle-limit',type=float,default=3.)
 p.add_argument('--e2e-limit',type=float,default=20.);p.add_argument('--oracle-tol',type=float,default=1e-5)
 p.add_argument('--oracle-modes',nargs='+',choices=['exact','decision'],default=['exact'])
 p.add_argument('--e2e-formulations',nargs='+',choices=['indicator','bigm'],default=['indicator'])
 p.add_argument('--perspective',action='store_true')
 args=p.parse_args()
 sys.path.insert(0,str(args.repo.resolve()))
 import IP_algorithm as ip
 import projected_algorithm as projected
 import utils
 source_names=('IP_algorithm.py','projected_algorithm.py','utils.py')
 result=dict(metadata=dict(repo=str(args.repo),python=sys.version,platform=platform.platform(),gurobi=gp.gurobi.version(),
  threads=1,numpy_version=np.__version__,seeds=args.seeds,repeats=args.repeats,oracle_tol=args.oracle_tol,oracle_limit=args.oracle_limit,
  e2e_limit=args.e2e_limit,oracle_modes=args.oracle_modes,e2e_formulations=args.e2e_formulations,
  source_sha256={n:hashlib.sha256((args.repo/n).read_bytes()).hexdigest() for n in source_names},
  timing='alternating order, serial, model build + solve + extraction; certificate checks excluded',
  notes='Timeouts retained and subsequent repetitions skipped for both sides of that paired case; perspective only on four designated sparse cases.'),
  oracle_cases=ORACLE_CASES,e2e_cases=E2E_CASES,oracle_records=[],e2e_records=[])
 if args.stage in ('oracle','both'):run_oracles(args,ip,result)
 if args.stage in ('e2e','both'):run_e2e(args,ip,projected,utils,result)
 dump(args.output,result)
 print('Saved',args.output,flush=True)

if __name__=='__main__':main()
