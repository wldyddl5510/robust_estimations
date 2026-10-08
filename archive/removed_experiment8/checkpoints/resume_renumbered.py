import os,sys,json,time,hashlib,traceback,concurrent.futures as cf,multiprocessing,contextlib
from pathlib import Path
import numpy as np
ROOT=Path('/tmp/experiment26_eps005');REPO=Path('/Users/jiyoungpark/postdoc/robust_estimation_IP/robust_estimations')
sys.path.insert(0,str(ROOT/'source'))
from utils import sparse_skew_t_dist_data_generation,adversarial_sparse_contamination
from experiments import coordinate_mom_estimation,geometric_mom_estimation,plain_sample_mean_estimation
from projected_algorithm import projected_estimation
from haar_projected_algorithm import haar_projected_estimation
from random_direction_algorithm import random_mom_estimation,random_trimmed_estimation
SETTINGS=json.loads((ROOT/'settings.json').read_text())
REFERENCE={r['seed']:r for r in json.loads((ROOT/'reference19.json').read_text())['runs']}
FAST=['Coordinate-wise MoM','Geometric MoM + HT','Sample mean','Random MoM (tilde J=1000)','Random trimmed mean (tilde J=1000)']
GROUPS={'fast':FAST,'partition':['Projected Algorithm 1 (LS)','Projected Algorithm 1 (max)']}
for r,J in [(10,10),(5,10),(5,20)]:
 GROUPS[f'haar_{r}_{J}']=[f'Haar-projection LS (r={r}, J={J})',f'Haar-projection max (r={r}, J={J})']
def save(path,payload):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def run_task(seed,group):
 prefix=ROOT/f'seed_{seed}_{group}'
 np.random.seed(seed);loc=np.random.uniform(1,3)
 data,truth=sparse_skew_t_dist_data_generation(1000,loc,2.5,20,5,scale=np.array(SETTINGS['shape']),skew=np.array(SETTINGS['skew']))
 clean_digest=hashlib.sha256(data.tobytes()).hexdigest()
 state=np.random.get_state();old=adversarial_sparse_contamination(data,.01,5,strength=100)
 assert hashlib.sha256(old.tobytes()).hexdigest()==REFERENCE[seed]['data_sha256'] and truth.tolist()==REFERENCE[seed]['truth']
 np.random.set_state(state);data=adversarial_sparse_contamination(data,.05,5,strength=100)
 record=dict(seed=seed,group=group,loc=float(loc),truth=truth.tolist(),clean_data_sha256=clean_digest,data_sha256=hashlib.sha256(data.tobytes()).hexdigest(),methods={})
 existing=prefix.with_suffix('.json')
 if existing.exists():
  previous=json.loads(existing.read_text())
  assert previous['data_sha256']==record['data_sha256'] and previous['truth']==record['truth']
  record['methods']=previous['methods']
 common=dict(tol=1e-5,seed=seed,C=2)
 for index,name in enumerate(GROUPS[group]):
  if name in record['methods']:continue
  save(prefix.with_suffix('.status.json'),dict(seed=seed,group=group,method=name,state='running',pid=os.getpid(),started_at=time.time()))
  start=time.perf_counter()
  try:
   with prefix.with_suffix('.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
    args=(data,5,.05,SETTINGS['lambda_upper'],.05)
    if name=='Sample mean':estimate,info=plain_sample_mean_estimation(data)
    elif name=='Coordinate-wise MoM':estimate,info=coordinate_mom_estimation(*args,**common)
    elif name=='Geometric MoM + HT':estimate,info=geometric_mom_estimation(*args,**common)
    elif name.startswith('Random MoM'):estimate,info=random_mom_estimation(*args,**common,tilde_J=1000,direction_seed=seed)
    elif name.startswith('Random trimmed'):estimate,info=random_trimmed_estimation(data,5,.05,.05,tol=1e-5,seed=seed,tilde_J=1000,direction_seed=seed)
    elif group=='partition':estimate,info=projected_estimation(*args,**common,r=10,objective='least_squares' if index==0 else 'max')
    else:
     _,r,J=group.split('_');estimate,info=haar_projected_estimation(*args,**common,r=int(r),J=int(J),objective='least_squares' if index==0 else 'max')
   result=dict(status='completed' if info['converged'] else 'unconverged',error=float(np.linalg.norm(estimate-truth)),runtime=info['runtime'],support_recovery=None if name=='Sample mean' else float(np.sum((np.abs(estimate)>1e-8)&(truth!=0))/5),estimate=estimate.tolist(),info=info)
  except Exception as exc:
   result=dict(status='failed',exception=repr(exc),traceback=traceback.format_exc(),elapsed=time.perf_counter()-start)
  record['methods'][name]=result;save(prefix.with_suffix('.json'),record)
  print('FINISHED',seed,name,result['status'],result.get('runtime',result.get('elapsed')),flush=True)
 save(prefix.with_suffix('.status.json'),dict(seed=seed,group=group,state='finished',pid=os.getpid()))
 return seed,group

def snapshot(final=False):
 runs={};counts={name:dict(completed=0,unconverged=0,failed=0) for names in GROUPS.values() for name in names}
 for path in ROOT.glob('seed_*.json'):
  if path.name.endswith('.status.json'):continue
  record=json.loads(path.read_text());seed=record['seed']
  if seed not in runs:runs[seed]={k:v for k,v in record.items() if k not in ['methods','group']};runs[seed]['methods']={}
  assert runs[seed]['data_sha256']==record['data_sha256']
  runs[seed]['methods'].update(record['methods'])
 for run in runs.values():
  for name,result in run['methods'].items():counts[name][result['status']]+=1
 summary={}
 for name,count in counts.items():
  good=[r['methods'][name] for r in runs.values() if name in r['methods'] and r['methods'][name]['status']=='completed']
  summary[name]=dict(count)
  if good:
   summary[name].update(error=float(np.mean([r['error'] for r in good])),runtime=float(np.mean([r['runtime'] for r in good])),runtime_median=float(np.median([r['runtime'] for r in good])),support_recovery=None if name=='Sample mean' else float(np.mean([r['support_recovery'] for r in good])))
 active=[json.loads(p.read_text()) for p in ROOT.glob('seed_*.status.json') if json.loads(p.read_text())['state']=='running']
 payload=dict(title='Experiment 8: Experiment 1 with epsilon=0.05',settings=SETTINGS,source_sha256=json.loads((ROOT/'source_hashes.json').read_text()),experiment_number=8,original_experiment_number=26,status='finished' if final else 'running',last_updated_epoch=time.time(),runner_pid=os.getpid(),summary=summary,runs=[runs[s] for s in sorted(runs)],active=active)
 save(ROOT/'progress.json',payload);save(REPO/'experiment8_progress.json',payload)
 if final:save(ROOT/'experiment8_results.json',payload)
 print('PROGRESS',sum(sum(c.values()) for c in counts.values()),'/1300',flush=True)

def main():
 with cf.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
  futures={}
  for group in GROUPS:
   for seed in range(100):
    path=ROOT/f'seed_{seed}_{group}.json'
    previous=json.loads(path.read_text())['methods'] if path.exists() else {}
    if all(name in previous for name in GROUPS[group]):continue
    futures[pool.submit(run_task,seed,group)]=(seed,group)
  snapshot()
  while futures:
   done,_=cf.wait(futures,timeout=30,return_when=cf.FIRST_COMPLETED)
   for future in done:
    task=futures.pop(future)
    try:future.result()
    except Exception as exc:print('TASK FAILURE',task,repr(exc),flush=True)
   snapshot()
 snapshot(final=True)
 import subprocess
 subprocess.run([sys.executable,str(ROOT/'publish.py')],check=True)
if __name__=='__main__':main()
