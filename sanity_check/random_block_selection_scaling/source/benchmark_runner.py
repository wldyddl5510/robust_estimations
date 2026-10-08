import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import sys,json,time,hashlib,subprocess,signal,platform,traceback,contextlib,shutil
from pathlib import Path
import numpy as np
REPO=Path('/Users/jiyoungpark/postdoc/robust_estimation_IP/robust_estimations');OUT=REPO/'sanity_check/random_block_selection_scaling';OUT.mkdir(exist_ok=True)
SOURCE=OUT/'source';SOURCE.mkdir(exist_ok=True)
for f in ['IP_algorithm.py','utils.py','random_block_selection.py']:
 if not (SOURCE/f).exists():shutil.copy2(REPO/f,SOURCE/f)
sys.path.insert(0,str(SOURCE))
from utils import sparse_t_dist_data_generation,adversarial_sparse_contamination
from random_block_selection import random_block_selection
REFERENCE=OUT/'full_algorithm1_reference.json'
if not REFERENCE.exists():shutil.copy2(REPO/'sanity_check/algorithm1_runtime_scaling/results.json',REFERENCE)
reference=json.loads(REFERENCE.read_text());lookup={(r['n'],r['d'],r['seed']):r for r in reference['runs']}
SETTINGS=dict(reference['settings'],method='random_block_selection',sampling_with_replacement=True,block_selection_seed='equal to data seed, separate SeedSequence spawn_key=(2,)',K_sub='smallest odd integer >= 2*(s*log(d/s)+log(4/delta))',reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest())
CONFIGS=[(100,5),(100,10),(100,20),(200,10),(300,10)]
BACKGROUND=Path('/tmp/experiment26_eps005');background_stopped=False

def save(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def processes():
 output=subprocess.check_output(['ps','-axo','pid=,ppid=,command='],text=True)
 return {int(f[0]):(int(f[1]),f[2]) for line in output.splitlines() if len(f:=line.strip().split(None,2))==3}
def stop_background():
 global background_stopped
 p=BACKGROUND/'runner_pid.json'
 if not p.exists():return
 root=json.loads(p.read_text())['pid'];procs=processes()
 if root not in procs or str(BACKGROUND/'resume_ht.py') not in procs[root][1]:return
 ids={root}
 while True:
  new=ids|{pid for pid,(parent,cmd) in procs.items() if parent in ids}
  if new==ids:break
  ids=new
 for pid in sorted(ids):
  try:os.kill(pid,signal.SIGSTOP)
  except ProcessLookupError:pass
 for pid in sorted(ids):
  try:os.kill(pid,signal.SIGKILL)
  except ProcessLookupError:pass
 background_stopped=True
 save(OUT/'background_interruption.json',dict(root_pid=root,processes=sorted(ids),epoch=time.time(),reason='Isolated runtime measurement; completed checkpoints preserved.'))
 for p in BACKGROUND.glob('seed_*.status.json'):
  x=json.loads(p.read_text())
  if x['state']=='running':x['state']='interrupted_for_random_block_benchmark';save(p,x)
 print('Stopped background experiment from checkpoints',sorted(ids),flush=True)
def resume_background():
 if not background_stopped:return
 log=(BACKGROUND/'resume_after_random_block_scaling.log').open('a')
 runner=subprocess.Popen([sys.executable,'-u',str(BACKGROUND/'resume_ht.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(BACKGROUND/'runner_pid.json',dict(pid=runner.pid,started_epoch=time.time(),script='resume_ht.py',experiment_number=8,reason='Resume unfinished fits after isolated random block selection benchmark.'))
 print('Restarted background experiment',runner.pid,flush=True)
def stop(sig,frame):raise KeyboardInterrupt(f'signal {sig}')
for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
def snapshot(active=None,finished=False):
 runs=[json.loads(p.read_text()) for p in sorted(OUT.glob('n*_d*_seed*.json'))]
 payload=dict(settings=SETTINGS,status='finished' if finished else 'running',runner_pid=os.getpid(),last_updated_epoch=time.time(),active=active,runs=runs,completed_fits=len(runs),total_fits=50,source_sha256={f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in ['IP_algorithm.py','utils.py','random_block_selection.py']},platform=platform.platform())
 save(OUT/'progress.json',payload)
 if finished:save(OUT/'results.json',payload)
def main():
 stop_background()
 try:
  for seed in range(10):
   order=CONFIGS if seed==0 else [CONFIGS[i] for i in np.random.default_rng(10000+seed).permutation(len(CONFIGS))]
   for n,d in order:
    path=OUT/f'n{n}_d{d}_seed{seed}.json'
    if path.exists():continue
    np.random.seed(seed);loc=np.random.uniform(1,3);data,truth=sparse_t_dist_data_generation(n,loc,3,d,2,scale=1);data=adversarial_sparse_contamination(data,.1,2,strength=30)
    ref=lookup[n,d,seed];digest=hashlib.sha256(data.tobytes()).hexdigest();assert digest==ref['data_sha256'];assert truth.tolist()==ref['truth'] and loc==ref['loc']
    active=dict(n=n,d=d,seed=seed,started_epoch=time.time());snapshot(active);print('START',active,flush=True);start=time.perf_counter()
    try:
     with (OUT/f'n{n}_d{d}_seed{seed}.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
      estimate,info=random_block_selection(data,2,.1,6,.05,tol=1e-5,seed=seed,C=2,block_selection_seed=seed)
     result=dict(status='completed' if info['converged'] else 'unconverged',runtime=info['runtime'],error=float(np.linalg.norm(estimate-truth)),support_recovery=float(np.sum((np.abs(estimate)>1e-8)&(truth!=0))/2),estimate=estimate.tolist(),info=info)
    except Exception as exc:result=dict(status='failed',elapsed=time.perf_counter()-start,exception=repr(exc),traceback=traceback.format_exc())
    save(path,dict(n=n,d=d,seed=seed,loc=loc,truth=truth.tolist(),data_sha256=digest,**result));snapshot();print('DONE',n,d,seed,result['status'],result.get('runtime',result.get('elapsed')),flush=True)
  snapshot(finished=True)
 finally:resume_background()
if __name__=='__main__':main()
