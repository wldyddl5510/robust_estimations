import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import sys,json,time,hashlib,subprocess,signal,platform,traceback,contextlib
from pathlib import Path
import numpy as np
ROOT=Path('/tmp/algorithm1_runtime_scaling');REPO=Path('/Users/jiyoungpark/postdoc/robust_estimation_IP/robust_estimations');OUT=REPO/'sanity_check/algorithm1_runtime_scaling';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(REPO))
from utils import sparse_t_dist_data_generation,adversarial_sparse_contamination,mom_initialization
from IP_algorithm import ip_estimation
SETTINGS=dict(s=2,delta=.05,epsilon=.1,nu=3,scale=1,strength=30,C=2,tol=1e-5,seeds=list(range(10)),solver_threads=1,workers=1,data_distribution='multivariate t, shape I_d',oracle_strategy='hybrid',oracle_formulation='indicator',max_iter=1000,max_inner_iter=1000)
CONFIGS=[(100,5),(100,10),(100,20),(200,10),(300,10)]
def save(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def rows():
 output=subprocess.check_output(['ps','-axo','pid=,ppid=,command='],text=True)
 return {int(f[0]):(int(f[1]),f[2]) for line in output.splitlines() if len(f:=line.strip().split(None,2))==3}
paused={}
def pause_background():
 p=Path('/tmp/experiment26_eps005/runner_pid.json')
 if not p.exists():return
 root=json.loads(p.read_text())['pid'];procs=rows()
 if root not in procs or '/tmp/experiment26_eps005/resume_ht.py' not in procs[root][1]:return
 ids={root}
 while True:
  new=ids|{i for i,(parent,cmd) in procs.items() if parent in ids}
  if new==ids:break
  ids=new
 for i in sorted(ids):
  try:os.kill(i,signal.SIGSTOP);paused[i]=procs[i][1]
  except ProcessLookupError:pass
 save(ROOT/'paused_background.json',paused);print('Paused background experiment',sorted(paused),flush=True)
def resume_background():
 procs=rows()
 for pid,cmd in paused.items():
  if pid in procs and procs[pid][1]==cmd:
   try:os.kill(pid,signal.SIGCONT)
   except ProcessLookupError:pass
 print('Resumed background experiment',flush=True)
def stop(sig,frame):raise KeyboardInterrupt(f'signal {sig}')
for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
def publish_progress(active=None,finished=False):
 runs=[json.loads(p.read_text()) for p in sorted(OUT.glob('n*_d*_seed*.json'))]
 payload=dict(settings=SETTINGS,status='finished' if finished else 'running',runner_pid=os.getpid(),last_updated_epoch=time.time(),active=active,runs=runs,completed_fits=len(runs),total_fits=50,source_sha256={f:hashlib.sha256((REPO/f).read_bytes()).hexdigest() for f in ['IP_algorithm.py','utils.py']},platform=platform.platform())
 save(OUT/'progress.json',payload)
 if finished:save(OUT/'results.json',payload)
def main():
 pause_background()
 try:
  for seed in range(10):
   order=CONFIGS if seed==0 else [CONFIGS[i] for i in np.random.default_rng(10000+seed).permutation(len(CONFIGS))]
   for n,d in order:
    path=OUT/f'n{n}_d{d}_seed{seed}.json'
    if path.exists():continue
    np.random.seed(seed);loc=np.random.uniform(1,3);data,truth=sparse_t_dist_data_generation(n,loc,3,d,2,scale=1);data=adversarial_sparse_contamination(data,.1,2,strength=30)
    block_means,*_=mom_initialization(data,2,.1,6,.05,1e-5,seed,C=2)
    active=dict(n=n,d=d,seed=seed,K=len(block_means),started_epoch=time.time());publish_progress(active)
    print('START',active,flush=True);start=time.perf_counter()
    try:
     with (OUT/f'n{n}_d{d}_seed{seed}.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
      estimate,info=ip_estimation(data,2,.1,6,.05,tol=1e-5,seed=seed,C=2)
     result=dict(status='completed' if info['converged'] else 'unconverged',runtime=info['runtime'],error=float(np.linalg.norm(estimate-truth)),support_recovery=float(np.sum((np.abs(estimate)>1e-8)&(truth!=0))/2),estimate=estimate.tolist(),info=info)
    except Exception as exc:result=dict(status='failed',elapsed=time.perf_counter()-start,exception=repr(exc),traceback=traceback.format_exc())
    save(path,dict(n=n,d=d,seed=seed,K=len(block_means),loc=loc,truth=truth.tolist(),data_sha256=hashlib.sha256(data.tobytes()).hexdigest(),**result));publish_progress()
    print('DONE',n,d,seed,result['status'],result.get('runtime',result.get('elapsed')),flush=True)
  publish_progress(finished=True)
 finally:resume_background()
if __name__=='__main__':main()
