"""Three-seed DL/PTZ timing run, without an SDP iteration or time limit."""
import argparse,contextlib,hashlib,json,os,shlex,signal,subprocess,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
PARENT=OUT.parent
REPO = next(parent for parent in Path(__file__).resolve().parents if (parent / 'IP_algorithm.py').is_file())
ARTIFACTS = REPO / 'artifacts'
sys.path.insert(0,str(OUT/'source'))
from depersin_lecue import depersin_lecue_estimation
CFG=json.loads((OUT/'config.json').read_text())
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def save(p,x):
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def processes():
    return {int(f[0]):(int(f[1]),f[2],shlex.split(f[3]))
            for row in subprocess.check_output(['ps','-axo','pid=,ppid=,state=,command='],text=True).splitlines()
            if len(f:=row.strip().split(None,3))==4}
def worker(seed):
    p=OUT/f'seed_{seed}_dl_ptz'
    arrays=np.load(PARENT/f'data_seed_{seed}.npz')
    data,truth=arrays['data'],arrays['truth']
    record=dict(seed=seed,method='dl_ptz',variant='eta20_uncapped_3seeds',truth=truth.tolist(),
      data_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
      clean_data_sha256=hashlib.sha256(arrays['clean_data'].tobytes()).hexdigest(),
      source_manifest_sha256=hashlib.sha256((OUT/'source_hashes.json').read_bytes()).hexdigest())
    last=[0.]
    call_count=[0]
    started_epoch=time.time()
    def progress(info):
        if info.get('iteration')==1:call_count[0]+=1
        if time.time()-last[0]>=10:
            save(p.with_suffix('.status.json'),dict(seed=seed,state='running',pid=os.getpid(),progress=info,updated_epoch=time.time(),started_epoch=started_epoch,elapsed_seconds=time.perf_counter()-started,decision_call_index=call_count[0]))
            last[0]=time.time()
    save(p.with_suffix('.status.json'),dict(seed=seed,state='running',pid=os.getpid(),started_epoch=started_epoch))
    started=time.perf_counter()
    try:
        with p.with_suffix('.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            estimate,info=depersin_lecue_estimation(data,K=CFG['literature_K'],seed=seed,
                tol=CFG['tol'],backend='ptz',sdp_eta=CFG['ptz_eta'],sdp_max_iter=CFG['ptz_decision_max_iter'],progress=progress)
        record.update(status='completed' if info['converged'] else 'unconverged',
            estimate=estimate.tolist(),error=float(np.linalg.norm(estimate-truth)),
            support_recovery=float(np.mean(np.abs(estimate)>1e-8)),runtime=info['runtime'],info=info)
    except Exception as exc:
        record.update(status='failed',runtime=time.perf_counter()-started,exception=repr(exc),traceback=traceback.format_exc())
    save(p.with_suffix('.json'),record)
    save(p.with_suffix('.status.json'),dict(seed=seed,state='finished',status=record['status'],pid=os.getpid()))
    print('FINISHED',seed,record['status'],record['runtime'],flush=True)
def resume_parent():
    # Experiment 9/10 CFB schedulers were cancelled; resume only Experiment 8.
    pending=OUT/'background_resume_pending.json'
    if not pending.exists():return
    bg=Path('/tmp/experiment26_eps005');script=str(bg/'resume_ht.py')
    found=[pid for pid,(_,_,argv) in processes().items() if script in argv]
    if found:
        pid=found[0]
    else:
        with (bg/'resume_after_non_sparse_sanity.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',script],stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        pid=proc.pid
        save(bg/'runner_pid.json',dict(pid=pid,started_epoch=time.time(),script='resume_ht.py',
            experiment_number=8,reason='Resume after PTZ retry; remaining CFB cancelled'))
    save(OUT/'background_resumed.json',dict(pid=pid,epoch=time.time(),experiment_number=8))
    pending.unlink()
    print('RESUMED Experiment 8:',pid,flush=True)
def drain():
    # The launch script preserves completed Experiment 8 checkpoints and stops its active fits.
    return


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);args=parser.parse_args()
    if (OUT/'cancellation.json').exists():
        print('This diagnostic was cancelled by the user; no jobs will start.',flush=True)
        return
    if args.worker is not None:return worker(args.worker)
    try:
        drain()
        save(OUT/'launch_status.json',dict(state='running',pid=os.getpid(),workers=3,started_epoch=time.time()))
        save(OUT/'runner_pid.json',dict(pid=os.getpid(),status='running',workers=3,started_epoch=time.time()))
        seeds=[s for s in CFG['seeds'] if not (OUT/f'seed_{s}_dl_ptz.json').exists()]
        def one(seed):
            subprocess.run([sys.executable,'-u',__file__,'--worker',str(seed)],check=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures={pool.submit(one,seed) for seed in seeds}
            while futures:
                done,futures=wait(futures,timeout=30,return_when=FIRST_COMPLETED)
                for future in done:future.result()
                subprocess.run([sys.executable,str(OUT/'publish.py')],check=True)
        save(OUT/'launch_status.json',dict(state='finished',finished_epoch=time.time()))
        subprocess.run([sys.executable,str(OUT/'publish.py')],check=True)
        print('UNCAPPED PTZ RUN FINISHED',flush=True)
    except Exception:
        save(OUT/'launch_status.json',dict(state='runner_failed',exception=traceback.format_exc(),updated_epoch=time.time()))
        subprocess.run([sys.executable,str(OUT/'publish.py')])
        raise
    finally:resume_parent()
if __name__=='__main__':main()

