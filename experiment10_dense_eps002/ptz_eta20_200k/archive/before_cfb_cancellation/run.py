"""Run the Experiment 10 DL/PTZ retry after in-flight CFB seeds finish."""
import argparse,contextlib,hashlib,json,os,shlex,signal,subprocess,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
PARENT=OUT.parent
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
    record=dict(seed=seed,method='dl_ptz',variant='eta20_200k',truth=truth.tolist(),
      data_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
      clean_data_sha256=hashlib.sha256(arrays['clean_data'].tobytes()).hexdigest(),
      source_manifest_sha256=hashlib.sha256((OUT/'source_hashes.json').read_bytes()).hexdigest())
    last=[0.]
    def progress(info):
        if time.time()-last[0]>=10:
            save(p.with_suffix('.status.json'),dict(seed=seed,state='running',pid=os.getpid(),progress=info,updated_epoch=time.time()))
            last[0]=time.time()
    save(p.with_suffix('.status.json'),dict(seed=seed,state='running',pid=os.getpid(),started_epoch=time.time()))
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
    pending=OUT/'previous_runner_pending.json'
    if not pending.exists():return
    old=json.loads(pending.read_text())
    procs=processes()
    found=[pid for pid,(_,_,argv) in procs.items() if old['script'] in argv and '--worker' not in argv]
    if old['pid'] in found:
        pid=old['pid']
        try:
            subprocess.run([sys.executable,str(PARENT/'publish.py')],check=True)
            for name,key in [('active.json','state'),('runner_pid.json','status')]:
                val=json.loads((PARENT/name).read_text());val[key]='running';val.pop('pause_reason',None);save(PARENT/name,val)
        finally:os.kill(pid,signal.SIGCONT)
    elif found:raise RuntimeError(f'Unexpected parent runner: {found}')
    else:
        with (PARENT/'run.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',old['script']],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        pid=proc.pid
        save(PARENT/'runner_pid.json',dict(pid=pid,status='running',workers=4,experiment_number=10,started_epoch=time.time()))
    save(OUT/'previous_runner_resumed.json',dict(pid=pid,epoch=time.time()))
    pending.unlink()
    print('RESUMED Experiment 10:',pid,flush=True)
def drain():
    old=json.loads((OUT/'previous_runner_pending.json').read_text())
    while True:
        procs=processes()
        root=procs.get(old['pid'])
        assert root and old['script'] in root[2] and 'T' in root[1],root
        waiting=[]
        for w in old['children']:
            proc=procs.get(w['pid'])
            if proc is not None and 'Z' not in proc[1]:waiting.append(w)
            elif not (PARENT/f"seed_{w['seed']}_{w['method']}.json").exists():
                raise RuntimeError(f'Previous fit exited without checkpoint: {w}')
        save(OUT/'launch_status.json',dict(state='waiting_for_cfb' if waiting else 'ready',waiting=waiting,updated_epoch=time.time()))
        if not waiting:break
        time.sleep(10)
    subprocess.run([sys.executable,str(PARENT/'publish.py')],check=True)
    for name,key in [('progress.json','status'),('active.json','state'),('runner_pid.json','status')]:
        v=json.loads((PARENT/name).read_text());v[key]='paused_for_ptz_retry';v['pause_reason']='Resume after eta/20, 200000-update DL/PTZ comparison';save(PARENT/name,v)
    # No parent publisher is running while its scheduler is stopped.
    report=PARENT.parent/'results.md';text=report.read_text()
    a=text.index('<!-- experiment10-dense:start -->');b=text.index('<!-- experiment10-dense:end -->',a)
    section=text[a:b].replace('Batch status: **running**','Batch status: **paused_for_ptz_retry**')
    report.write_text(text[:a]+section+text[b:])
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);args=parser.parse_args()
    if args.worker is not None:return worker(args.worker)
    try:
        drain()
        save(OUT/'launch_status.json',dict(state='running',pid=os.getpid(),workers=4,started_epoch=time.time()))
        seeds=[s for s in CFG['seeds'] if not (OUT/f'seed_{s}_dl_ptz.json').exists()]
        def one(seed):
            subprocess.run([sys.executable,'-u',__file__,'--worker',str(seed)],check=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(one,seed) for seed in seeds]
            for i,future in enumerate(as_completed(futures),1):
                future.result()
                if i%10==0:
                    subprocess.run([sys.executable,str(OUT/'publish.py')],check=True)
        save(OUT/'launch_status.json',dict(state='finished',finished_epoch=time.time()))
        subprocess.run([sys.executable,str(OUT/'publish.py')],check=True)
        print('PTZ RETRY FINISHED',flush=True)
    except Exception:
        save(OUT/'launch_status.json',dict(state='runner_failed',exception=traceback.format_exc(),updated_epoch=time.time()))
        subprocess.run([sys.executable,str(OUT/'publish.py')])
        raise
    finally:resume_parent()
if __name__=='__main__':main()

