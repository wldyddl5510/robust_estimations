"""Fresh, serial, checkpointed 10-seed dense sanity check."""
import argparse, contextlib, hashlib, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
REPO=OUT.parent
BG=Path('/tmp/experiment26_eps005')
sys.path.insert(0,str(OUT/'source'))
from utils import sparse_skew_t_dist_data_generation,skew_t_covariance,adversarial_sparse_contamination
from experiments import coordinate_mom_estimation,geometric_mom_estimation,sample_mean_ht_estimation
from IP_algorithm import ip_estimation
from non_sparse_algorithm import non_sparse_estimation
from brute_force import brute_force_estimation, NetSizeError
from random_block_selection import random_block_selection
from random_direction_algorithm import random_mom_estimation,random_trimmed_estimation
from projected_algorithm import projected_estimation
from haar_projected_algorithm import haar_projected_estimation
from depersin_lecue import depersin_lecue_estimation
from cherapanamjeri_flammarion_bartlett import cfb_estimation

METHODS=['sample_mean', 'coordinate_mom', 'geometric_mom', 'algorithm_1', 'random_block_selection', 'dl_clarabel', 'cfb']
SETTINGS=json.loads((OUT/'config.json').read_text())


def active_methods():
    selection=OUT/'method_selection.json'
    excluded=json.loads(selection.read_text()).get('excluded_methods',[]) if selection.exists() else []
    if not set(excluded) <= set(METHODS)|{'algorithm_2','dl_ptz'}:raise ValueError('Unknown excluded method')
    return [name for name in METHODS if name not in excluded]


def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)


def generate(seed):
    np.random.seed(seed)
    amplitude=np.random.uniform(1,3)
    clean_data,truth=sparse_skew_t_dist_data_generation(SETTINGS['n'],amplitude,SETTINGS['nu'],SETTINGS['d'],SETTINGS['s'],
                     scale=np.array(SETTINGS['shape']),skew=np.array(SETTINGS['skew']))
    data=adversarial_sparse_contamination(clean_data,SETTINGS['epsilon'],SETTINGS['s'],strength=SETTINGS['strength'])
    return data,truth,clean_data,float(amplitude)


def run_method(seed,name):
    if name not in active_methods():raise ValueError(f'Method excluded: {name}')
    data,truth,original,amplitude=generate(seed)
    prefix=OUT/f'seed_{seed}_{name}'
    statusfile=prefix.with_suffix('.status.json')
    last=[0.]
    def progress(info):
        if time.time()-last[0] >= 10:
            save(statusfile,dict(seed=seed,method=name,state='running',pid=os.getpid(),
                                updated_epoch=time.time(),progress=info))
            last[0]=time.time()
    save(statusfile,dict(seed=seed,method=name,state='running',pid=os.getpid(),started_epoch=time.time()))
    args=(data,SETTINGS['s'],SETTINGS['epsilon'],SETTINGS['lambda_upper'],SETTINGS['delta'])
    kw=dict(seed=seed,tol=SETTINGS['tol'],C=SETTINGS['C'])
    start=time.perf_counter()
    record=dict(seed=seed,method=name,truth=truth.tolist(),amplitude=amplitude,
                source_manifest_sha256=hashlib.sha256((OUT/'source_hashes.json').read_bytes()).hexdigest(),
                block_rule_revision=SETTINGS.get('block_rule_revision'),
                data_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
                clean_data_sha256=hashlib.sha256(original.tobytes()).hexdigest())
    try:
        with prefix.with_suffix('.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            if name=='sample_mean':estimate,info=sample_mean_ht_estimation(data,SETTINGS['s'])
            elif name=='coordinate_mom':estimate,info=coordinate_mom_estimation(*args,**kw)
            elif name=='geometric_mom':estimate,info=geometric_mom_estimation(*args,**kw)
            elif name=='random_mom':estimate,info=random_mom_estimation(*args,**kw,tilde_J=SETTINGS['tilde_J'])
            elif name=='random_trimmed':estimate,info=random_trimmed_estimation(data,SETTINGS['s'],SETTINGS['epsilon'],SETTINGS['delta'],seed=seed,tol=SETTINGS['tol'],tilde_J=SETTINGS['tilde_J'])
            elif name=='brute_force':estimate,info=brute_force_estimation(*args,**kw)
            elif name=='algorithm_1':estimate,info=ip_estimation(*args,**kw)
            elif name=='dense_algorithm_1':estimate,info=non_sparse_estimation(data,SETTINGS['epsilon'],SETTINGS['lambda_upper'],SETTINGS['delta'],**kw)
            elif name=='random_block_selection':estimate,info=random_block_selection(*args,**kw)
            elif name.startswith('partition'):
                estimate,info=projected_estimation(*args,**kw,r=SETTINGS['partition_r'],
                         objective='least_squares' if name.endswith('ls') else 'max')
            elif name.startswith('haar'):
                estimate,info=haar_projected_estimation(*args,**kw,r=SETTINGS['haar_r'],J=SETTINGS['haar_J'],
                         objective='least_squares' if name.endswith('ls') else 'max')
            elif name.startswith('dl_'):
                estimate,info=depersin_lecue_estimation(data,K=SETTINGS['literature_K'],seed=seed,tol=SETTINGS['tol'],
                              backend=name.split('_')[1],sdp_eta=SETTINGS['ptz_eta'],sdp_max_iter=SETTINGS['ptz_decision_max_iter'],progress=progress)
            elif name=='cfb':estimate,info=cfb_estimation(data,K=SETTINGS['literature_K'],seed=seed,tol=SETTINGS['tol'],max_iter=SETTINGS['cfb_iterations'],step_size=SETTINGS['cfb_step_size'],progress=progress)
            else:raise ValueError(name)
        record.update(status='completed' if info['converged'] else 'unconverged',
                      error=float(np.linalg.norm(estimate-truth)),
                      support_recovery=float(np.mean(np.abs(estimate)>1e-8)),
                      estimate=estimate.tolist(),runtime=info['runtime'],info=info)
    except NetSizeError as exc:
        record.update(status='skipped',runtime=time.perf_counter()-start,reason='existing_net_size_guard',net_size=exc.size,max_net_points=200_000,exception=str(exc))
    except Exception as exc:
        record.update(status='failed',runtime=time.perf_counter()-start,
                      exception=repr(exc),traceback=traceback.format_exc())
    save(prefix.with_suffix('.json'),record)
    save(statusfile,dict(seed=seed,method=name,state='finished',status=record['status'],pid=os.getpid()))
    print('FINISHED',seed,name,record['status'],round(record['runtime'],6),flush=True)


def snapshot():
    runs=[]
    selected=active_methods()
    for seed in SETTINGS['seeds']:
        for name in selected:
            path=OUT/f'seed_{seed}_{name}.json'
            if path.exists():runs.append(json.loads(path.read_text()))
    summary={}
    for name in selected:
        all_runs=[r for r in runs if r['method']==name]
        good=[r for r in all_runs if r['status']=='completed']
        item={status:sum(r['status']==status for r in all_runs)
              for status in ('completed','unconverged','failed','skipped')}
        if all_runs:item['mean_attempt_runtime']=float(np.mean([r['runtime'] for r in all_runs]))
        if good:
            for metric in ('error','runtime','support_recovery'):
                item['mean_'+metric]=float(np.mean([r[metric] for r in good]))
            item['median_runtime']=float(np.median([r['runtime'] for r in good]))
        summary[name]=item
    total=len(SETTINGS['seeds'])*len(selected)
    payload=dict(settings={**SETTINGS,'methods':selected},summary=summary,runs=runs,
                 finished_attempts=len(runs),total_attempts=total,
                 status='finished' if len(runs)==total else 'running',updated_epoch=time.time())
    save(OUT/'progress.json',payload)
    if len(runs)==total:save(OUT/'results.json',payload)
    return payload


def stop_background():
    pending=OUT/'background_resume_pending.json'
    if pending.exists():return
    pidfile=BG/'runner_pid.json'
    if not pidfile.exists():return
    root=json.loads(pidfile.read_text())['pid']
    text=subprocess.check_output(['ps','-axo','pid=,ppid=,command='],text=True)
    procs={int(f[0]):(int(f[1]),f[2]) for line in text.splitlines()
           if len(f:=line.strip().split(None,2))==3}
    if root not in procs or str(BG/'resume_ht.py') not in procs[root][1]:return
    ids={root}
    while True:
        more=ids|{pid for pid,(ppid,_) in procs.items() if ppid in ids}
        if more==ids:break
        ids=more
    save(pending,dict(root=root,processes=sorted(ids),epoch=time.time()))
    for sig in (signal.SIGSTOP,signal.SIGKILL):
        for pid in sorted(ids):
            try:os.kill(pid,sig)
            except ProcessLookupError:pass
    for file in BG.glob('seed_*.status.json'):
        value=json.loads(file.read_text())
        if value.get('state')=='running':
            value['state']='interrupted_for_non_sparse_sanity'
            save(file,value)
    mainprogress=REPO/'experiment8_progress.json'
    if mainprogress.exists():
        value=json.loads(mainprogress.read_text())
        save(OUT/'experiment8_progress_before_pause.json',value)
        value['status']='paused_for_non_sparse_sanity_timing'
        save(mainprogress,value)
    print('Experiment 8 checkpoints preserved; stopped active fits for isolated timing.',flush=True)


def resume_background():
    pending=OUT/'background_resume_pending.json'
    if not pending.exists():return
    with (BG/'resume_after_non_sparse_sanity.log').open('a') as log:
        proc=subprocess.Popen([sys.executable,'-u',str(BG/'resume_ht.py')],
                              stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(BG/'runner_pid.json',dict(pid=proc.pid,started_epoch=time.time(),script='resume_ht.py',
                                 experiment_number=8,reason='Resume after dense sanity comparison'))
    save(OUT/'background_resumed.json',dict(pid=proc.pid,epoch=time.time()))
    pending.unlink()
    print('Resumed Experiment 8:',proc.pid,flush=True)



def main():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--method')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.worker:return run_method(args.seed,args.method)
    assert set(active_methods())==set(SETTINGS['methods']) and 'algorithm_2' not in active_methods()
    save(OUT/'settings.json',{**SETTINGS,'methods':active_methods()})
    for seed in SETTINGS['seeds']:
        data,truth,original,amplitude=generate(seed)
        np.savez(OUT/f'data_seed_{seed}.npz',data=data,truth=truth,clean_data=original,amplitude=amplitude)
    if args.prepare_only:
        snapshot()
        print('Prepared',len(SETTINGS['seeds']),'shared datasets; no fits started.',flush=True)
        return
    stop_background()
    save(OUT/'active.json',dict(state='running',workers=SETTINGS['workers'],started_epoch=time.time()))
    jobs=[(seed,name) for name in active_methods() for seed in SETTINGS['seeds']
          if not (OUT/f'seed_{seed}_{name}.json').exists()]
    def one(seed,name):
        print('START',seed,name,flush=True)
        code=subprocess.run([sys.executable,'-u',__file__,'--worker','--seed',str(seed),'--method',name]).returncode
        if code:raise RuntimeError(f'Worker exited {code}: {seed} {name}')
        return seed,name
    completed=False
    next_publish=100
    try:
        with ThreadPoolExecutor(max_workers=SETTINGS['workers']) as pool:
            futures=[pool.submit(one,seed,name) for seed,name in jobs]
            for future in as_completed(futures):
                future.result()
                payload=snapshot()
                print('PROGRESS',payload['finished_attempts'],'/',payload['total_attempts'],flush=True)
                if payload['finished_attempts']>=next_publish:
                    subprocess.run([sys.executable,str(OUT/'publish.py'),'--skip-plot'],check=True)
                    next_publish=100*(payload['finished_attempts']//100+1)
        completed=True
    finally:
        try:
            subprocess.run([sys.executable,str(OUT/'publish.py')],check=True)
            payload=json.loads((OUT/'progress.json').read_text())
            save(OUT/'active.json',dict(state=payload['status'],finished_attempts=payload['finished_attempts'],
                                      total_attempts=payload['total_attempts'],updated_epoch=time.time()))
        finally:
            if completed:resume_background()

if __name__=='__main__':main()
