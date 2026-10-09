"""100-seed dimension comparison using the frozen Experiment 9 estimators."""
import argparse, contextlib, hashlib, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
REPO=OUT.parent
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

METHODS=['sample_mean', 'coordinate_mom', 'geometric_mom', 'algorithm_1', 'random_block_selection', 'dl_clarabel']
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
    if seed in cancelled_seeds(name):raise ValueError(f'Cancelled by user: {seed} {name}')
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


def cancelled_seeds(name):
    path=OUT/'cfb_cancellation.json'
    if name!='cfb' or not path.exists():return set()
    return set(json.loads(path.read_text())['cancelled_seeds'])


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
        item['cancelled']=len(cancelled_seeds(name))
        if all_runs:item['mean_attempt_runtime']=float(np.mean([r['runtime'] for r in all_runs]))
        if good:
            for metric in ('error','runtime','support_recovery'):
                item['mean_'+metric]=float(np.mean([r[metric] for r in good]))
            item['median_runtime']=float(np.median([r['runtime'] for r in good]))
        summary[name]=item
    total=len(SETTINGS['seeds'])*len(selected)
    cancelled=sum(len(cancelled_seeds(name)) for name in selected)
    resolved=len(runs)+cancelled
    payload=dict(settings={**SETTINGS,'methods':selected},summary=summary,runs=runs,
                 finished_attempts=len(runs),total_attempts=total,
                 cancelled_attempts=cancelled,resolved_attempts=resolved,
                 status=('finished_with_cfb_cancelled' if cancelled else 'finished') if resolved==total else 'running',updated_epoch=time.time())
    if not runs and not (OUT/'active.json').exists():payload['status']='queued'
    save(OUT/'progress.json',payload)
    if resolved==total:save(OUT/'results.json',payload)
    return payload


def process_matches(pid, seed, name):
    """Check the exact adopted worker identity, including zombie/exited state."""
    import shlex
    value=subprocess.run(['ps','-p',str(pid),'-o','stat=','-o','command='],capture_output=True,text=True)
    if value.returncode or not value.stdout.strip():return False
    state,command=value.stdout.strip().split(None,1)
    if state.startswith('Z'):return False
    args=shlex.split(command)
    return (str(OUT/'run.py') in args and '--worker' in args and
            '--seed' in args and args[args.index('--seed')+1]==str(seed) and
            '--method' in args and args[args.index('--method')+1]==name)


def main():
    import fcntl
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--method')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.worker:return run_method(args.seed,args.method)
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert active_methods()==SETTINGS['methods']
        assert not {'algorithm_2','cfb','dl_ptz','random_block_selection'} & set(active_methods())
        order=SETTINGS['execution_order']
        assert set(order)==set(active_methods()) and order.index('dl_clarabel')<order.index('algorithm_1')
        save(OUT/'settings.json',SETTINGS)
        for seed in SETTINGS['seeds']:
            data,truth,original,amplitude=generate(seed)
            with np.load(OUT/f'data_seed_{seed}.npz') as saved:
                for key,value in [('data',data),('truth',truth),('clean_data',original)]:
                    np.testing.assert_array_equal(saved[key],value)
        if args.prepare_only:
            snapshot()
            subprocess.run([sys.executable,str(OUT/'publish.py')],check=True)
            return
        active={}
        handoff=json.loads((OUT/'queue_handoff.json').read_text())
        for item in handoff['adopted_workers']:
            pid,seed,name=item['pid'],item['seed'],item['method']
            if process_matches(pid,seed,name):
                assert name=='algorithm_1'
                active[pid]={**item,'process':None,'adopted':True}
            else:
                assert (OUT/f'seed_{seed}_{name}.json').exists(),f'Adopted worker lost: {seed} {name}'
        assert len(active)<=SETTINGS['workers']
        started=time.time()
        next_publish=50*(snapshot()['finished_attempts']//50+1)
        last_publish=0.
        last_heartbeat=0.
        subprocess.run([sys.executable,str(OUT/'publish.py')],check=True)
        print('Adopted in-flight workers without pausing fits:',sorted(active),flush=True)
        try:
            while True:
                changed=False
                for pid,item in list(active.items()):
                    proc=item['process']
                    done=(proc.poll() is not None) if proc is not None else not process_matches(pid,item['seed'],item['method'])
                    if not done:continue
                    path=OUT/f"seed_{item['seed']}_{item['method']}.json"
                    assert path.exists(),f'Worker exited without result: {pid} {item}'
                    if proc is not None:assert proc.returncode==0,f'Worker exited {proc.returncode}'
                    print('RESOLVED',item['seed'],item['method'],'adopted' if item['adopted'] else 'new',flush=True)
                    del active[pid]
                    changed=True
                phase=None
                for name in order:
                    if any(not (OUT/f'seed_{seed}_{name}.json').exists() for seed in SETTINGS['seeds']):
                        phase=name
                        break
                running={(item['seed'],item['method']) for item in active.values()}
                if phase:
                    candidates=[seed for seed in SETTINGS['seeds'] if (seed,phase) not in running and not (OUT/f'seed_{seed}_{phase}.json').exists()]
                    for seed in candidates[:SETTINGS['workers']-len(active)]:
                        proc=subprocess.Popen([sys.executable,'-u',str(OUT/'run.py'),'--worker','--seed',str(seed),'--method',phase])
                        active[proc.pid]=dict(pid=proc.pid,seed=seed,method=phase,process=proc,adopted=False)
                        print('START',seed,phase,flush=True)
                        changed=True
                if changed or time.time()-last_heartbeat>=30:
                    payload=snapshot()
                    state=dict(state=payload['status'],pid=os.getpid(),workers=SETTINGS['workers'],started_epoch=started,
                               priority_method=phase,execution_order=order,finished_attempts=payload['finished_attempts'],
                               total_attempts=payload['total_attempts'],updated_epoch=time.time(),
                               active_workers=[{k:v for k,v in item.items() if k not in ('process','command')} for item in active.values()])
                    save(OUT/'active.json',state)
                    last_heartbeat=time.time()
                    print('PROGRESS',payload['finished_attempts'],'/',payload['total_attempts'],'priority',phase,'active',[(v['seed'],v['method']) for v in active.values()],flush=True)
                    if payload['finished_attempts']>=next_publish or time.time()-last_publish>=300:
                        subprocess.run([sys.executable,str(OUT/'publish.py')],check=True)
                        next_publish=50*(payload['finished_attempts']//50+1)
                        last_publish=time.time()
                if phase is None and not active:break
                time.sleep(2)
        finally:
            subprocess.run([sys.executable,str(OUT/'publish.py')],check=True)
            payload=json.loads((OUT/'progress.json').read_text())
            save(OUT/'active.json',dict(state=payload['status'],pid=os.getpid(),finished_attempts=payload['finished_attempts'],
                                      total_attempts=payload['total_attempts'],execution_order=order,updated_epoch=time.time()))
if __name__=='__main__':main()
