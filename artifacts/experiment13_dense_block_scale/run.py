"""Checkpointed C=1,...,5 comparison on the identical Experiment 11 datasets."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

OUT=Path(__file__).resolve().parent
REPO = next(parent for parent in Path(__file__).resolve().parents if (parent / 'IP_algorithm.py').is_file())
ARTIFACTS = REPO / 'artifacts'
sys.path.insert(0,str(OUT/'source'))
from utils import sparse_skew_t_dist_data_generation, adversarial_sparse_contamination
from experiments import coordinate_mom_estimation, geometric_mom_estimation
from IP_algorithm import ip_estimation
from random_block_selection import random_block_selection
from depersin_lecue import depersin_lecue_estimation

SETTINGS=json.loads((OUT/'config.json').read_text())
METHODS=SETTINGS['methods']


def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def save(path,value):
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)


def generate(seed):
    np.random.seed(seed)
    amplitude=np.random.uniform(1,3)
    original,truth=sparse_skew_t_dist_data_generation(SETTINGS['n'],amplitude,SETTINGS['nu'],SETTINGS['d'],SETTINGS['s'],
        scale=np.array(SETTINGS['shape']),skew=np.array(SETTINGS['skew']))
    data=adversarial_sparse_contamination(original,SETTINGS['epsilon'],SETTINGS['s'],strength=SETTINGS['strength'])
    return data,truth,original,float(amplitude)


def result_path(C,seed,name):
    return OUT/f'C_{C}'/f'seed_{seed}_{name}.json'


def run_method(C,seed,name):
    assert C in SETTINGS['C_values'] and seed in SETTINGS['seeds'] and name in METHODS
    path=result_path(C,seed,name)
    assert not path.exists(), 'Completed records must never be overwritten.'
    case=SETTINGS['cases'][str(C)]
    with np.load(OUT/f'data_seed_{seed}.npz') as saved:
        data=saved['data'];truth=saved['truth'];original=saved['clean_data'];amplitude=float(saved['amplitude'])
    statusfile=path.with_suffix('.status.json')
    save(statusfile,dict(C=C,seed=seed,method=name,state='running',pid=os.getpid(),started_epoch=time.time()))
    last=[0.]
    def progress(info):
        if time.time()-last[0]>=10:
            save(statusfile,dict(C=C,seed=seed,method=name,state='running',pid=os.getpid(),updated_epoch=time.time(),progress=info))
            last[0]=time.time()
    args=(data,SETTINGS['s'],SETTINGS['epsilon'],SETTINGS['lambda_upper'],SETTINGS['delta'])
    kw=dict(C=C,seed=seed,tol=SETTINGS['tol'])
    record=dict(C=C,seed=seed,method=name,truth=truth.tolist(),amplitude=amplitude,
        source_manifest_sha256=hashlib.sha256((OUT/'source_hashes.json').read_bytes()).hexdigest(),
        block_rule_revision=SETTINGS['block_rule_revision'],
        data_sha256=hashlib.sha256(data.tobytes()).hexdigest(),clean_data_sha256=hashlib.sha256(original.tobytes()).hexdigest())
    start=time.perf_counter()
    try:
        with path.with_suffix('.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            if name=='algorithm_1':estimate,info=ip_estimation(*args,**kw)
            elif name=='random_block_selection':estimate,info=random_block_selection(*args,**kw)
            elif name=='coordinate_mom':estimate,info=coordinate_mom_estimation(*args,**kw)
            elif name=='geometric_mom':estimate,info=geometric_mom_estimation(*args,**kw)
            elif name=='dl_clarabel':
                estimate,info=depersin_lecue_estimation(data,K=case['K_standard'],seed=seed,tol=SETTINGS['tol'],
                    backend='clarabel',sdp_eta=SETTINGS['ptz_eta'],sdp_max_iter=SETTINGS['ptz_decision_max_iter'],progress=progress)
            else:raise ValueError(name)
        record.update(status='completed' if info['converged'] else 'unconverged',estimate=estimate.tolist(),info=info,
            error=float(np.linalg.norm(estimate-truth)),runtime=info['runtime'],
            support_recovery=float(np.mean(np.abs(estimate)>1e-8)))
    except Exception as exc:
        record.update(status='failed',runtime=time.perf_counter()-start,exception=repr(exc),traceback=traceback.format_exc())
    save(path,record)
    save(statusfile,dict(C=C,seed=seed,method=name,state='finished',status=record['status'],pid=os.getpid()))
    print('FINISHED',C,seed,name,record['status'],round(record['runtime'],6),flush=True)


def snapshot():
    runs=[];summary={}
    for C in SETTINGS['C_values']:
        summary[str(C)]={}
        for name in METHODS:
            records=[]
            for seed in SETTINGS['seeds']:
                path=result_path(C,seed,name)
                if path.exists():records.append({**json.loads(path.read_text()),'C':C})
            runs+=records
            item={status:sum(v['status']==status for v in records) for status in ('completed','unconverged','failed')}
            good=[v for v in records if v['status']=='completed']
            item['reused_completed']=len(good) if C==2 and name in SETTINGS['reused_methods'] else 0
            if records:item['mean_attempt_runtime']=float(np.mean([v['runtime'] for v in records]))
            if good:
                for metric in ('error','runtime','support_recovery'):
                    values=np.array([v[metric] for v in good])
                    item['mean_'+metric]=float(values.mean())
                    item['std_'+metric]=float(values.std(ddof=1)) if len(values)>1 else 0.
                    item['min_'+metric]=float(values.min());item['max_'+metric]=float(values.max())
                item['median_runtime']=float(np.median([v['runtime'] for v in good]))
            summary[str(C)][name]=item
    total=len(SETTINGS['C_values'])*len(SETTINGS['seeds'])*len(METHODS)
    payload=dict(settings=SETTINGS,summary=summary,runs=runs,finished_attempts=len(runs),total_attempts=total,
        reused_attempts=len(SETTINGS['seeds'])*len(SETTINGS['reused_methods']),
        new_attempts_recorded=len(runs)-len(SETTINGS['seeds'])*len(SETTINGS['reused_methods']),
        status='finished' if len(runs)==total else 'running',updated_epoch=time.time())
    save(OUT/'progress.json',payload)
    save(OUT/'summary_statistics.json',{k:v for k,v in payload.items() if k!='runs'})
    if payload['status']=='finished':save(OUT/'results.json',payload)
    return payload


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true');parser.add_argument('--C',type=int)
    parser.add_argument('--seed',type=int);parser.add_argument('--method');parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.worker:return run_method(args.C,args.seed,args.method)
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert set(METHODS)=={'algorithm_1','random_block_selection','dl_clarabel','coordinate_mom','geometric_mom'}
        for seed in SETTINGS['seeds']:
            data,truth,original,amplitude=generate(seed)
            with np.load(OUT/f'data_seed_{seed}.npz') as saved:
                for name,value in [('data',data),('truth',truth),('clean_data',original)]:np.testing.assert_array_equal(saved[name],value)
                assert float(saved['amplitude'])==amplitude
        save(OUT/'settings.json',SETTINGS)
        payload=snapshot()
        subprocess.run([sys.executable,str(OUT/'publish.py')],check=True)
        if args.prepare_only:return
        pending=[(C,seed,name) for C in SETTINGS['C_values'] for name in SETTINGS['execution_order'] for seed in SETTINGS['seeds']
                 if not result_path(C,seed,name).exists()]
        active={};started=time.time();last_snapshot=0.;resolved_since_snapshot=0
        next_publish=50*(payload['finished_attempts']//50+1)
        try:
            while pending or active:
                for pid,item in list(active.items()):
                    proc=item['process']
                    if proc.poll() is None:continue
                    assert proc.returncode==0 and result_path(item['C'],item['seed'],item['method']).exists(),f'Worker failed: {pid} {proc.returncode}'
                    del active[pid];resolved_since_snapshot+=1
                while pending and len(active)<SETTINGS['workers']:
                    C,seed,name=pending.pop(0)
                    proc=subprocess.Popen([sys.executable,'-u',str(OUT/'run.py'),'--worker','--C',str(C),'--seed',str(seed),'--method',name])
                    active[proc.pid]=dict(pid=proc.pid,C=C,seed=seed,method=name,process=proc)
                    print('START',C,seed,name,flush=True)
                if resolved_since_snapshot>=10 or time.time()-last_snapshot>=15 or not (pending or active):
                    payload=snapshot();resolved_since_snapshot=0;last_snapshot=time.time()
                    save(OUT/'active.json',dict(state=payload['status'],pid=os.getpid(),workers=SETTINGS['workers'],
                        started_epoch=started,updated_epoch=time.time(),finished_attempts=payload['finished_attempts'],total_attempts=payload['total_attempts'],
                        active_workers=[{k:v for k,v in item.items() if k!='process'} for item in active.values()]))
                    print('PROGRESS',payload['finished_attempts'],'/',payload['total_attempts'],flush=True)
                    if payload['finished_attempts']>=next_publish:
                        subprocess.run([sys.executable,str(OUT/'publish.py')],check=True)
                        next_publish=50*(payload['finished_attempts']//50+1)
                if pending or active:time.sleep(.25)
        finally:
            subprocess.run([sys.executable,str(OUT/'publish.py')],check=True)
            payload=json.loads((OUT/'progress.json').read_text())
            save(OUT/'active.json',dict(state=payload['status'],pid=os.getpid(),updated_epoch=time.time(),
                finished_attempts=payload['finished_attempts'],total_attempts=payload['total_attempts'],
                active_workers=[{k:v for k,v in item.items() if k!='process'} for item in active.values()]))


if __name__=='__main__':main()
