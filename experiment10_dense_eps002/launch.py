"""Drain previous timed workers, publish its checkpoint, then run Experiment 10."""
from pathlib import Path
import json,subprocess,sys,time,os,signal,shlex,importlib.util,re
OUT=Path(__file__).resolve().parent
def save(p,v):
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(v,indent=2)+'\n');os.replace(temp,p)
previous=json.loads((OUT/'previous_runner_pending.json').read_text())
OLD=Path(previous['script']).parent
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
def processes():
    return {int(p):(int(pp),state,shlex.split(cmd)) for row in
            subprocess.check_output(['ps','-axo','pid=,ppid=,state=,command='],text=True).splitlines()
            if len(f:=row.strip().split(None,3))==4 for p,pp,state,cmd in [f]}
started=False
try:
    while True:
        procs=processes()
        root=procs.get(previous['pid'])
        assert root and previous['script'] in root[2] and 'T' in root[1],root
        waiting=[]
        for w in previous['children']:
            path=OLD/f"seed_{w['seed']}_{w['method']}.json"
            proc=procs.get(w['pid'])
            alive=proc is not None and 'Z' not in proc[1]
            if alive:waiting.append(w)
            elif not path.exists():raise RuntimeError(f'Previous worker died without checkpoint: {w}')
        save(OUT/'launch_status.json',dict(state='draining_previous' if waiting else 'ready',
             previous_pid=previous['pid'],waiting=waiting,updated_epoch=time.time()))
        if not waiting:break
        time.sleep(5)
    subprocess.run([sys.executable,str(OLD/'publish.py')],env=env,check=True)
    for name,key in [('progress.json','status'),('active.json','state'),('runner_pid.json','status')]:
        value=json.loads((OLD/name).read_text())
        value[key]='paused_for_experiment10'
        value['pause_reason']='Current fits completed; resumes automatically after Experiment 10'
        save(OLD/name,value)
    report=OUT.parent/'results.md'
    text=report.read_text()
    a=text.index('<!-- experiment9-dense:start -->')
    b=text.index('<!-- experiment9-dense:end -->',a)
    section=text[a:b].replace('Batch status: **running**','Batch status: **paused_for_experiment10**')
    report.write_text(text[:a]+section+text[b:])
    with (OUT/'run.log').open('a') as log:
        proc=subprocess.Popen([sys.executable,'-u',str(OUT/'run.py')],
             stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    started=True
    save(OUT/'runner_pid.json',dict(pid=proc.pid,status='running',workers=4,
         started_epoch=time.time(),experiment_number=10,methods=8))
    save(OUT/'launch_status.json',dict(state='started',pid=proc.pid,previous_pid=previous['pid'],
         previous_resolved=json.loads((OLD/'progress.json').read_text())['finished_attempts'],
         updated_epoch=time.time()))
    print('STARTED Experiment 10:',proc.pid,flush=True)
finally:
    if not started:
        root=processes().get(previous['pid'])
        if root and previous['script'] in root[2]:os.kill(previous['pid'],signal.SIGCONT)
        save(OUT/'launch_status.json',dict(state='launch_failed_previous_resumed',epoch=time.time()))

