"""Run the two requested dimension batches sequentially, with four workers total."""
import fcntl, json, os, subprocess, sys, time
from pathlib import Path
OUT=Path(__file__).resolve().parent
REPO = next(parent for parent in Path(__file__).resolve().parents if (parent / 'IP_algorithm.py').is_file())
ARTIFACTS = REPO / 'artifacts'
folders=['experiment11_dense_d5','experiment12_dense_d20']
def save(name,value):
    path=OUT/name;temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)
def main():
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for folder in folders:
            target=ARTIFACTS /folder
            cfg=json.loads((target/'config.json').read_text())
            total=len(cfg['methods'])*len(cfg['seeds'])
            result_path=target/'results.json'
            if result_path.exists():
                previous=json.loads(result_path.read_text())
                if previous['status']=='finished' and previous['finished_attempts']==total and previous['settings']['methods']==cfg['methods']:
                    print('BATCH ALREADY FINISHED',folder,flush=True)
                    continue
            with (target/'runner.log').open('a') as log:
                proc=subprocess.Popen([sys.executable,'-u',str(target/'run.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                save('progress.json',dict(state='running',active_batch=folder,pid=proc.pid,updated_epoch=time.time()))
                (target/'runner_pid.json').write_text(json.dumps(dict(pid=proc.pid,parent_pid=os.getpid(),started_epoch=time.time(),script=str(target/'run.py')),indent=2)+'\n')
                code=proc.wait()
            if code:
                save('progress.json',dict(state='failed',active_batch=folder,exit_code=code,updated_epoch=time.time()))
                raise SystemExit(code)
            result=json.loads((target/'results.json').read_text())
            assert result['status']=='finished' and result['finished_attempts']==result['total_attempts']==total
            print('BATCH FINISHED',folder,flush=True)
        save('progress.json',dict(state='finished',batches=folders,updated_epoch=time.time()))
if __name__=='__main__':main()
