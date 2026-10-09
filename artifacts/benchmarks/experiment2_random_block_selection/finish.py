import json, subprocess, sys, time, os
from pathlib import Path
OUT = Path(__file__).resolve().parent
while True:
    progress = json.loads((OUT / 'progress.json').read_text())
    if progress['status'] == 'cancelled':
        print('Cancelled: no results published', flush=True)
        break
    if progress['status'] == 'finished' and (OUT / 'background_resumed.json').exists():
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        result = subprocess.run([sys.executable, str(OUT / 'publish.py')], env=env)
        record = dict(epoch=time.time(), returncode=result.returncode)
        (OUT / 'publication_status.json').write_text(json.dumps(record, indent=2) + '\n')
        sys.exit(result.returncode)
    time.sleep(5)
