import json,time,argparse
from pathlib import Path
p=Path(__file__).with_name('progress.json')
parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
try:
 while True:
  d=json.loads(p.read_text());summary=d['summary']
  done=sum(sum(v.get(k,0) for k in ['completed','unconverged','failed']) for v in summary.values())
  print(time.strftime('%Y-%m-%d %H:%M:%S'),d['status'],f'{done}/1300 attempted')
  for name,v in summary.items():print(f"  {name}: converged={v['completed']}, unconverged={v['unconverged']}, failed={v['failed']}")
  for a in d['active']:print(f"  RUNNING seed {a['seed']}: {a['method']}, {(time.time()-a['started_at'])/60:.1f} min")
  print(flush=True)
  if not args.watch:break
  time.sleep(10)
except KeyboardInterrupt:pass
