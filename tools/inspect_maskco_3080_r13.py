"""One read-only current 3080 resource/owner observation for r13."""
import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from _srv_exec_v2 import build, run_capture

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/review_3080_20261010/goal_round_20261013_r13_model_action_resource'


def main():
    reg = json.loads((OUT / 'RESOURCE_REGISTRATION.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == reg['source_sha256']
    code = '''import json,os,psutil
me={os.getpid(),os.getppid()}; rows=[]
for p in psutil.process_iter(['pid','ppid','name','cmdline','username','create_time']):
 try:
  a=p.info['cmdline'] or []
  if p.pid not in me and 'python' in (p.info['name'] or '').lower():
   rows.append(dict(pid=p.pid,ppid=p.info['ppid'],username=p.info['username'],create_time=p.info['create_time'],owned=any('review_3080_20261010' in x and x.endswith('.py') for x in a),affinity=p.cpu_affinity(),rss=p.memory_info().rss,script=[x for x in a if x.endswith('.py')][-1:] or []))
 except (psutil.NoSuchProcess,psutil.AccessDenied): pass
print(json.dumps(dict(available_ram_gib=psutil.virtual_memory().available/2**30,cpu15_busy_percent=psutil.cpu_percent(interval=2,percpu=True)[15],allowed_cpu=psutil.Process().cpu_affinity(),python_processes=rows,gpu_used=False)))'''
    cfg = str(ROOT / '.vscode/sftp.3080.json')
    receipt = {'created_utc': datetime.now(timezone.utc).isoformat(), 'registration_sha256': hashlib.sha256((OUT / 'RESOURCE_REGISTRATION.json').read_bytes()).hexdigest(),
               'server_upload': False, 'model_or_solver_executed': False}
    try:
        raw = run_capture(build(cfg, r'C:\Users\zh\project\MASKCO-Main', r'D:\anaconda\envs\MASKCO_env\python.exe', '-B', ['-c', code], 'windows'), 40, cfg)
        receipt['transport_exit_code'] = raw['exit_code']
        if raw['exit_code'] == 0:
            receipt['actual_server'] = json.loads(raw['stdout'])
        else:
            receipt['transport_failure'] = {'stdout': raw['stdout'], 'stderr': raw['stderr']}
    except Exception:
        receipt['transport_exception'] = traceback.format_exc()
    with (OUT / 'RESOURCE_OBSERVATION.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
    server = receipt.get('actual_server')
    print(json.dumps({'observed': bool(server), 'ram_gib': server['available_ram_gib'] if server else None,
                      'cpu15_busy_percent': server['cpu15_busy_percent'] if server else None,
                      'owned_research_trees': sum(p['owned'] for p in server['python_processes']) if server else None}))


if __name__ == '__main__':
    main()
