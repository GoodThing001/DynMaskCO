"""Read-only two-host process snapshot for the current reviewed continuation.

No experiment launch, remote write, scientific replay, or team-process mutation.
Root processes are identified once, then descendants are followed by parent PID.
"""
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex

from _srv import run

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
OUT = EXT / 'results/research_20261010_continuation_v2/review_followup_v3/heartbeat_checks'
REMOTE_CODE = r'''
import datetime,json,os,psutil,subprocess
rows=[]
ownuser=psutil.Process().username()
for p in psutil.process_iter(['pid','ppid','name','cmdline','username']):
 try:
  d=p.info
  if d['username']!=ownuser:continue
  args=d['cmdline'] or []
  cmd=' '.join(args)
  rows.append((p,d,cmd))
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
# Avoid considering this base64 read-only checker an experiment root.
roots=[p.pid for p,d,cmd in rows if ('python' in (d['name'] or '').lower()
       or 'tmux' in (d['name'] or '').lower()) and
       any(marker in cmd.replace('\\','/') for marker in
       ('research_20261010','research_1010','review_3080_20261010',
        'pickup_20round_audit','pickup_midcourse_probe'))]
selected=set(roots)
while True:
 extra={d['pid'] for p,d,cmd in rows if d['ppid'] in selected}
 if extra<=selected:break
 selected|=extra
tree=[]
for p,d,cmd in rows:
 if p.pid not in selected:continue
 try:
  times=p.cpu_times()
  tree.append({'pid':p.pid,'ppid':d['ppid'],'name':d['name'],
    'argv':d['cmdline'],'created_utc':datetime.datetime.fromtimestamp(
    p.create_time(),datetime.timezone.utc).isoformat(),
    'status':p.status(),'cpu_seconds':times.user+times.system,
    'cpu_affinity':p.cpu_affinity(),'rss_bytes':p.memory_info().rss,
    'root_marker_match':p.pid in roots})
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
gpu={}
try:
 r=subprocess.run(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
 '--format=csv,noheader'],capture_output=True,text=True,timeout=15)
 gpu={'exit':r.returncode,'summary':r.stdout.strip()}
except (OSError,subprocess.TimeoutExpired) as e:
 gpu={'unavailable':type(e).__name__}
print(json.dumps({'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'user_scoped_roots':roots,'recursive_process_tree':tree,'gpu':gpu,
 'cpu_busy_percent':psutil.cpu_percent(interval=1,percpu=True),
 'memory_available_bytes':psutil.virtual_memory().available,
 'read_only':True,'started_experiments':0,'stopped_processes':0}))
'''


def probe(host):
    payload = base64.b64encode(REMOTE_CODE.encode()).decode()
    expression = "exec(__import__('base64').b64decode('%s'))" % payload
    if host == '5090':
        cmd = '/home/hzeng/miniconda3/envs/MASKCO_env/bin/python -c ' + shlex.quote(expression)
        config = None
    else:
        ps = "& 'D:\\anaconda\\envs\\MASKCO_env\\python.exe' '-c' '%s'; exit $LASTEXITCODE" % expression.replace("'", "''")
        cmd = 'powershell.exe -NoProfile -NonInteractive -EncodedCommand ' + base64.b64encode(ps.encode('utf-16-le')).decode()
        config = '.vscode/sftp.3080.json'
    try:
        return {'host': host, 'data': json.loads(run(cmd, 50, config))}
    except Exception as exc:
        return {'host': host, 'error_type': type(exc).__name__,
                'process_state_unknown': True}


def main():
    reg = EXT / 'results/research_20261010_continuation_v2/review_followup_v3/CONTINUATION_V3_REGISTRATION.json'
    registration = json.loads(reg.read_text(encoding='utf-8'))
    pins = []
    for ref in registration['pins'] + [registration['parent_program']]:
        p = ROOT / ref['path']
        actual = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
        pins.append({'path': ref['path'], 'expected': ref['sha256'], 'actual': actual,
                     'same': actual == ref['sha256']})
    with ThreadPoolExecutor(max_workers=2) as pool:
        hosts = list(pool.map(probe, ['5090', '3080']))
    d = {'schema': 'heartbeat-current-review-readonly-v1',
         'time_utc': datetime.now(timezone.utc).isoformat(),
         'registration_sha256': hashlib.sha256(reg.read_bytes()).hexdigest(),
         'registered_input_pins': pins, 'hosts': hosts,
         'scientific_verdict_issued': False, 'remote_documents_uploaded': False}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ('CHECK_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    with path.open('x', encoding='utf-8') as f:
        json.dump(d, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({'saved': str(path), 'all_registration_pins_match': all(x['same'] for x in pins),
      'hosts': [{ 'host': h['host'], 'error_type': h.get('error_type'),
        'owned_descendant_count': len(h.get('data', {}).get('recursive_process_tree', [])),
        'gpu': h.get('data', {}).get('gpu')} for h in hosts]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
