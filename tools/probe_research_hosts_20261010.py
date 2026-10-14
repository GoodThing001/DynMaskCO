"""Read-only resource checks for the two user-authorized hosts."""
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
from _srv_exec_v2 import run_capture

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'research/review_3080_20261010/resources'
LINUX = r'''
import json,subprocess,sys,os
def cmd(args):
 p=subprocess.run(args,text=True,capture_output=True,timeout=30)
 return {'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
print(json.dumps({'python':sys.version,'cpu_affinity':sorted(os.sched_getaffinity(0)),
 'gpu':cmd(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader']),
 'gpu_processes':cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])}))
'''
WINDOWS = r'''
$g = & nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
$gexit = $LASTEXITCODE
$gp = & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
$pexit = $LASTEXITCODE
$ps = @(Get-Process python,pythonw -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU)
@{gpu=@($g);gpu_exit=$gexit;gpu_processes=@($gp);process_exit=$pexit;python_processes=$ps} | ConvertTo-Json -Depth 5 -Compress
'''


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    linux_command = "/home/hzeng/miniconda3/envs/MASKCO_env/bin/python -c \"import base64;exec(base64.b64decode('%s'))\"" % base64.b64encode(LINUX.encode()).decode()
    ps = 'powershell.exe -NoProfile -NonInteractive -EncodedCommand ' + base64.b64encode(WINDOWS.encode('utf-16-le')).decode()
    report = {'time_utc': datetime.now(timezone.utc).isoformat(), 'kind': 'read_only_resource_probe'}
    for name, config, command in [('5090', None, linux_command), ('3080', '.vscode/sftp.3080.json', ps)]:
        try:
            r = run_capture(command, 90, config)
            report[name] = {'transport_exit_code':r['exit_code'],
                            'data':json.loads(r['stdout']), 'stderr':r['stderr']}
        except Exception as exc:
            # Avoid including a connection exception containing configuration values.
            report[name] = {'error_type': type(exc).__name__, 'resource_state':'unverified'}
    path = DEST / ('RESOURCE_PROBE_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
    with path.open('x', encoding='utf-8') as f:
        json.dump(report,f,indent=2,ensure_ascii=False)
    print(json.dumps(report,indent=2,ensure_ascii=True))


if __name__ == '__main__':
    main()
