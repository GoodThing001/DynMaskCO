"""Audit and synchronize the user-owned 3080/5090 project without live-job resets.

Credentials are consumed only from the existing local SFTP profiles. Reports
never contain credentials. Synchronization uses explicit file manifests and
recoverable server-side archives, never a blind recursive mirror or git reset.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings('ignore', category=CryptographyDeprecationWarning)
import paramiko

SCRIPTS = next(p / 'scripts' for p in Path(__file__).resolve().parents
               if (p / 'scripts/project_paths.py').is_file())
sys.path.insert(0, str(SCRIPTS))
from project_paths import EXTENSION_ROOT as EXT, WORKSPACE_ROOT as ROOT

OUT = EXT / 'results/workspace_sync_20261021'
PROFILES = {
    '5090': dict(config='sftp.json', python='/home/hzeng/envs/MASKCO_env/bin/python', windows=False),
    '3080': dict(config='sftp.3080.json', python='D:/anaconda/envs/cc_compare/python.exe', windows=True),
}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def connect(label):
    profile = PROFILES[label]
    settings = json.loads((ROOT / '.vscode' / profile['config']).read_text(encoding='utf-8-sig'))
    remote = settings['remotePath'].rstrip('/')
    if profile['windows']:
        if not remote.startswith('/C:/Users/zh/project/'):
            raise ValueError('Unexpected Windows project root')
    elif not remote.startswith('/home/hzeng/project/'):
        raise ValueError('Unexpected Linux project root')
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(settings['host'], port=settings.get('port', 22), username=settings['username'],
                   password=settings.get('password'), timeout=20, look_for_keys=False, allow_agent=False)
    return client, remote


def command(label, code):
    profile = PROFILES[label]
    encoded = base64.b64encode(code.encode('utf-8')).decode('ascii')
    inline = "import base64;exec(base64.b64decode('%s'))" % encoded
    if profile['windows']:
        ps = "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); & '" + profile['python'] + "' -B -c \"" + inline + "\""
        return 'powershell.exe -NoLogo -NoProfile -NonInteractive -EncodedCommand ' + base64.b64encode(ps.encode('utf-16le')).decode('ascii')
    return shlex.quote(profile['python']) + ' -B -c ' + shlex.quote(inline)


def execute(client, label, code, timeout=120):
    _, stdout, stderr = client.exec_command(command(label, code), timeout=timeout)
    out = stdout.read().decode('utf-8', 'replace')
    err = stderr.read().decode('utf-8', 'replace')
    result = stdout.channel.recv_exit_status()
    if result:
        raise RuntimeError('Server operation failed: ' + str(result) + '; ' + err[-1800:])
    return json.loads(out)


def audit_remote(label):
    client, remote = connect(label)
    native = remote.lstrip('/') if PROFILES[label]['windows'] else remote
    code = r'''
import os,json,stat,subprocess,platform,shutil
from pathlib import Path
try:
 import psutil
except ImportError:
 psutil=None
r=Path(ROOT)
def ordinary(p):
 s=p.lstat()
 return not (stat.S_ISLNK(s.st_mode) or getattr(s,'st_file_attributes',0)&1024)
def normalize(value):return str(value).replace('\\','/').lower().rstrip('/')
prefix=normalize(r)
active=[]
for proc in (psutil.process_iter(['pid','name','cwd','cmdline','create_time']) if psutil else []):
 try:
  info=proc.info;args=info.get('cmdline') or [];cwd=info.get('cwd') or ''
  paths=[a for a in args if normalize(a).startswith(prefix+'/') and len(a)<500 and '\n' not in a]
  if paths or normalize(cwd)==prefix or normalize(cwd).startswith(prefix+'/'):
   active.append(dict(pid=info['pid'],name=info['name'],cwd=cwd,project_paths=paths,create_time=info['create_time']))
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
if psutil is None and os.name=='nt':
 import ctypes
 from ctypes import wintypes
 split=ctypes.windll.shell32.CommandLineToArgvW
 split.argtypes=[wintypes.LPCWSTR,ctypes.POINTER(ctypes.c_int)]
 split.restype=ctypes.POINTER(wintypes.LPWSTR)
 def arguments(value):
  count=ctypes.c_int();items=split(value,ctypes.byref(count))
  if not items:return []
  try:return [items[i] for i in range(count.value)]
  finally:ctypes.windll.kernel32.LocalFree(items)
 probe=subprocess.run(['powershell.exe','-NoLogo','-NoProfile','-NonInteractive','-Command',"[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*MASKCO-Main*' } | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"],capture_output=True,text=True,encoding='utf-8')
 records=json.loads(probe.stdout.strip() or '[]')
 if isinstance(records,dict):records=[records]
 for row in records:
  paths=[a for a in arguments(row.get('CommandLine') or '') if normalize(a).startswith(prefix+'/') and len(a)<500 and '\n' not in a]
  if paths:active.append(dict(pid=row['ProcessId'],name=row['Name'],cwd=None,project_paths=paths,create_time=None))
tree=[]
for p in sorted(r.iterdir(),key=lambda p:p.name.lower()):
 s=p.lstat();row=dict(name=p.name,directory=stat.S_ISDIR(s.st_mode),ordinary=ordinary(p),bytes=s.st_size)
 if row['directory'] and row['ordinary'] and p.name not in {'.git','.vscode','.codex','.cursor','.claude','private'}:
  row['children']=[q.name for q in p.iterdir()][:100]
 tree.append(row)
g=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
tmux=[]
if os.name!='nt':
 t=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True);tmux=t.stdout.splitlines()
git=dict(present=(r/'.git').exists())
if git['present']:
 try:
  v=subprocess.run(['git','--no-optional-locks','-C',str(r),'rev-parse','HEAD'],capture_output=True,text=True);git['head']=v.stdout.strip();git['exit_code']=v.returncode
 except FileNotFoundError:git['command_unavailable']=True
print(json.dumps(dict(root=str(r),platform=platform.platform(),python=platform.python_version(),disk_free_bytes=shutil.disk_usage(r).free,top_level=tree,active_project_processes=active,process_audit='psutil' if psutil else 'Windows CIM project command paths; cwd unavailable',gpu=g.stdout.strip(),tmux_sessions=tmux,git=git)))
'''.replace('ROOT', repr(native))
    try:
        result = execute(client, label, code)
        save(OUT / ('server_' + label + '_before.json'), result)
        print(json.dumps(dict(server=label, root=result['root'], platform=result['platform'],
             free_GiB=round(result['disk_free_bytes']/2**30, 1),
             top_level=[r['name'] for r in result['top_level']],
             active_processes=result['active_project_processes'], gpu=result['gpu'],
             tmux_sessions=result['tmux_sessions']), ensure_ascii=False), flush=True)
    finally:
        client.close()


def stop_authorized_3080_legacy():
    """Stop only the explicitly approved old CC processes, with a fresh check."""
    label = '3080'
    before = json.loads((OUT / 'server_3080_before.json').read_text(encoding='utf-8'))
    approved = [r['pid'] for r in before['active_project_processes']]
    if set(approved) != {16964, 2400, 18836}:
        raise ValueError('Unexpected original task set; do not broaden termination scope')
    client, remote = connect(label)
    native = remote.lstrip('/')
    code = r'''
import json,subprocess,time
approved=__APPROVED_PIDS__
prefix=PREFIX.lower().replace(chr(92),'/').rstrip('/')+'/cc_compare/'
query="[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*MASKCO-Main*' } | Select-Object ProcessId,Name,CommandLine,ParentProcessId | ConvertTo-Json -Compress"
def current():
 p=subprocess.run(['powershell.exe','-NoLogo','-NoProfile','-NonInteractive','-Command',query],capture_output=True,text=True,encoding='utf-8')
 if p.returncode:raise RuntimeError('CIM refresh failed')
 rows=json.loads(p.stdout.strip() or '[]')
 return rows if isinstance(rows,list) else [rows]
rows=current();targets=[]
for pid in approved:
 row=next((r for r in rows if r['ProcessId']==pid),None)
 if row is None:continue
 cmd=(row.get('CommandLine') or '').lower().replace(chr(92),'/')
 if prefix not in cmd:raise RuntimeError('Approved PID no longer matches the old project task')
 targets.append(dict(pid=pid,name=row['Name'],parent=row['ParentProcessId']))
results=[]
for row in targets:
 p=subprocess.run(['taskkill.exe','/PID',str(row['pid']),'/T','/F'],capture_output=True,text=True)
 results.append(dict(pid=row['pid'],taskkill_returncode=p.returncode))
time.sleep(1)
remaining=[r['ProcessId'] for r in current() if r['ProcessId'] in approved]
if remaining:raise RuntimeError('Some approved legacy tasks remain')
print(json.dumps(dict(status='STOPPED_USER_APPROVED_OLD_RRNCO_TASKS',approved_pids=approved,checked_targets=targets,results=results,remaining_approved_pids=remaining,other_gpu_or_project_jobs_not_targeted=True)))
'''.replace('__APPROVED_PIDS__', repr(approved)).replace('PREFIX', repr(native))
    try:
        result = execute(client, label, code)
        save(OUT / 'server_3080_approved_legacy_stop.json', result)
        print(json.dumps(result), flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit', 'stop-approved-legacy'])
    parser.add_argument('--server', choices=list(PROFILES), required=True)
    args = parser.parse_args()
    if args.action == 'audit': audit_remote(args.server)
    elif args.action == 'stop-approved-legacy':
        if args.server != '3080': raise ValueError('Approval applies only to the 3080 old tasks')
        stop_authorized_3080_legacy()
