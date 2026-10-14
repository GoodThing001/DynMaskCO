"""Exact exclusive deployment or bounded readonly archival, never automatic retry."""
import argparse,ast,hashlib,json,posixpath,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import paramiko
from _srv import _cfg
from _srv_exec_v2 import build,run_capture
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'research/review_3080_20261010/delivery/split_final_timed_execution_root_v1'
CONFIG=str(ROOT/'.vscode/sftp.3080.json');WIN=r'C:\Users\zh\project\MASKCO-Main';PY=r'D:\anaconda\envs\MASKCO_env\python.exe'
REL='research/review_3080_20261010/split_final_timed_execution_root_v1'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(name,x):
    with (O/name).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def connect():
    h,u,p=_cfg(CONFIG);c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect(h,port=22,username=u,password=p,timeout=30);return c
def remote():return posixpath.join(json.loads(Path(CONFIG).read_text(encoding='utf-8'))['remotePath'].rstrip('/'),REL)
def call(script,args=(),timeout=40):return run_capture(build(CONFIG,WIN,PY,'-B',[REL+'/'+script,*args],'windows'),timeout,CONFIG)
def verify_local():
    reg=json.loads((O/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
    binding=json.loads((O/'LOCAL_EXECUTION_BINDING.json').read_text(encoding='utf-8'))
    assert reg['execution_authorized'] and sha(O/'EXECUTION_REGISTRATION.json')==binding['registration_sha256']
    assert sha(__file__)==reg['transport_tool_sha256']
    for path,h in reg['local_transport_helpers'].items():
        assert sha(path)==h
    assert reg['independent_timed_request_checker_sealed']
    for n,h in reg['package_files'].items():
        assert Path(n).name==n and sha(O/n)==h
        if n.endswith('.py'):ast.parse((O/n).read_text(encoding='utf-8'))
    return reg,binding
def deploy():
    reg,binding=verify_local()
    if (O/'DEPLOY_ATTEMPT.json').exists():raise RuntimeError('attempt already exists; inspect actual process/receipts, no retry')
    # One actual resource check precedes every mutation. All owned tree matches
    # use observed argv, not a historical PID. Other users/processes are untouched.
    code="""import json,os,psutil
me={os.getpid(),os.getppid()};procs=[]
for p in psutil.process_iter(['pid','ppid','name','cmdline','username','create_time']):
 try:
  a=p.info['cmdline'] or []
  if p.pid not in me and 'python' in (p.info['name'] or '').lower():procs.append(dict(**p.info,owned=any('review_3080_20261010' in s and s.endswith('.py') for s in a),affinity=p.cpu_affinity(),rss=p.memory_info().rss))
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
print(json.dumps(dict(available_ram_gib=psutil.virtual_memory().available/2**30,allowed_cpu=psutil.Process().cpu_affinity(),cpu15_busy_percent=psutil.cpu_percent(interval=2,percpu=True)[15],processes=procs)))"""
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    raw=run_capture(build(CONFIG,WIN,PY,'-B',['-c',code],'windows'),30,CONFIG)
    save('RESOURCE_PREFLIGHT_'+stamp+'.json',raw)
    if raw['exit_code']!=0:raise RuntimeError('preflight failed; no upload/launch')
    observed=json.loads(raw['stdout'])
    if observed['available_ram_gib']<8 or 15 not in observed['allowed_cpu'] or observed['cpu15_busy_percent']>20 or any(p['owned'] for p in observed['processes']):raise RuntimeError('actual resource/owned process barrier; no upload/launch')
    names=list(reg['package_files'])+['EXECUTION_REGISTRATION.json']
    save('DEPLOY_ATTEMPT.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),registration_sha256=binding['registration_sha256'],resource_observation=observed,remote_directory=REL))
    c=connect()
    try:
        s=c.open_sftp();s.get_channel().settimeout(30);target=remote()
        # mkdir must succeed newly; an existing or uncertain directory is never reused.
        s.mkdir(target);checks={}
        for n in names:
            b=(O/n).read_bytes()
            with s.open(target+'/'+n,'wx') as f:f.write(b)
            with s.open(target+'/'+n,'rb') as f:checks[n]=hashlib.sha256(f.read()).hexdigest()==hashlib.sha256(b).hexdigest()
        save('UPLOAD_ACCEPTANCE.json',checks)
        if not all(checks.values()):raise RuntimeError('upload SHA mismatch; launch prohibited')
        s.close()
    finally:c.close()
    launched=call('score_split_remote_launcher_v1.py',timeout=50);save('LAUNCH_RESPONSE.json',launched)
    if launched['exit_code']!=0:raise RuntimeError('uncertain launch; inspect only')
    inspected=call('score_split_remote_status_v1.py');save('INITIAL_STATUS.json',inspected)
    print(json.dumps(dict(launched=True,launch=json.loads(launched['stdout']),status=json.loads(inspected['stdout']) if inspected['exit_code']==0 else inspected)))
def inspect():
    verify_local();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    r=call('score_split_remote_status_v1.py');save('INSPECTION_'+stamp+'.json',r)
    if r['exit_code']!=0:raise RuntimeError('readonly inspection failed')
    x=json.loads(r['stdout']);print(json.dumps(dict(complete=x['complete'],owned_processes=x['owned_processes'],receipts=x['receipts'],files_count=len(x['files']))))
def download():
    reg,binding=verify_local();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    raw=call('score_split_remote_status_v1.py');save('DOWNLOAD_PREFLIGHT_'+stamp+'.json',raw)
    actual=json.loads(raw['stdout']) if raw['exit_code']==0 else {}
    if not actual.get('complete'):raise RuntimeError('actual chain not ended or owned process active; download prohibited')
    # Seal all actual server file hashes after actual process exit. This is only
    # readonly archival; no solver/model/QA invocation and no input rewriting.
    base=str(Path(WIN)/Path(REL))
    code="""import json,hashlib
from pathlib import Path
b=Path(%r);rows=[]
for p in sorted(b.rglob('*')):
 if not p.is_file():continue
 if p.is_symlink() or not p.resolve().is_relative_to(b.resolve()):raise RuntimeError('outside exclusive artifact tree')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 rows.append(dict(name=p.relative_to(b).as_posix(),size=p.stat().st_size,sha256=h.hexdigest()))
print(json.dumps(rows))"""%base
    hashed=run_capture(build(CONFIG,WIN,PY,'-B',['-c',code],'windows'),50,CONFIG)
    if hashed['exit_code']!=0:raise RuntimeError('actual artifact identity failed')
    manifest=json.loads(hashed['stdout'])
    if any(x['size']>reg['download_max_file_bytes'] or Path(x['name']).is_absolute() or '..' in Path(x['name']).parts for x in manifest) or sum(x['size'] for x in manifest)>reg['download_max_total_bytes']:raise RuntimeError('registered archival bounds')
    dest=O/'downloaded';dest.mkdir(exist_ok=True);mp=O/'DOWNLOAD_MANIFEST.json'
    if mp.exists():
        if json.loads(mp.read_text(encoding='utf-8'))!=manifest:raise RuntimeError('original remote artifacts changed; no overwrite')
    else:save('DOWNLOAD_MANIFEST.json',manifest)
    c=connect()
    try:
        s=c.open_sftp();s.get_channel().settimeout(30);target=remote()
        for x in manifest:
            p=dest/x['name']
            if not p.resolve().is_relative_to(dest.resolve()):raise RuntimeError('outside local archive tree')
            p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():
                if p.stat().st_size!=x['size'] or sha(p)!=x['sha256']:raise RuntimeError('completed local artifact changed')
                continue
            part=dest/(x['name']+'.part');offset=part.stat().st_size if part.exists() else 0
            if offset>x['size']:raise RuntimeError('partial archive exceeds original size')
            with s.open(target+'/'+x['name'],'rb') as f,part.open('ab') as out:
                while offset<x['size']:
                    length=min(262144,x['size']-offset)
                    chunk=list(f.readv([(offset,length)]))[0]
                    if len(chunk)!=length:raise RuntimeError('short bounded readv; partial preserved')
                    out.write(chunk);out.flush();offset+=length
            if sha(part)!=x['sha256']:raise RuntimeError('actual downloaded hash differs; original part preserved')
            part.rename(p)
        s.close()
    finally:c.close()
    assert all((dest/x['name']).stat().st_size==x['size'] and sha(dest/x['name'])==x['sha256'] for x in manifest)
    save('DOWNLOAD_RECEIPT_'+stamp+'.json',dict(files=len(manifest),total_bytes=sum(x['size'] for x in manifest),registration_sha256=binding['registration_sha256'],all_original_hashes_match=True,remote_reexecuted=False))
    print(json.dumps(dict(downloaded=len(manifest),total_bytes=sum(x['size'] for x in manifest))))
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['deploy','inspect','download']);mode=p.parse_args().mode
    try:globals()[mode]()
    except Exception:
        save('TRANSPORT_FAILURE_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json',dict(mode=mode,traceback=traceback.format_exc(),no_automatic_retry=True));raise

if __name__=='__main__':main()
