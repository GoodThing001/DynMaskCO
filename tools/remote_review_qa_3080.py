"""Upload only a new synthetic QA bundle, read archived results, retain exit codes."""
from datetime import datetime, timezone
import hashlib
import json
import posixpath
from pathlib import Path
import sys
import time
import paramiko
from _srv import _cfg
from _srv_exec_v2 import build, run_capture

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'research/review_3080_20261010/delivery'
CONFIG = ROOT / '.vscode/sftp.3080.json'
WIN_ROOT = r'C:\Users\zh\project\MASKCO-Main'
PY = r'D:\anaconda\envs\MASKCO_env\python.exe'
REL = 'research/review_3080_20261010/remote_full_masked_qa_v1'
WRAPPER = '''import os,runpy,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
 os.environ[k]='1'
os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
runpy.run_path(os.path.join(os.path.dirname(__file__),'training_engineering.py'),run_name='__main__')
'''


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    bundle = LOCAL / 'remote_bundle_v1'
    bundle.mkdir(exist_ok=False)
    materials = {n: (LOCAL / n).read_bytes() for n in [
        'training_engineering.py', 'FULL_MASKED_TRAINING_ENGINEERING_REGISTRATION.json']}
    materials['runner.py'] = WRAPPER.encode()
    manifest = {'time_utc':datetime.now(timezone.utc).isoformat(),
                'stage':'synthetic_engineering_only', 'host':'3080_windows_cpu',
                'output_directory_exclusive':REL+'/output', 'max_foreground_seconds':180,
                'research_training_enabled':False, 'scientific_positive_claim_enabled':False,
                'files':{n:sha(raw) for n,raw in materials.items()}}
    materials['REMOTE_ENGINEERING_REGISTRATION.json'] = json.dumps(manifest,indent=2).encode()
    for name,raw in materials.items():
        (bundle / name).write_bytes(raw)
    cfg = json.loads(CONFIG.read_text())
    base = cfg['remotePath'].rstrip('/')
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    host,user,password = _cfg(str(CONFIG))
    client.connect(host,port=22,username=user,password=password,timeout=30)
    sftp = client.open_sftp()
    try:
        # A zero-byte local placeholder cannot establish a cross-host result.
        # Read the archived remote original into a new local audit path only.
        old = 'C-VRP_Cold-chainVehicleRoutingProblem/results/research_20261010_continuation_v2/delivery/dev_gate_3080_v1/result.json'
        archived = {'relative_remote_path':old}
        try:
            with sftp.open(posixpath.join(base,old),'rb') as f:
                raw=f.read()
            archived.update(bytes=len(raw),sha256=sha(raw),json_parseable=True)
            payload=json.loads(raw)
            (bundle/'ARCHIVED_REMOTE_DEV_RESULT.json').write_bytes(raw)
            archived['top_level_fields']=list(payload)
        except FileNotFoundError:
            archived['state']='remote_path_missing'
        except (ValueError,OSError) as exc:
            archived['state']='read_or_parse_failed:'+type(exc).__name__
        (bundle/'ARCHIVED_REMOTE_DEV_IDENTITY.json').write_text(json.dumps(archived,indent=2))
        target=posixpath.join(base,REL)
        try:
            sftp.stat(target)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError('Exclusive QA directory already exists; no rerun or overwrite')
        cur=base
        for part in REL.split('/'):
            cur=posixpath.join(cur,part)
            try:sftp.stat(cur)
            except FileNotFoundError:sftp.mkdir(cur)
        for name,raw in materials.items():
            with sftp.open(posixpath.join(target,name),'wb') as f:f.write(raw)
    finally:
        sftp.close();client.close()
    command=build(str(CONFIG),WIN_ROOT,PY,REL+'/runner.py',[
        '--stage','synthetic_engineering_only','--upstream','MASKCO_code',
        '--registration',REL+'/FULL_MASKED_TRAINING_ENGINEERING_REGISTRATION.json',
        '--out',REL+'/output'],'windows')
    start=time.perf_counter()
    r=run_capture(command,180,str(CONFIG))
    (bundle/'RUN_EXIT.json').write_text(json.dumps({'exit_code':r['exit_code'],
        'wall_s':time.perf_counter()-start,'stdout':r['stdout'],'stderr':r['stderr']},indent=2))
    client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host,port=22,username=user,password=password,timeout=30)
    sftp=client.open_sftp()
    try:
        checks={}
        for name,raw in materials.items():
            with sftp.open(posixpath.join(target,name),'rb') as f:checks[name]=sha(f.read())==sha(raw)
        if r['exit_code']==0:
            with sftp.open(posixpath.join(target,'output/ENGINEERING_QA.json'),'rb') as f:
                q=f.read()
            (bundle/'INDEPENDENT_REMOTE_ENGINEERING_QA.json').write_bytes(q)
        (bundle/'SOURCE_END_CHECK.json').write_text(json.dumps(checks,indent=2))
    finally:sftp.close();client.close()
    print(json.dumps({'archived_remote_dev':archived,'remote_exit':r['exit_code'],
                      'actual_source_end_unchanged':all(checks.values()),
                      'qa_downloaded':(bundle/'INDEPENDENT_REMOTE_ENGINEERING_QA.json').exists()}))
    return r['exit_code']


if __name__=='__main__':
    raise SystemExit(main())
