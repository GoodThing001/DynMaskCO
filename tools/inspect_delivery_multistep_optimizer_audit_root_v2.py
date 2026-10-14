"""Read actual owned process tree and stage files; never launch or rerun."""
import hashlib,json,posixpath,sys
from datetime import datetime,timezone
from pathlib import Path
import paramiko
from _srv import _cfg
from _srv_exec_v2 import build,run_capture
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/review_3080_20261010/delivery/multistep_optimizer_audit_root_v2'
REL='research/review_3080_20261010/multistep_optimizer_audit_root_v2';WIN=r'C:\Users\zh\project\MASKCO-Main';PY=r'D:\anaconda\envs\MASKCO_env\python.exe';CONFIG=str(ROOT/'.vscode/sftp.3080.json')
def main():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    observed=run_capture(build(CONFIG,WIN,PY,'-B',[REL+'/status.py'],'windows'),30,CONFIG)
    with (OUT/('PROCESS_STATUS_'+stamp+'.json')).open('x',encoding='utf-8') as f:json.dump(observed,f,indent=2)
    if observed['exit_code']!=0:print(json.dumps(observed));return 1
    status=json.loads(observed['stdout']);host,user,password=_cfg(CONFIG)
    client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy());client.connect(host,port=22,username=user,password=password,timeout=30)
    files={};downloaded=[]
    try:
        s=client.open_sftp();remote=posixpath.join(json.loads(Path(CONFIG).read_text())['remotePath'].rstrip('/'),REL)
        files={a.filename:a.st_size for a in s.listdir_attr(remote)}
        if status['finished']:
            dest=OUT/'downloaded';dest.mkdir(exist_ok=True)
            for name in ('RUN_EXIT.json','WORKER_IDENTITY.json','LAUNCH.json','run.stdout.log','run.stderr.log','FAILURE.json','AUDIT_FAILURE.json','INDEPENDENT_OPTIMIZER_ADJUDICATION.json','AUDIT_SOURCE_START.json','AUDIT_SOURCE_END.json','FULL_SOURCE_START.json','FULL_SOURCE_END.json','GRADIENT_0.0.json','GRADIENT_0.2.json','RESULT.json'):
                if name not in files:continue
                target=dest/name
                if target.exists():continue
                with s.open(remote+'/'+name,'rb') as f:
                    blocks=[(i,min(1024*1024,files[name]-i)) for i in range(0,files[name],1024*1024)]
                    data=b''.join(f.readv(blocks))
                if len(data)!=files[name]:raise RuntimeError('Incomplete read-only stage download: '+name)
                with target.open('xb') as f:f.write(data)
                downloaded.append({'name':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
            with (OUT/('DOWNLOAD_RECEIPT_'+stamp+'.json')).open('x') as f:json.dump({'files':downloaded,'experiment_rerun':False},f,indent=2)
        s.close()
    finally:client.close()
    print(json.dumps({'status':status,'files':files,'downloaded':len(downloaded)}));return 0
if __name__=='__main__':raise SystemExit(main())
