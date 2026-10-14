"""Read-only actual tree/status and incremental small phase artifacts, no relaunch."""
import hashlib,json,posixpath
from datetime import datetime,timezone
from pathlib import Path
import paramiko
from _srv import _cfg
from _srv_exec_v2 import build,run_capture
from deploy_delivery_multistep_family2 import OUT,REL,WIN,PY,CONFIG
def main():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=OUT/('inspection_'+stamp);out.mkdir()
    r=run_capture(build(CONFIG,WIN,PY,'-B',[REL+'/status.py'],'windows'),30,CONFIG)
    (out/'STATUS.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    if r['exit_code']!=0:raise ValueError('Actual tree lookup failed')
    host,user,password=_cfg(CONFIG);c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect(host,port=22,username=user,password=password,timeout=30)
    try:
        s=c.open_sftp();remote=posixpath.join(json.loads(Path(CONFIG).read_text())['remotePath'].rstrip('/'),REL);manifest=json.loads((OUT/'UPLOAD_MANIFEST.json').read_text());checks={}
        for n,h in manifest['source_sha256'].items():
            with s.open(remote+'/'+n,'rb') as f:checks[n]=hashlib.sha256(f.read()).hexdigest()==h
        for n in ('progress.json','SOURCE_START.json','SOURCE_END.json','RESULT.json','FAILURE.json','RUN_EXIT.json','WORKER_IDENTITY.json'):
            try:
                with s.open(remote+'/'+n,'rb') as f:raw=f.read()
            except FileNotFoundError:continue
            try:json.loads(raw)
            except json.JSONDecodeError:continue # Atomic evidence is immutable; live progress may be mid-write.
            (out/n).write_bytes(raw)
        for n in ('run.stdout.log','run.stderr.log'):
            try:
                with s.open(remote+'/'+n,'rb') as f:
                    size=s.stat(remote+'/'+n).st_size;f.seek(max(0,size-10000));raw=f.read(10000)
                (out/n).write_bytes(raw)
            except FileNotFoundError:pass
        s.close()
    finally:c.close()
    (out/'ACTUAL_INPUT_HASHES.json').write_text(json.dumps(checks,indent=2));assert all(checks.values())
    answer={'finished':json.loads(r['stdout'])['finished'],'actual_inputs_match':all(checks.values()),'inspection':str(out)}
    for n in ('progress.json','RUN_EXIT.json','FAILURE.json'):
        if (out/n).exists():answer[n]=json.loads((out/n).read_text(encoding='utf-8'))
    print(json.dumps(answer))
if __name__=='__main__':main()
