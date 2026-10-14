"""After auditv1 has failed/exited, new byte-pinned streaming diagnostic, same raw data."""
import hashlib,json,posixpath
from datetime import datetime,timezone
from pathlib import Path
import paramiko
from _srv import _cfg
from _srv_exec_v2 import build,run_capture
from deploy_delivery_multistep_family2 import OUT as SOURCE,REL as SOURCE_REL,WIN,PY,CONFIG
ROOT=Path(__file__).resolve().parents[1];PRIOR=SOURCE.parent/'multistep_insertable_family2_data_audit_root_v1';OUT=SOURCE.parent/'multistep_insertable_family2_data_audit_root_v2';PRIOR_REL='research/review_3080_20261010/multistep_insertable_family2_data_audit_root_v1';REL='research/review_3080_20261010/multistep_insertable_family2_data_audit_root_v2'
def sha(b):return hashlib.sha256(b).hexdigest()
def create(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    if OUT.exists():raise FileExistsError('Recovery already exists; inspect actual tree, no relaunch')
    status=run_capture(build(CONFIG,WIN,PY,'-B',[PRIOR_REL+'/status.py'],'windows'),30,CONFIG)
    if status['exit_code']!=0:raise RuntimeError('Actual original audit tree lookup failed')
    state=json.loads(status['stdout'])
    if not state['finished']:raise RuntimeError('Original audit still active; wait, never duplicate it')
    if state['run_exit']['exit_code']==0:raise RuntimeError('Original audit passed; no engineering recovery authorized by a failure')
    host,user,password=_cfg(CONFIG);c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect(host,port=22,username=user,password=password,timeout=30)
    try:
        s=c.open_sftp();remote=json.loads(Path(CONFIG).read_text())['remotePath'].rstrip('/');prior=posixpath.join(remote,PRIOR_REL);src=posixpath.join(remote,SOURCE_REL)
        saved=PRIOR/'downloaded';saved.mkdir(exist_ok=False)
        for name in ('RUN_EXIT.json','AUDIT_FAILURE.json','INDEPENDENT_FAMILY2_DATA_ACCEPTANCE.json','AUDIT_SOURCE_START.json','AUDIT_SOURCE_END.json','WORKER_IDENTITY.json','worker.pid','run.stderr.log'):
            try:
                with s.open(prior+'/'+name,'rb') as f:raw=f.read()
            except FileNotFoundError:continue
            (saved/name).write_bytes(raw)
        if not (saved/'AUDIT_FAILURE.json').exists():raise RuntimeError('No proven audit program exception; review rather than numeric rerun')
        failure=json.loads((saved/'AUDIT_FAILURE.json').read_text())['traceback']
        if 'NoneType' not in failure:raise RuntimeError('Unproven audit recovery root cause; preserve and diagnose first')
        worker=int((saved/'worker.pid').read_text());supervisor=json.loads((saved/'WORKER_IDENTITY.json').read_text())['supervisor_pid']
        checkcode='import psutil,json; ids='+repr([worker,supervisor])+"; print(json.dumps({'alive':[i for i in ids if psutil.pid_exists(i)]}))"
        check=run_capture(build(CONFIG,WIN,PY,'-B',['-c',checkcode],'windows'),30,CONFIG)
        if check['exit_code']!=0 or json.loads(check['stdout'])['alive']:raise RuntimeError('Original audit process still exists; wait')
        original=json.loads((PRIOR/'AUDIT_REGISTRATION.json').read_text());program=(ROOT/'tools/audit_delivery_multistep_family2_data.py').read_bytes();compile(program,'audit.py','exec')
        OUT.mkdir(exist_ok=False)
        reg=dict(original,created_utc=datetime.now(timezone.utc).isoformat(),program_sha256=sha(program),engineering_recovery_of=PRIOR_REL,original_audit_registration_sha256=sha((PRIOR/'AUDIT_REGISTRATION.json').read_bytes()),original_audit_failure_sha256=sha((saved/'AUDIT_FAILURE.json').read_bytes()),same_original_raw_inputs=True,change='Stream original1GiB+ raw rows and explicitly audit stored incomplete candidate/None report and fallback; no solver, teacher, training or scientific rule change')
        create(OUT/'AUDIT_REGISTRATION.json',reg);create(OUT/'ORIGINAL_EXITED_PROCESS_CHECK.json',check)
        # Every scientific raw input remains bound to the exact earlier audit SHA.
        verifycode="import hashlib,json,pathlib; base=pathlib.Path(r'"+WIN+"')/r'"+SOURCE_REL+"'; expected="+repr(original['inputs'])+"; print(json.dumps({n:hashlib.sha256((base/n).read_bytes()).hexdigest()==h for n,h in expected.items()}))"
        actual=run_capture(build(CONFIG,WIN,PY,'-B',['-c',verifycode],'windows'),60,CONFIG);create(OUT/'ACTUAL_ORIGINAL_INPUTS.json',actual)
        if actual['exit_code']!=0 or not all(json.loads(actual['stdout']).values()):raise RuntimeError('Original raw byte identity changed; no recovery')
        files={'audit.py':program,'AUDIT_REGISTRATION.json':(OUT/'AUDIT_REGISTRATION.json').read_bytes()}
        for n in ('supervisor.py','launcher.py','status.py'):files[n]=(PRIOR/n).read_bytes()
        target=posixpath.join(remote,REL);s.mkdir(target);checks={}
        for n,b in files.items():
            if n!='AUDIT_REGISTRATION.json':(OUT/n).write_bytes(b)
            with s.open(target+'/'+n,'wx') as f:f.write(b)
            with s.open(target+'/'+n,'rb') as f:checks[n]=sha(f.read())==sha(b)
        s.close();create(OUT/'UPLOAD_ACCEPTANCE.json',checks)
        if not all(checks.values()):raise ValueError('Recovery uploaded source mismatch')
    finally:c.close()
    launch=run_capture(build(CONFIG,WIN,PY,'-B',[REL+'/launcher.py'],'windows'),60,CONFIG);create(OUT/'LAUNCH_RESPONSE.json',launch)
    if launch['exit_code']!=0:raise RuntimeError('Uncertain recovery launch; actual tree before another action')
    print(json.dumps({'launch_exit':0,'same_original_data':True,'solver_rerun':False,'training_started':False,'original_scientific_FAIL_preserved':True}))
if __name__=='__main__':main()
