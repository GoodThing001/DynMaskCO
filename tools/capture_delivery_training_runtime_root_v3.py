"""Exclusive PY-only transport recovery from Windows SSH command-size failure."""
import hashlib,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
import paramiko
from _srv import _cfg
from _srv_exec_v2 import build,run_capture
from capture_delivery_training_runtime_root_v1 import ROOT,REMOTE

OUT=ROOT/'research/review_3080_20261010/delivery/training_runtime_preflight_root_v3'
OLD=OUT.parent/'training_runtime_preflight_root_v1'
REMOTE_DIR='C:/Users/zh/project/MASKCO-Main/research/review_3080_20261010/training_runtime_preflight_root_v3'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def main():
    OUT.mkdir(exist_ok=False)
    failure=OLD/'TRANSPORT_FAILURE.json'
    if not failure.exists():write(failure,{'reason':'SSHException Channel closed while submitting long encoded command',
        'failure_stage':'SSH exec channel; no remote completion received','actual_remote_start_known':False,
        'old_program_sha256':sha(OLD/'REMOTE_CENSUS_PROGRAM.py'),'registration_sha256':sha(OLD/'REGISTRATION.json')})
    cpu_python=r'D:\anaconda\envs\MASKCO_env\python.exe'
    # Read actual Python process metadata before any recovery dispatch.
    inspect="import psutil,json;print(json.dumps([{'pid':p.pid,'ppid':p.ppid(),'name':p.name(),'exe':p.exe(),'cmd_sha256':__import__('hashlib').sha256(repr(p.cmdline()).encode()).hexdigest(),'preflight_marker':any('training_runtime_preflight_root_' in x for x in p.cmdline())} for p in psutil.process_iter() if 'python' in p.name().lower() and p.pid not in (__import__('os').getpid(),__import__('os').getppid())]))"
    preflight=run_capture(build('.vscode/sftp.3080.json',r'C:\Users\zh\project\MASKCO-Main',cpu_python,'-B',['-c',inspect],'windows'),30,'.vscode/sftp.3080.json')
    write(OUT/'PROCESS_PREFLIGHT.json',preflight)
    if preflight['exit_code']!=0:raise RuntimeError('Actual process check failed')
    processes=json.loads(preflight['stdout'])
    # A non-inspection Python process must be resolved before retrying an uncertain dispatch.
    if processes:raise RuntimeError('Review actual Python processes before recovering census')
    program=OUT/'REMOTE_CENSUS_PROGRAM.py';program.write_text(REMOTE,encoding='utf-8')
    reg={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'REGISTERED_ENGINEERING_IMPORT_ONLY',
        'server':'3080','cpu':[15],'python':cpu_python,'runtime_program_sha256':sha(program),
        'old_runtime_program_sha256':sha(OLD/'REMOTE_CENSUS_PROGRAM.py'),'old_failure_sha256':sha(failure),
        'recovery':'same import-only CPU program, short exclusive PY filename transport; no scientific changes',
        'new_scientific_data':False,'training_performed':False,'shared_MD_uploaded':False,'launcher_sha256':sha(__file__)}
    write(OUT/'REGISTRATION.json',reg)
    host,user,password=_cfg('.vscode/sftp.3080.json');client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host,port=22,username=user,password=password,timeout=30)
        sftp=client.open_sftp()
        # mkdir must fail if a previous run exists; never overwrite an uncertain job.
        sftp.mkdir(REMOTE_DIR)
        with sftp.open(REMOTE_DIR+'/runtime_census.py','wx') as f:f.write(program.read_bytes())
        with sftp.open(REMOTE_DIR+'/ENGINEERING_REGISTRATION.json','wx') as f:f.write((OUT/'REGISTRATION.json').read_bytes())
        with sftp.open(REMOTE_DIR+'/runtime_census.py','rb') as f:uploaded=hashlib.sha256(f.read()).hexdigest()
        write(OUT/'UPLOAD_IDENTITY.json',{'remote_program_sha256':uploaded,'expected_sha256':sha(program),'PY_JSON_only':True})
        if uploaded!=sha(program):raise RuntimeError('Exclusive PY transport changed bytes')
        sftp.close()
    finally:client.close()
    started=time.perf_counter()
    raw=run_capture(build('.vscode/sftp.3080.json',r'C:\Users\zh\project\MASKCO-Main',cpu_python,'-B',
                         [REMOTE_DIR+'/runtime_census.py'],'windows'),180,'.vscode/sftp.3080.json')
    write(OUT/'REMOTE_CAPTURE.json',raw)
    if raw['exit_code']!=0:raise RuntimeError('Import census failed; original response preserved')
    result=json.loads(raw['stdout']);result['registration_sha256']=sha(OUT/'REGISTRATION.json')
    write(OUT/'RESULT.json',result)
    print(json.dumps({'files':len(result['source_start']),'changed':len(result['changed_or_missing_files']),
                     'new_files':len(result['new_loaded_files']),'packages':result['packages'],'runtime_qualified':result['pure_import_cpu_runtime_qualified'],
                     'wall_s':time.perf_counter()-started}))
    return int(not result['pure_import_cpu_runtime_qualified'])
if __name__=='__main__':raise SystemExit(main())
