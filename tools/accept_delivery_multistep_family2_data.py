"""Only after the unique collector exits, download raw data and launch one new audit."""
import hashlib,json,posixpath,sys
from datetime import datetime,timezone
from pathlib import Path
import paramiko
from _srv import _cfg
from _srv_exec_v2 import build,run_capture
from remote_actual_fullstep_3080_v2 import LAUNCHER,STATUS
from deploy_delivery_multistep_family2 import OUT as SOURCE,REL as SOURCE_REL,WIN,PY,CONFIG
OUT=SOURCE.parent/'multistep_insertable_family2_data_audit_root_v1';REL='research/review_3080_20261010/multistep_insertable_family2_data_audit_root_v1'
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def create(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    if OUT.exists():raise FileExistsError('Audit exists; inspect actual process/artifacts, never relaunch')
    state=run_capture(build(CONFIG,WIN,PY,'-B',[SOURCE_REL+'/status.py'],'windows'),30,CONFIG)
    if state['exit_code']!=0:raise RuntimeError('Actual original process lookup failed')
    actual=json.loads(state['stdout'])
    if not actual['finished']:raise RuntimeError('Unique original collector still active; continue waiting without launch')
    if actual['run_exit']['exit_code'] not in (0,1):raise RuntimeError('Original phase interrupted; preserve and inspect before diagnostic dispatch')
    host,user,password=_cfg(CONFIG);c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect(host,port=22,username=user,password=password,timeout=30)
    names=['RESEARCH_REGISTRATION.json','EXTERNAL_REGISTRATION_BINDING.json','collector.py','retained_gls.py','QUALIFIED_SERVER_REFERENCE.json','STATES.jsonl','REQUESTS_STARTED.jsonl','SOURCE_START.json','SOURCE_END.json','RESULT.json','RUN_EXIT.json','WORKER_IDENTITY.json']
    try:
        s=c.open_sftp();remote_root=json.loads(Path(CONFIG).read_text())['remotePath'].rstrip('/');src=posixpath.join(remote_root,SOURCE_REL)
        # Verify original finished supervisor/worker no longer exists before dependent dispatch.
        with s.open(src+'/WORKER_IDENTITY.json','rb') as f:worker_identity=json.loads(f.read())
        with s.open(src+'/worker.pid','rb') as f:worker_pid=int(f.read())
        checkcode='import psutil,json; ids='+repr([worker_identity['supervisor_pid'],worker_pid])+"; print(json.dumps({'alive':[i for i in ids if psutil.pid_exists(i)]}))"
        check=run_capture(build(CONFIG,WIN,PY,'-B',['-c',checkcode],'windows'),30,CONFIG)
        if check['exit_code']!=0 or json.loads(check['stdout'])['alive']:raise RuntimeError('Original real process has not exited; wait')
        download=SOURCE/'downloaded';download.mkdir(exist_ok=False)
        for n in names:
            p=download/n;size=s.stat(src+'/'+n).st_size
            # Bounded readv batches avoid a single long SFTP request; original run never repeats.
            with s.open(src+'/'+n,'rb') as f,p.open('xb') as dest:
                at=0
                while at<size:
                    chunks=[(x,min(65536,size-x)) for x in range(at,min(size,at+16*65536),65536)]
                    blocks=list(f.readv(chunks))
                    if len(blocks)!=len(chunks) or any(len(b)!=length for b,(_,length) in zip(blocks,chunks)):raise IOError('Incomplete bounded download; preserve partial original bytes')
                    for block in blocks:dest.write(block)
                    at+=sum(length for _,length in chunks)
            if s.stat(src+'/'+n).st_size!=size:raise ValueError('Finished raw artifact changed during read')
        producer=json.loads((download/'RESULT.json').read_text(encoding='utf-8'))
        if producer['counts']['states']!=256:raise ValueError('Incomplete scientific data; no complete-data audit launch')
        OUT.mkdir(exist_ok=False);program=(ROOT/'tools/audit_delivery_multistep_family2_data.py').read_bytes();compile(program,'audit.py','exec')
        binding=json.loads((download/'EXTERNAL_REGISTRATION_BINDING.json').read_text());inputs={n:sha(download/n) for n in names}
        reg={'created_utc':datetime.now(timezone.utc).isoformat(),'program_sha256':hashlib.sha256(program).hexdigest(),'inputs':inputs,'research_registration_sha256':binding['registration_sha256'],'scope':'independent full256 raw replay; diagnostic when the original declared barrier fails; never turn original FAIL into PASS','producer_original_actual_run_exit':actual['run_exit'],'max_s':1200,'cpu':[15],'shared_physical_evaluator':True,'automatic_training':False}
        create(OUT/'AUDIT_REGISTRATION.json',reg);create(OUT/'ORIGINAL_FINISHED_PROCESS_CHECK.json',check);create(download/'DOWNLOAD_MANIFEST.json',{'files':inputs,'source_run_exit':actual['run_exit'],'all_originals_preserved':True})
        supervisor="""import json,os,pathlib,subprocess,sys,time,traceback
base=pathlib.Path(__file__).resolve().parent
argv=[sys.executable,'-B',str(base/'audit.py')];start=time.time();code=1;failure=None
with (base/'WORKER_IDENTITY.json').open('x') as f:json.dump({'supervisor_pid':os.getpid(),'argv':argv,'started_epoch':start,'maximum_seconds':1200},f)
try:
 with (base/'run.stdout.log').open('xb') as out,(base/'run.stderr.log').open('xb') as err:
  worker=subprocess.Popen(argv,cwd=base,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
  (base/'worker.pid').write_text(str(worker.pid))
  try:code=worker.wait(timeout=1200)
  except subprocess.TimeoutExpired:worker.kill();worker.wait();code=124;failure='bounded_independent_audit_timeout'
except Exception:failure=traceback.format_exc()
with (base/'RUN_EXIT.json').open('x') as f:json.dump({'exit_code':code,'failure':failure,'wall_s':time.time()-start,'supervisor_pid':os.getpid(),'optimizer_steps':0},f)
"""
        files={'audit.py':program,'AUDIT_REGISTRATION.json':(OUT/'AUDIT_REGISTRATION.json').read_bytes(),'supervisor.py':supervisor.encode(),'launcher.py':LAUNCHER.replace("'previous_attempt':'v1 interrupted before supervisor body; retained'","'phase':'family2_data_audit'").encode(),'status.py':STATUS.encode()}
        target=posixpath.join(remote_root,REL);s.mkdir(target);hashes={}
        for n,b in files.items():
            if n!='AUDIT_REGISTRATION.json':(OUT/n).write_bytes(b)
            with s.open(target+'/'+n,'wx') as f:f.write(b)
            with s.open(target+'/'+n,'rb') as f:hashes[n]=hashlib.sha256(f.read()).hexdigest()==hashlib.sha256(b).hexdigest()
        s.close();create(OUT/'UPLOAD_ACCEPTANCE.json',hashes)
        if not all(hashes.values()):raise ValueError('Actual audit upload mismatch')
    finally:c.close()
    launch=run_capture(build(CONFIG,WIN,PY,'-B',[REL+'/launcher.py'],'windows'),60,CONFIG);create(OUT/'LAUNCH_RESPONSE.json',launch)
    if launch['exit_code']!=0:raise RuntimeError('Uncertain audit launch; verify actual tree before recovery')
    print(json.dumps({'launch_exit':launch['exit_code'],'downloaded_files':len(names),'audit_only':True,'training_started':False}))
if __name__=='__main__':main()
