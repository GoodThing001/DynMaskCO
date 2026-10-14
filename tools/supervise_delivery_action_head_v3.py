"""Exclusive bounded supervisor, no automatic retries after any failure."""
import os,json,subprocess,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent
from action_head_runtime_v1 import sha,save

def main():
    import psutil
    psutil.Process().cpu_affinity([15])
    reg=json.loads((BASE/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
    if sha(__file__)!=reg['package_files'][Path(__file__).name]:raise ValueError('supervisor identity')
    save(BASE,'SUPERVISOR_STARTED.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),started=time.time(),registration_sha256=sha(BASE/'EXECUTION_REGISTRATION.json')))
    for name,receipt in [('run_delivery_actual_action_head_v2.py','RUN_EXIT.json'),('audit_delivery_actual_action_head_v2.py','AUDIT_RUN_EXIT.json')]:
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        began=time.perf_counter()
        with (BASE/(name+'.stdout.txt')).open('xb') as out,(BASE/(name+'.stderr.txt')).open('xb') as err:
            p=subprocess.Popen([sys.executable,'-B',str(BASE/name)],cwd=str(BASE),env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
            save(BASE,name+'.PROCESS.json',dict(pid=p.pid,ppid=os.getpid(),create_time=psutil.Process(p.pid).create_time(),argv=[sys.executable,'-B',str(BASE/name)]))
            timed_out=False
            try:code=p.wait(timeout=reg['stage_timeout_seconds'])
            except subprocess.TimeoutExpired:
                timed_out=True;p.kill();code=p.wait(timeout=reg['reap_timeout_seconds'])
        save(BASE,receipt,dict(exit_code=code,timed_out=timed_out,wall_seconds=time.perf_counter()-began,natural_exit=not timed_out,worker_pid=p.pid))
        if code!=0:return code
    return 0

if __name__=='__main__':
    started=time.time();code=1
    try:code=main()
    except Exception:
        save(BASE,'SUPERVISOR_FAILURE.json',dict(traceback=traceback.format_exc()))
    save(BASE,'CHAIN_EXIT.json',dict(exit_code=code,supervisor_pid=os.getpid(),wall_seconds=time.time()-started,optimizer_steps=0,scientific_family3_started=False))
    raise SystemExit(code)
