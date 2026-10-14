import json,os,pathlib,subprocess,sys,time,traceback
base=pathlib.Path(__file__).resolve().parent
reg=json.loads((base/'RESEARCH_REGISTRATION.json').read_text(encoding='utf-8'))
argv=[sys.executable,'-B',str(base/'collector.py')]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
started=time.time();code=1;failure=None
with (base/'WORKER_IDENTITY.json').open('x') as f:json.dump({'supervisor_pid':os.getpid(),'argv':argv,'started_epoch':started,'max_s':1800,'phase':'fresh_teacher_only'},f)
try:
 with (base/'run.stdout.log').open('xb') as out,(base/'run.stderr.log').open('xb') as err:
  child=subprocess.Popen(argv,cwd=base,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
  (base/'worker.pid').write_text(str(child.pid))
  try:code=child.wait(timeout=1800)
  except subprocess.TimeoutExpired:child.kill();child.wait();code=124;failure='bounded_teacher_phase_timeout_no_retry'
except Exception:failure=traceback.format_exc()
with (base/'RUN_EXIT.json').open('x') as f:json.dump({'exit_code':code,'failure':failure,'wall_s':time.time()-started,'optimizer_steps':0,'supervisor_pid':os.getpid(),'training_launched':False},f,indent=2)
