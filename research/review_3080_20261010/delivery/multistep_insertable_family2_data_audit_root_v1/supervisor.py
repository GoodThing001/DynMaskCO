import json,os,pathlib,subprocess,sys,time,traceback
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
