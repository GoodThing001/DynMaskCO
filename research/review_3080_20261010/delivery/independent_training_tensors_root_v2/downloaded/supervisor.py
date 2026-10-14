import hashlib,json,os,pathlib,psutil,subprocess,sys,time
base=pathlib.Path(__file__).resolve().parent
start=time.time();env=dict(os.environ,JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='-1',PYTHONIOENCODING='utf-8')
with (base/'stdout.log').open('xb') as stdout,(base/'stderr.log').open('xb') as stderr:
 p=subprocess.Popen([sys.executable,'-B',str(base/'verify_saved.py')],cwd=base,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
 psutil.Process(p.pid).cpu_affinity([15])
 try:code=p.wait(timeout=300)
 except subprocess.TimeoutExpired:
  children=psutil.Process(p.pid).children(recursive=True)
  for child in reversed(children):
   try:child.kill()
   except psutil.NoSuchProcess:pass
  p.kill();p.wait();psutil.wait_procs(children,timeout=10);code=124
with (base/'RUN_EXIT.json').open('x') as f:json.dump({'exit_code':code,'wall_seconds':time.time()-start,'own_worker_reaped':p.poll() is not None},f)
