"""Read-only inventory for newly authorized corrective work."""
from pathlib import Path
import sys,json,shlex
E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(E.parent/'tools'))
from _srv import run
code=r'''
import os,json,subprocess,shutil,datetime,importlib.util
from pathlib import Path
root=Path('/home/hzeng/project/MASKCO-Main')
def command(args):
 p=subprocess.run(args,text=True,capture_output=True,timeout=20)
 return {'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
out={'checked_at':str(datetime.datetime.now()),'cpu_count':os.cpu_count(),
 'affinity':sorted(os.sched_getaffinity(0)),
 'disk_free_gb':shutil.disk_usage(root).free/(1024**3),
 'memory':Path('/proc/meminfo').read_text(),
 'gpu':command(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader']),
 'gpu_processes':command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']),
 'processes':command(['ps','-eo','user=,pid=,ppid=,psr=,pcpu=,etime=,args=']),
 'packages':{x:bool(importlib.util.find_spec(x)) for x in ['numpy','jax','flax','ortools','pytest','psutil']},
 'other_project_archives':[]}
for folder,dirs,names in os.walk(root,followlinks=False):
 dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','.venv','node_modules')]
 for name in names:
  if name.endswith(('.zip','.tar','.tar.gz','.tgz')) or name in ('repro_check.txt','mpre_trained.py'):
   p=Path(folder)/name
   if 'C-VRP_Cold-chainVehicleRoutingProblem' not in p.parts:out['other_project_archives'].append(str(p))
print(json.dumps(out))
'''
data=json.loads(run('/home/hzeng/miniconda3/envs/MASKCO_env/bin/python -c '+shlex.quote(code),timeout=120))
out=E/'results/three_line_corrections_20261009/SERVER_RESOURCES_AND_ARCHIVES.json'
with out.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
print(json.dumps({k:data[k] for k in ('checked_at','cpu_count','affinity','disk_free_gb','gpu','gpu_processes','packages','other_project_archives')},ensure_ascii=False))
