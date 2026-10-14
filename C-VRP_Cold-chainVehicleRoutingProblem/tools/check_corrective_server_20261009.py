"""Read-only status, recursive own-process inventory and expanded historical search."""
from pathlib import Path
import sys,json,shlex,argparse
E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E.parent/'tools'))
from _srv import run
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--version',choices=['v2','v3'],default='v2');args=p.parse_args()
code=r'''
import json,os,hashlib,subprocess,datetime,importlib.metadata
from pathlib import Path
R=Path('/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/correction_validation_1009_v2')
root=R.parents[2]
out={'checked_at':str(datetime.datetime.now()),'reports':{},'training_candidates':[],
     'environment':{'python':__import__('sys').version,'packages':{}}}
for n in ('numpy','jax','flax','ortools','matplotlib','psutil'):
 try:out['environment']['packages'][n]=importlib.metadata.version(n)
 except importlib.metadata.PackageNotFoundError:pass
for name in ('pipeline.exit','pipeline_failure.json','engineering.exit.json','fixed_state.exit.json',
             'headroom.exit.json','figures.exit.json','QA_RESULT.json'):
 p=R/name
 if p.is_file():out['reports'][name]=p.read_text()
for p in R.glob('*.log'):
 out['reports'][p.name]=p.read_text(errors='replace')[-1600:]
rows=[]
for line in subprocess.check_output(['ps','-eo','pid=,ppid=,psr=,pcpu=,etime=,args='],text=True).splitlines():
 a=line.strip().split(None,5)
 if len(a)==6:rows.append({'pid':int(a[0]),'ppid':int(a[1]),'cpu':a[2],'pcpu':a[3],'elapsed':a[4],'args':a[5]})
parents=set()
for x in rows:
 try:
  argv=Path('/proc/'+str(x['pid'])+'/cmdline').read_bytes().split(b'\x00')
  if Path('/proc/'+str(x['pid'])+'/cwd').resolve()==R and len(argv)>1 and argv[1]==b'runner.py':parents.add(x['pid'])
 except OSError:pass
selected=set(parents)
while True:
 children={x['pid'] for x in rows if x['ppid'] in selected}
 if children<=selected:break
 selected|=children
out['process_tree']=[x for x in rows if x['pid'] in selected]
for folder,dirs,files in os.walk(root,followlinks=False):
 dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','node_modules')]
 for name in files:
  if name.startswith('mpre_trained.py'):
   p=Path(folder)/name
   try:out['training_candidates'].append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
   except OSError as e:out['training_candidates'].append({'path':str(p),'error':str(e)})
print(json.dumps(out))
'''
code=code.replace('correction_validation_1009_v2','correction_validation_1009_'+args.version)
data=json.loads(run('/home/hzeng/miniconda3/envs/MASKCO_env/bin/python -c '+shlex.quote(code),timeout=120))
target=E/'results/three_line_corrections_20261009'/args.output
if target.parent!=(E/'results/three_line_corrections_20261009'):raise ValueError('Output outside audit folder')
with target.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,ensure_ascii=False)
print(json.dumps(data,ensure_ascii=False))
