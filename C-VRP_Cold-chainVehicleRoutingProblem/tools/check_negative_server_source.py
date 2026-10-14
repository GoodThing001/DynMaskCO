"""Read-only search within the authorized server extension; archive evidence locally."""
from pathlib import Path
import sys,json,shlex
E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(E.parent/'tools'))
from _srv import run
code=r'''
from pathlib import Path
import os,json,hashlib,zipfile,base64
e=Path('/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem')
target='e88c7d0d0d21c73b6335880a20111ee4d02b0a6693071872c74d217d34beb300'
report={'checked':[],'exact_hits':[],'matching_log_filenames':[],'errors':[]}
def check(label,data):
 h=hashlib.sha256(data).hexdigest();report['checked'].append({'path':label,'sha256':h})
 if h==target:report['exact_hits'].append({'path':label,'sha256':h,'source_base64':base64.b64encode(data).decode()})
for folder,dirs,names in os.walk(e,followlinks=False):
 dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','.venv','node_modules')]
 for name in names:
  p=Path(folder)/name
  if name=='mpre_trained.py':
   try:check(str(p),p.read_bytes())
   except OSError as exc:report['errors'].append({'path':str(p),'error':str(exc)})
  elif name=='repro_check.txt':report['matching_log_filenames'].append(str(p))
  elif name.endswith('.zip'):
   try:
    with zipfile.ZipFile(p) as z:
     for item in z.infolist():
      if item.filename.replace('\\','/').split('/')[-1]=='mpre_trained.py':check(str(p)+'::'+item.filename,z.read(item))
   except (OSError,zipfile.BadZipFile) as exc:report['errors'].append({'path':str(p),'error':str(exc)})
print(json.dumps(report))
'''
data=json.loads(run('/home/hzeng/miniconda3/envs/MASKCO_env/bin/python -c '+shlex.quote(code),timeout=120))
for hit in data['exact_hits']:
    import base64,hashlib
    raw=base64.b64decode(hit.pop('source_base64'))
    if hashlib.sha256(raw).hexdigest()!=hit['sha256']:raise ValueError('Transfer hash mismatch')
    p=E/'03_负结果线/核查工件/recovered_mpre_trained_e88c7d0d.py'
    if not p.exists():
        with p.open('xb') as f:f.write(raw)
data['remote_writes']=False
with (E/'results/three_line_corrections_20261009/NEGATIVE_SERVER_SOURCE_SEARCH.json').open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
print(json.dumps({'checked':len(data['checked']),'hits':len(data['exact_hits']),'log_filenames':data['matching_log_filenames'],'errors':data['errors']},ensure_ascii=False))
