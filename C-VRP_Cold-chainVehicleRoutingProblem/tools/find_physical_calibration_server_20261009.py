"""Bounded read-only inventory of project measurement tables; no private config reads."""
from pathlib import Path
import sys,json,shlex
E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E.parent/'tools'))
from _srv import run
code=r'''
import os,json,datetime
from pathlib import Path
root=Path('/home/hzeng/project/MASKCO-Main')
files=[];errors=[];count=0
for folder,dirs,names in os.walk(root,followlinks=False,onerror=lambda e:errors.append(str(e))):
 dirs[:]=[d for d in dirs if d not in ('.git','.vscode','.idea','__pycache__','node_modules')]
 for name in names:
  count+=1;p=Path(folder)/name
  if p.suffix.lower() in ('.csv','.tsv','.xlsx','.xls','.parquet','.h5','.hdf5'):
   row={'path':str(p),'bytes':p.stat().st_size,'kind':'unclassified_table'}
   if p.suffix.lower() in ('.csv','.tsv'):
    try:
     with p.open(errors='replace') as f:row['header']=f.readline()[:1000]
    except OSError as e:row['error']=str(e)
   files.append(row)
print(json.dumps({'checked_at':str(datetime.datetime.now()),'scope':str(root),
 'walk_follow_symlinks':False,'files_seen':count,'measurement_table_candidates':files,'errors':errors,
 'scope_limit':'No personal directories, external mounted stores or inaccessible symlink targets searched.'}))
'''
data=json.loads(run('/home/hzeng/miniconda3/envs/MASKCO_env/bin/python -c '+shlex.quote(code),timeout=120))
with (E/'results/three_line_corrections_20261009/PHYSICAL_CALIBRATION_DATA_SEARCH.json').open('x',encoding='utf-8') as f:
 json.dump(data,f,indent=2,ensure_ascii=False)
print(json.dumps(data,ensure_ascii=False))
