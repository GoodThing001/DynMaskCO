"""Read-only remote discovery for the independently recorded RRP recovery audit."""
from pathlib import Path
import importlib.util
import hashlib
import json
import shlex
import sys

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('static_rrp_remote_readonly', ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem/tools/static_rrp_remote.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
client, settings = module.connect(ROOT / '.vscode/sftp.json')
try:
    sftp = client.open_sftp()
    remote = settings['remotePath'].rstrip('/')
    entries = sftp.listdir_attr(remote)
    top = [e.filename for e in entries if not any(t in e.filename.lower() for t in ('private', 'credential', 'secret', '.git', '.vscode'))]
    code = '''import os,json,re,time
root=ROOT_VALUE
start=time.time()
skip={'.git','.hg','.svn','.vscode','.cursor','.claude','__pycache__','node_modules','.venv','venv','env','cache','.cache','wandb','checkpoints','private'}
out={'root':root,'max_depth':12,'max_entries':180000,'max_seconds':22,'excluded':sorted(skip),'excluded_recent_runs':['run01','run02','run03','run04','run05'],'visited_entries':0,'visited_directories':0,'matches':[],'pruned_depth':0,'pruned_recent_runs':[],'errors':[]}
for current,dirs,files in os.walk(root,followlinks=False,onerror=lambda e:out['errors'].append({'type':type(e).__name__})):
 out['visited_directories']+=1
 depth=os.path.relpath(current,root).count(os.sep)+(current!=root)
 kept=[]
 for d in dirs:
  p=os.path.join(current,d)
  if d.lower() in skip or any(t in d.lower() for t in ('private','credential','secret')) or os.path.islink(p): continue
  if d in out['excluded_recent_runs'] and 'static_rrp_20261020' in p: out['pruned_recent_runs'].append(p); continue
  if depth>=out['max_depth']: out['pruned_depth']+=1;continue
  kept.append(d)
 dirs[:]=kept
 for name in dirs+files:
  out['visited_entries']+=1
  rel=os.path.relpath(os.path.join(current,name),root)
  direct=bool(re.search(r'rrp|refrigerated|energies.?13.?6214|sol-D1-n9-np32',name,re.I))
  in_rrp=bool(re.search(r'rrp|refrigerated',os.path.relpath(current,root),re.I))
  if direct or (in_rrp and name.lower().endswith(('.json','.txt','.cpp','.h','.mzn','.py','.md','.zip'))):
   p=os.path.join(current,name)
   if not os.path.islink(p): out['matches'].append({'path':p,'directory':os.path.isdir(p),'size':os.path.getsize(p) if os.path.isfile(p) else None})
 if out['visited_entries']>=out['max_entries'] or time.time()-start>out['max_seconds']:
  out['stopped_early']=True; break
else: out['stopped_early']=False
out['elapsed_seconds']=round(time.time()-start,3)
print(json.dumps(out))
'''.replace('ROOT_VALUE',repr(remote))
    result=json.loads(module.execute(client,module.PYTHON+' -B -c '+shlex.quote(code)))
    result['top_level']=top
    reference_relative='C-VRP_Cold-chainVehicleRoutingProblem/data/public_sources_20261020/refrigerated_routing_rrp/Solutions/D1/sol-D1-n9-np32.txt'
    local_bytes=(ROOT/reference_relative).read_bytes()
    local_sha=hashlib.sha256(local_bytes).hexdigest()
    result['known_uploaded_copy_checks']=[]
    for label in ('run01','run02','run03','run04','run05'):
        path=remote+'/C-VRP_Cold-chainVehicleRoutingProblem/research/static_rrp_20261020/'+label+'/workspace/'+reference_relative
        try:
            with sftp.open(path,'rb') as source:
                raw=source.read()
            digest=hashlib.sha256(raw).hexdigest()
            result['known_uploaded_copy_checks'].append({'path':path,'bytes':len(raw),'sha256':digest,'matches_local':digest==local_sha,'independent_recovery_source':False})
        except FileNotFoundError:
            result['known_uploaded_copy_checks'].append({'path':path,'missing':True,'independent_recovery_source':False})
    target=Path(__file__).parent/'linux_discovery.json'
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'evidence':str(target),'visited_entries':result['visited_entries'],'stopped_early':result['stopped_early'],'matches':result['matches']},ensure_ascii=False))
finally:
    client.close()
