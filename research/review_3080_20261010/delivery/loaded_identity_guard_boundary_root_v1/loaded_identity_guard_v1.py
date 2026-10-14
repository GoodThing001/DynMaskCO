import sys,hashlib
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def collect_loaded(extra):
 import psutil
 paths={Path(sys.executable).resolve()};paths.update(Path(p).resolve() for p in extra);maps={}
 for name,m in list(sys.modules.items()):
  row={}
  for attr in ('__file__','__cached__'):
   raw=getattr(m,attr,None)
   if raw and Path(raw).is_file():p=Path(raw).resolve();paths.add(p);row[attr]=str(p)
  origin=getattr(getattr(m,'__spec__',None),'origin',None)
  if origin and origin not in ('built-in','frozen') and Path(origin).is_file():p=Path(origin).resolve();paths.add(p);row['spec_origin']=str(p)
  if row:maps[name]=row
 native={}
 for m in psutil.Process().memory_maps():
  if m.path and Path(m.path).is_file():
   p=Path(m.path).resolve();paths.add(p);native[str(p)]=True
 return dict(files={str(p):sha(p) for p in sorted(paths)},module_bindings=maps,native_files=sorted(native))
def guard(seal,baseline_bindings,actual,initial_params,actual_params):
 reasons=[]
 for p,h in actual['files'].items():
  if p not in seal['files']:reasons.append(dict(kind='unsealed_loaded_file',path=p))
  elif seal['files'][p]!=h:reasons.append(dict(kind='loaded_byte_changed',path=p))
 for n,b in baseline_bindings.items():
  if n in actual['module_bindings'] and actual['module_bindings'][n]!=b:reasons.append(dict(kind='existing_module_origin_rebound',module=n))
 if actual_params!=initial_params:reasons.append(dict(kind='live_parameter_identity_changed'))
 return dict(accepted=not reasons,reasons=reasons)
