"""Refine independent witness scope before first upload; retain original bytes."""
import ast,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'research/review_3080_20261010/delivery/loaded_identity_guard_boundary_root_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if (O/'DEPLOY_ATTEMPT.json').exists():raise RuntimeError('Already deployed; original tree immutable')
 reg_path=O/'EXECUTION_REGISTRATION.json';old=reg_path.read_bytes()
 with (O/'EXECUTION_REGISTRATION_PREEXECUTION_R1.json').open('xb') as f:f.write(old)
 src=(O/'audit_loaded_guard_v1.py').read_text(encoding='utf-8')
 insert="""for record in result['records']:
  snapshot=read(record['file'])
  for p,h in snapshot['full_reference']['files'].items():
   witness=B/('attempt_loaded_change.py' if Path(p).name=='loaded_probe.py' and record['name']=='06_loaded_byte_changed' else 'attempt_unloaded_change.py' if Path(p).name=='unloaded_registered_probe.py' and record['name']=='07_unloaded_tree_changed' else p)
   check('all_full_reference_byte_witness:'+record['name']+':'+p,sha(witness)==h)"""
 src=src.replace("for record in result['records']:",insert);ast.parse(src)
 (O/'audit_loaded_guard_v2.py').write_text(src,encoding='utf-8',newline='\n')
 src=(O/'supervise_loaded_guard_v1.py').read_text(encoding='utf-8').replace('audit_loaded_guard_v1.py','audit_loaded_guard_v2.py')
 (O/'supervise_loaded_guard_v2.py').write_text(src,encoding='utf-8',newline='\n')
 src=(O/'loaded_guard_remote_launcher_v1.py').read_text(encoding='utf-8').replace('supervise_loaded_guard_v1.py','supervise_loaded_guard_v2.py')
 (O/'loaded_guard_remote_launcher_v2.py').write_text(src,encoding='utf-8',newline='\n')
 t=R/'tools/transport_delivery_loaded_guard_v1.py';src=t.read_text(encoding='utf-8').replace('loaded_guard_remote_launcher_v1.py','loaded_guard_remote_launcher_v2.py')
 with (O/'TRANSPORT_SOURCE_PREEXECUTION_R1.py').open('x',encoding='utf-8') as f:f.write(t.read_text(encoding='utf-8'))
 t.write_text(src,encoding='utf-8',newline='\n');ast.parse(src)
 reg=json.loads(old);reg['pre_execution_original_registration_sha256']=hashlib.sha256(old).hexdigest()
 reg['pre_execution_r2']='Independent full byte witnesses for every original file/case before any upload/execution. Frozen first audit source and registration retained; no experiment retry.'
 reg['package_files']={p.name:sha(p) for p in O.iterdir() if p.suffix in ('.py','.json') and p.name not in ('EXECUTION_REGISTRATION.json','LOCAL_EXECUTION_BINDING.json')}
 reg['execution_source_names']=[n for n in reg['package_files'] if 'probe' not in n]
 reg['transport_tool_sha256']=sha(t)
 reg_path.write_text(json.dumps(reg,indent=2),encoding='utf-8',newline='\n')
 with (O/'LOCAL_EXECUTION_BINDING_R1.json').open('xb') as f:f.write((O/'LOCAL_EXECUTION_BINDING.json').read_bytes())
 (O/'LOCAL_EXECUTION_BINDING.json').write_text(json.dumps(dict(registration_sha256=sha(reg_path)),indent=2),encoding='utf-8',newline='\n')
 print(json.dumps(dict(registration_sha256=sha(reg_path),files=len(reg['package_files']),no_prior_execution=True)))
if __name__=='__main__':main()
