"""Explicit loaded-only candidate; negative full-tree case cannot be erased."""
import ast,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1];OLD=R/'research/review_3080_20261010/delivery/parallel_identity_collection_root_v1';O=R/'research/review_3080_20261010/delivery/loaded_identity_guard_boundary_root_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,s):
 if n.endswith('.py'):ast.parse(s)
 (O/n).write_text(s,encoding='utf-8',newline='\n')
GUARD='''import sys,hashlib
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
'''
RUN='''import os
os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import ctypes,importlib.util,json,sys,time,traceback
from pathlib import Path
B=Path(__file__).resolve().parent
from action_head_runtime_v1 import sha,save,closure,delta
from loaded_identity_guard_v1 import collect_loaded,guard
from startup_native_identity_v1 import prime_native_identity
def import_probe(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
 import psutil
 native=prime_native_identity();psutil.Process().cpu_affinity([15]);reg=json.loads((B/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
 if psutil.virtual_memory().available<8*2**30:raise RuntimeError('8GiB barrier')
 from delivery_score_split_engine_v1 import InitialEngine
 engine=InitialEngine(B,reg,'pretrained_initial')
 probe=import_probe('guard_existing_probe',B/'loaded_probe.py')
 params=lambda:dict(backbone=engine.inventory(engine.model),head=engine.inventory(engine.head))
 initial=params();save(B,'INITIAL_LIVE_PARAMETERS.json',initial)
 extras=engine.extras;trees=engine.trees
 closure(extras,trees);seal=closure(extras,trees);save(B,'SOURCE_START.json',seal)
 mandatory=[B/n for n in reg['execution_source_names']]+[B/'EXECUTION_REGISTRATION.json']
 baseline=collect_loaded(mandatory);save(B,'BASELINE_LOADED_FILES.json',baseline)
 records=[];native_handle=None
 def observe(name,expected_loaded,expected_full,case_note):
  t0=time.perf_counter();actual=collect_loaded(mandatory);t1=time.perf_counter();actual_params=params();t2=time.perf_counter();decision=guard(seal,baseline['module_bindings'],actual,initial,actual_params);t3=time.perf_counter()
  reference=closure(extras,trees);t4=time.perf_counter();changes=delta(seal,reference)
  # Full-tree gate also checks origin and live parameters; files outside seal fail.
  full_accepted=not changes['changed'] and not changes['new_files'] and actual_params==initial and not any(x['kind']=='existing_module_origin_rebound' for x in decision['reasons'])
  file=name+'.json';save(B,file,dict(name=name,actual_loaded=actual,actual_params=actual_params,decision=decision,full_reference=reference,full_delta=changes,full_accepted=full_accepted,expected_loaded_accept=expected_loaded,expected_full_accept=expected_full,case_note=case_note,clock_hex=dict(loaded_start=t0.hex(),loaded_end=t1.hex(),parameter_end=t2.hex(),decision_end=t3.hex(),reference_end=t4.hex())))
  records.append(dict(name=name,file=file,sha256=sha(B/file),accepted=decision['accepted'],full_accepted=full_accepted,expected_loaded_accept=expected_loaded,expected_full_accept=expected_full,loaded_seconds=t1-t0,full_live_parameter_seconds=t2-t1,guard_decision_seconds=t3-t2,sequential_full_reference_seconds=t4-t3,actual_loaded_file_count=len(actual['files']),reason_kinds=[x['kind'] for x in decision['reasons']]))
 observe('01_normal',True,True,'Actual initial76M model and loaded_probe imports, no model forward.')
 m=import_probe('guard_new_presealed_probe',B/'presealed_new_probe.py');observe('02_new_presealed_module',True,True,'Real exec_module after full seal; source presealed.');sys.modules.pop('guard_new_presealed_probe')
 original_file=probe.__file__;probe.__file__=str(B/'presealed_rebound_probe.py');observe('03_origin_rebound',False,False,'Real already-loaded module.__file__ rebound to different presealed source; module.spec.origin retained.');probe.__file__=original_file
 old=engine.head.input.bias.value;engine.head.input.bias.value=old+engine.jnp.ones_like(old);observe('04_parameter_rebound',False,False,'Actual NNX Param.value rebinding in private engineering instance; no optimizer.');engine.head.input.bias.value=old
 new=B/'actual_unsealed_module.py';new.write_text('VALUE=7001\\n',encoding='utf-8');m=import_probe('guard_unsealed_module',new);observe('05_unsealed_module',False,False,'Real new source created after start and actually imported.');sys.modules.pop('guard_unsealed_module')
 # Preserve original registered bytes as separate ORIGINAL fixture, do not touch upstream.
 p=B/'loaded_probe.py';p.write_text('VALUE=42\\n',encoding='utf-8');observe('06_loaded_byte_changed',False,False,'Actual source file of still-loaded module changed; original bytes preserved in loaded_probe_original.py.');sys.modules.pop('guard_existing_probe')
 p=B/'unloaded_registered_probe.py';p.write_text('VALUE=1002\\n',encoding='utf-8');observe('07_unloaded_tree_changed',True,False,'Real registered but never-imported source changed. Loaded-only candidate misses original complete-tree gate; explicit necessary negative sufficiency boundary.')
 path=Path(os.environ['SystemRoot'])/'System32'/'winmm.dll'
 native_already_sealed=str(path.resolve()) in seal['files'];native_handle=ctypes.WinDLL(str(path))
 observe('08_actual_native_load',False,False,'Fixed Win32 actual winmm.dll load after start. If file was already sealed this requested unsealed-native boundary is unexercised, not relabeled PASS.')
 end=closure(extras,trees);save(B,'SOURCE_END.json',dict(**end,**delta(seal,end)));save(B,'FINAL_LIVE_PARAMETERS.json',params())
 result=dict(records=records,checks_match_prospective_cases=all(r['accepted']==r['expected_loaded_accept'] and r['full_accepted']==r['expected_full_accept'] for r in records),native_unsealed_case_exercised=not native_already_sealed,actual_full_tree_contract_sufficient=False,loaded_only_component_qualified=False,candidate_installed=False,source_fault_changes_expected_and_preserved=True,initial_and_final_live_params_equal=initial==params(),model_forwards=0,C0_solver_calls=0,optimizer_updates=0,scientific_family3_started=False,stable_online_gain=False)
 save(B,'RESULT.json',result);return 0 if result['checks_match_prospective_cases'] and result['native_unsealed_case_exercised'] else 1
if __name__=='__main__':
 code=1
 try:code=main()
 except BaseException:save(B,'FAILURE.json',dict(traceback=traceback.format_exc()))
 raise SystemExit(code)
'''

def main():
 O.mkdir(parents=True,exist_ok=False)
 for n in ('action_head_runtime_v1.py','startup_native_identity_v1.py','timed_evidence_io_v1.py','delivery_score_split_engine_v1.py','features_objective.py','delivery_actual_action_head_v1.py','delivery_actual_score_chain_v1.py','delivery_feasible_action_options_v1.py','delivery_retained_edge_action_guard_v1.py'):write(n,(OLD/n).read_text(encoding='utf-8'))
 write('loaded_identity_guard_v1.py',GUARD);write('run_loaded_guard_v1.py',RUN)
 for n,value in [('loaded_probe.py',41),('loaded_probe_original.py',41),('presealed_new_probe.py',51),('presealed_rebound_probe.py',61),('unloaded_registered_probe.py',1001),('unloaded_registered_probe_original.py',1001)]:write(n,'VALUE=%d\n'%value)
 # Independent audit source prepared separately before execution registration.
 print(json.dumps(dict(root=str(O),source_prepared=True,execution_authorized=False)))
if __name__=='__main__':main()
