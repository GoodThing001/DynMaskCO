"""Seal source-observed cases and independent audit before first execution."""
import ast,json,hashlib
from datetime import datetime,timezone
from pathlib import Path
R=Path(__file__).resolve().parents[1];OLD=R/'research/review_3080_20261010/delivery/parallel_identity_collection_root_v1';O=R/'research/review_3080_20261010/delivery/loaded_identity_guard_boundary_root_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,s):
 if n.endswith('.py'):ast.parse(s)
 (O/n).write_text(s,encoding='utf-8',newline='\n')

AUDIT='''"""Saved-only exact file/map/case reference plus independent full model initialization."""
import os
os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import copy,gc,json,hashlib,sys,traceback
from pathlib import Path
B=Path(__file__).resolve().parent
from action_head_runtime_v1 import sha,save,closure,delta
from startup_native_identity_v1 import prime_native_identity
def read(n):return json.loads((B/n).read_text(encoding='utf-8'))
def main():
 import psutil,numpy as np,jax,jax.numpy as jnp
 from flax import nnx
 native=prime_native_identity();psutil.Process().cpu_affinity([15]);reg=read('EXECUTION_REGISTRATION.json');result=read('RESULT.json')
 if psutil.virtual_memory().available<8*2**30:raise RuntimeError('Independent8GiB barrier')
 old=Path(reg['old_remote_root']);up=old/'upstream';ext=old/'extension'
 for p in (up,old/'actual_checkpoint_engineering_v1',ext/'scripts/delivery',ext/'scripts/coldchain'):sys.path.insert(0,str(p))
 from inspect_checkpoint import load_trusted
 saved,h=load_trusted(Path(reg['checkpoint']),up)
 if h!=reg['checkpoint_sha256']:raise ValueError('Original checkpoint identity')
 cfg=copy.deepcopy(saved['model_config']);cfg.dtype='float32';cfg.encoder_input_dim=10;cfg.rngs=888
 params=copy.deepcopy(saved['params']);k=np.asarray(params['init_proj']['kernel'].value);params['init_proj']['kernel'].value=jnp.asarray(np.concatenate([k,np.zeros((7,512),k.dtype)],axis=0));model=cfg.construct_model();nnx.update(model,params);del params,saved;gc.collect()
 rng=nnx.Rngs(20261012);a=nnx.Linear(3075,64,dtype=jnp.float32,param_dtype=jnp.float32,rngs=rng);b=nnx.Linear(64,1,dtype=jnp.float32,param_dtype=jnp.float32,rngs=rng)
 def inventory(state):return {str(k):dict(shape=list(np.asarray(v.value).shape),dtype=str(np.asarray(v.value).dtype),sha256=hashlib.sha256(np.asarray(v.value).tobytes()).hexdigest()) for k,v in state.flat_state()}
 head={str((layer,part)):dict(shape=list(np.asarray(v).shape),dtype=str(np.asarray(v).dtype),sha256=hashlib.sha256(np.asarray(v).tobytes()).hexdigest()) for layer,part,v in [('input','kernel',a.kernel.value),('input','bias',a.bias.value),('output','kernel',b.kernel.value),('output','bias',b.bias.value)]}
 reference_params=dict(backbone=inventory(nnx.state(model,nnx.Param)),head=head)
 extras=[B/n for n in reg['package_files']]+[B/n for n in ('EXECUTION_REGISTRATION.json','RESULT.json','SOURCE_START.json','SOURCE_END.json','INITIAL_LIVE_PARAMETERS.json','FINAL_LIVE_PARAMETERS.json','BASELINE_LOADED_FILES.json')]+[B/r['file'] for r in result['records']]
 trees=[up,ext/'scripts/delivery',ext/'scripts/coldchain'];closure(extras,trees);s=closure(extras,trees);save(B,'AUDIT_SOURCE_START.json',s)
 checks=[]
 def check(n,v):checks.append(dict(name=n,ok=bool(v)))
 seal=read('SOURCE_START.json');initial=read('INITIAL_LIVE_PARAMETERS.json');base=read('BASELINE_LOADED_FILES.json')
 for domain,values in reference_params.items():
  for k,v in values.items():check('independent_initial_live:'+domain+':'+k,initial[domain].get(k)==v);check('independent_final_live:'+domain+':'+k,read('FINAL_LIVE_PARAMETERS.json')[domain].get(k)==v)
 for record in result['records']:
  case=read(record['file']);check('case_original_sha:'+record['name'],sha(B/record['file'])==record['sha256'])
  actual=case['actual_loaded'];reasons=[]
  for p,h in actual['files'].items():
   if p not in seal['files']:reasons.append(('unsealed_loaded_file',p))
   elif seal['files'][p]!=h:reasons.append(('loaded_byte_changed',p))
   check('actual_loaded_full_reference_sha:'+record['name']+':'+p,case['full_reference']['files'].get(p)==h)
   # Mutation-source bytes are archived separately, never fabricated later.
   witness=B/('attempt_loaded_change.py' if Path(p).name=='loaded_probe.py' and record['name']=='06_loaded_byte_changed' else 'attempt_unloaded_change.py' if Path(p).name=='unloaded_registered_probe.py' and record['name']=='07_unloaded_tree_changed' else p)
   check('actual_loaded_byte_witness:'+record['name']+':'+p,sha(witness)==h)
  for n,binding in base['module_bindings'].items():
   if n not in actual['module_bindings'] or actual['module_bindings'][n]!=binding:reasons.append(('existing_module_origin_rebound',n))
  if case['actual_params']!=initial:reasons.append(('live_parameter_identity_changed',''))
  check('independent_guard_decision:'+record['name'],(not reasons)==case['decision']['accepted'])
  check('registered_expected_loaded:'+record['name'],case['decision']['accepted']==case['expected_loaded_accept'])
  full_changed=[p for p,h in seal['files'].items() if case['full_reference']['files'].get(p)!=h];full_new=sorted(set(case['full_reference']['files'])-set(seal['files']))
  check('independent_full_delta:'+record['name'],full_changed==case['full_delta']['changed'] and full_new==case['full_delta']['new_files'])
  full=not full_changed and not full_new and case['actual_params']==initial and not any(k=='existing_module_origin_rebound' for k,p in reasons)
  check('independent_full_gate:'+record['name'],full==case['full_accepted']);check('registered_expected_full:'+record['name'],full==case['expected_full_accept'])
  if record['name']=='04_parameter_rebound':
   altered=copy.deepcopy(initial);v=np.asarray(a.bias.value)+np.ones_like(np.asarray(a.bias.value));altered['head']["('input', 'bias')"]['sha256']=hashlib.sha256(v.tobytes()).hexdigest();check('actual_rebound_parameter_reference',altered==case['actual_params'])
  else:check('all_live_reference:'+record['name'],case['actual_params']==reference_params)
  check('clock_order:'+record['name'],list(map(float.fromhex,case['clock_hex'].values()))==sorted(map(float.fromhex,case['clock_hex'].values())))
 negative=read('07_unloaded_tree_changed.json');check('unloaded_sufficiency_counterexample_preserved',negative['decision']['accepted'] and not negative['full_accepted'] and any(Path(p).name=='unloaded_registered_probe.py' for p in negative['full_delta']['changed']))
 check('native_boundary_actually_exercised',result['native_unsealed_case_exercised']);end=closure(extras,trees);d=delta(s,end);save(B,'AUDIT_SOURCE_END.json',dict(**end,**d));check('independent_loaded_byte_closure',not d['changed'] and not d['new_files'])
 output=dict(check_count=len(checks),failure_count=sum(not c['ok'] for c in checks),checks=checks,all_file_and_live_reference_scope='Actual loaded maps/files compared to every saved original full-tree SHA and independently read original byte witnesses, all234+4 live leaves independent original initialization; zero guard or fault rerun.',original_full_tree_sufficiency=False,candidate_installed=False,stable_online_gain=False)
 save(B,'INDEPENDENT_LOADED_GUARD_AUDIT.json',output);return int(output['failure_count']>0)
if __name__=='__main__':
 code=1
 try:code=main()
 except BaseException:save(B,'AUDIT_FAILURE.json',dict(traceback=traceback.format_exc()))
 raise SystemExit(code)
'''
def main():
 guard=(O/'loaded_identity_guard_v1.py').read_text(encoding='utf-8');write('loaded_identity_guard_v2.py',guard.replace("if n in actual['module_bindings'] and actual['module_bindings'][n]!=b:","if n not in actual['module_bindings'] or actual['module_bindings'][n]!=b:"))
 run=(O/'run_loaded_guard_v1.py').read_text(encoding='utf-8').replace('from loaded_identity_guard_v1 import','from loaded_identity_guard_v2 import')
 run=run.replace("p=B/'loaded_probe.py';p.write_text('VALUE=42\\n',encoding='utf-8');observe('06_loaded_byte_changed',False,False,'Actual source file of still-loaded module changed; original bytes preserved in loaded_probe_original.py.');sys.modules.pop('guard_existing_probe')", "p=B/'loaded_probe.py';p.write_text('VALUE=42\\n',encoding='utf-8');(B/'attempt_loaded_change.py').write_bytes(p.read_bytes());observe('06_loaded_byte_changed',False,False,'Actual loaded source altered; mutation bytes preserved, then own mutable fixture restored.');p.write_bytes((B/'loaded_probe_original.py').read_bytes())")
 run=run.replace("p=B/'unloaded_registered_probe.py';p.write_text('VALUE=1002\\n',encoding='utf-8');observe('07_unloaded_tree_changed',True,False,'Real registered but never-imported source changed. Loaded-only candidate misses original complete-tree gate; explicit necessary negative sufficiency boundary.')", "p=B/'unloaded_registered_probe.py';p.write_text('VALUE=1002\\n',encoding='utf-8');(B/'attempt_unloaded_change.py').write_bytes(p.read_bytes());observe('07_unloaded_tree_changed',True,False,'Real never-loaded registered source changed; loaded-only misses original full-tree gate. Mutation archived, own fixture restored afterward.');p.write_bytes((B/'unloaded_registered_probe_original.py').read_bytes())")
 write('run_loaded_guard_v2.py',run);write('audit_loaded_guard_v1.py',AUDIT)
 mapping={'run_parallel_identity_v1.py':'run_loaded_guard_v2.py','audit_parallel_identity_v1.py':'audit_loaded_guard_v1.py','supervise_parallel_identity_v1.py':'supervise_loaded_guard_v1.py','identity_remote_launcher_v1.py':'loaded_guard_remote_launcher_v1.py','identity_remote_status_v1.py':'loaded_guard_remote_status_v1.py','INDEPENDENT_IDENTITY_AUDIT.json':'INDEPENDENT_LOADED_GUARD_AUDIT.json'}
 for old,new in [(k,v) for k,v in mapping.items() if k.endswith('.py') and k not in ('run_parallel_identity_v1.py','audit_parallel_identity_v1.py')]:
  raw=(OLD/old).read_text(encoding='utf-8')
  for k,v in mapping.items():raw=raw.replace(k,v)
  # Diagnostics may naturally exit1; always audit completed failure evidence once.
  if new=='supervise_loaded_guard_v1.py':raw=raw.replace('if code!=0:return code','if code!=0 and not (BASE/\'RESULT.json\').is_file():return code')
  write(new,raw)
 transport=R/'tools/transport_delivery_loaded_guard_v1.py';raw=(R/'tools/transport_delivery_parallel_identity_v1.py').read_text(encoding='utf-8').replace('parallel_identity_collection_root_v1','loaded_identity_guard_boundary_root_v1').replace('identity_remote_launcher_v1.py','loaded_guard_remote_launcher_v1.py').replace('identity_remote_status_v1.py','loaded_guard_remote_status_v1.py');transport.write_text(raw,encoding='utf-8',newline='\n');ast.parse(raw)
 prior=json.loads((OLD/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'));reg={k:prior[k] for k in ('old_remote_root','checkpoint','checkpoint_sha256','contract_hash','local_transport_helpers')}
 names=[p.name for p in O.iterdir() if p.suffix in ('.py','.json')]
 reg.update(created_utc=datetime.now(timezone.utc).isoformat(),status='PROSPECTIVE_LOADED_ONLY_GUARD_WITH_REAL_FULL_TREE_COUNTEREXAMPLE',execution_authorized=True,independent_timed_request_checker_sealed=True,package_files={n:sha(O/n) for n in names},transport_tool_sha256=sha(transport),execution_source_names=[n for n in names if 'probe' not in n],stage_timeout_seconds=180,reap_timeout_seconds=10,download_max_file_bytes=64*2**20,download_max_total_bytes=128*2**20,download_chunk_bytes=262144,
  method='Complete original source/runtime/native/parameter seal at day start, each candidate collects every actual sys.modules file/cached/spec_origin, interpreter and actual memory_maps file plus registered execution source SHA; all full live234+4 parameter SHA remains inside candidate guard. No metadata or pointer substitutes.',
  cases=['normal','actually_import_new_presealed_module','existing_module_file_origin_rebind','actual_NNX_parameter_rebind','actually_import_new_unsealed_module','actual_loaded_source_bytes_change','actual_unloaded_registered_source_bytes_change_candidate_accept_FULL_REJECT','actual_fixed_winmm_native_load_unsealed_or_unexercised_FAIL'],
  fixture_policy='Only this new directory owns mutable fixture files. Initial and mutated bytes retained separately; after observed mutation, own fixture restored for independent subsequent cases, no upstream/global/old source changed.',
  sufficiency_gate='Unloaded registered source change must be detected by original full-tree reference and missed by candidate: loaded-only cannot replace original all-file gate. In-memory monkeypatch/ephemeral imports removed before snapshot/arbitrary native writes and parent crash persistence not qualified. Native fixedcase alreadysealed means unexercised, no alternate DLL selection or retry.',
  cpu_affinity=[15],minimum_available_ram_gib=8,actual_model_parameter_count=76460353,model_forward_executions=0,C0_solver_calls=0,optimizer_updates=0,full_original_source_contract_qualified=False,candidate_installed=False,scientific_family3_started=False,full_request_budget_qualified=False,stable_online_gain=False)
 write('EXECUTION_REGISTRATION.json',json.dumps(reg,indent=2));write('LOCAL_EXECUTION_BINDING.json',json.dumps(dict(registration_sha256=sha(O/'EXECUTION_REGISTRATION.json')),indent=2))
 print(json.dumps(dict(root=str(O),registration_sha256=sha(O/'EXECUTION_REGISTRATION.json'),files=len(names))))
if __name__=='__main__':main()
