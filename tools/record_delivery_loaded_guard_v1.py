"""Seal actual guard counterexample, never promote the old full-tree contract."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'research/review_3080_20261010/delivery/loaded_identity_guard_boundary_root_v1';D=O/'downloaded'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(n):return json.loads((D/n).read_text(encoding='utf-8'))
def source(prefix):
 s=read(prefix+'SOURCE_START.json');e=read(prefix+'SOURCE_END.json');adds={n:dict(path=p,start_file_sha256=s['files'].get(p),end_file_sha256=e['files'].get(p)) for n,p in e['module_files'].items() if s['module_files'].get(n)!=p}
 return dict(start_file_count=len(s['files']),end_file_count=len(e['files']),start_modulemap=len(s['module_files']),end_modulemap=len(e['module_files']),maps_equal=s['module_files']==e['module_files'],changed=e['changed'],new_files=e['new_files'],module_mapping_additions=adds)
def main():
 manifest=json.loads((O/'DOWNLOAD_MANIFEST.json').read_text(encoding='utf-8'))
 if not all((D/x['name']).stat().st_size==x['size'] and sha(D/x['name'])==x['sha256'] for x in manifest):raise RuntimeError('Original archive SHA')
 result=read('RESULT.json');audit=read('INDEPENDENT_LOADED_GUARD_AUDIT.json');cases=[]
 for r in result['records']:
  c=read(r['file'])
  cases.append(dict(**r,reason_details=c['decision']['reasons'],full_changed_files=c['full_delta']['changed'],full_new_files=c['full_delta']['new_files'],actual_modules_count=len(c['actual_loaded']['module_bindings']),actual_native_count=len(c['actual_loaded']['native_files']),case_note=c['case_note'],source_snapshot_sha256=r['sha256']))
 first=read('INITIAL_LIVE_PARAMETERS.json');last=read('FINAL_LIVE_PARAMETERS.json');normal=cases[0];negative=next(x for x in cases if x['name']=='07_unloaded_tree_changed')
 resource=json.loads(sorted(O.glob('RESOURCE_PREFLIGHT_*.json'))[0].read_text(encoding='utf-8'));observed=json.loads(resource['stdout'])
 obj=dict(status='RAW_GUARD_BOUNDARIES_VERIFIED_ORIGINAL_FULL_TREE_SUFFICIENCY_FAIL_NO_REPLACEMENT',registration_sha256=sha(O/'EXECUTION_REGISTRATION.json'),independent_checks=audit['check_count'],independent_failures=audit['failure_count'],
  raw_boundary_observation_qualified=result['checks_match_prospective_cases'] and result['native_unsealed_case_exercised'] and audit['failure_count']==0,
  original_RESULT_loaded_only_component_qualified=result['loaded_only_component_qualified'],original_full_tree_contract_sufficient=False,negative_unloaded_registered_source_actual=dict(candidate_accepts=negative['accepted'],original_full_gate_accepts=negative['full_accepted'],changed_original_paths=negative['full_changed_files'],mutation_byte_witness='attempt_unloaded_change.py'),
  cases=cases,actual_live_parameter_identity=dict(backbone_leaves=len(first['backbone']),head_leaves=len(first['head']),initial_equals_final=first==last,all_normal_and_deliberately_changed_parameter_hashes_independently_verified=True),
  source_identity=dict(producer=source(''),independent=source('AUDIT_')),
  source_scope='Normal cases and every actual fault loaded source/pyc/spec_origin/interpreter/native SHA have corresponding original full-tree snapshots and independent complete-byte witnesses. Producer final deliberate unsealed WinMM native addition is preserved; do not report production source-start/end equality. Its original full-tree sufficiency FAIL is distinct from raw replay0fail.',
  normal_timing=dict(actual_loaded_file_count=normal['actual_loaded_file_count'],source_sha_seconds=normal['loaded_seconds'],actual_full_live_parameters_sha_seconds=normal['full_live_parameter_seconds'],decision_seconds=normal['guard_decision_seconds']),
  time_scope='Standalone candidate clocks include actual loaded set enumeration/full byte hashing and actual full live parameter hashing. Original full reference each case and archive/starting seal/model initialization are separately performed; not a whole8+2 request. File set and order/cache differ from prior n6, so no causal acceleration claim.',
  protections_not_proven=['unloaded registered tree mutation is a measured missed change','ephemeral imports removed before snapshot','in-memory code monkeypatch','native arbitrary memory writes','cryptographic trust of module.__file__/__spec__ metadata','parent crash/persistent logs/irreversible external effects','all scientific training/control chain dependency closure'],
  exits={n:read(n) for n in ('RUN_EXIT.json','AUDIT_RUN_EXIT.json','CHAIN_EXIT.json')},archive=dict(files=len(manifest),bytes=sum(x['size'] for x in manifest),chunk_bytes=262144,all_original_sha_match=True,actual_owned_tree_empty_before_download=True),resource=dict(available_ram_gib=observed['available_ram_gib'],cpu15_busy_percent=observed['cpu15_busy_percent'],observed_other_python_count=sum(not p['owned'] for p in observed['processes']),other_processes_not_touched=True),
  old_full_hash_publication_gate_unchanged=True,candidate_installed=False,full_request_budget_qualified=False,full_scientific_loaded_closure_qualified=False,formal_trained_control_qualified=False,equal_time_maskco_iteration_qualified=False,family2_closed=True,scientific_family3_started=False,model_forward_calls=0,C0_solver_calls=0,optimizer_updates=0,stable_online_gain=False,
  next='Do not rerun these8 fault/normal cases or104649 independent checks. Loaded-only cannot replace original complete-tree gate. Any immutable-source lease/ownership alternative must be newly preregistered and tested only on new exclusive source copies, without locking team/global files, and preserve full original file/loaded/native/parameter requirements or openly declare a new contract before scientific data. Per-expression whole-request deployment remains gated.')
 with (O/'STAGE_ADJUDICATION.json').open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
 print(json.dumps({k:obj[k] for k in ('status','independent_checks','independent_failures','negative_unloaded_registered_source_actual','actual_live_parameter_identity','source_identity','normal_timing','archive','resource')}))
if __name__=='__main__':main()
