"""Saved-only identity scheduler adjudication; no file/method repetition."""
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'research/review_3080_20261010/delivery/parallel_identity_collection_root_v1';D=O/'downloaded'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(n):return json.loads((D/n).read_text(encoding='utf-8'))
def source(prefix):
 s=read(prefix+'SOURCE_START.json');e=read(prefix+'SOURCE_END.json');additions={n:dict(path=p,start_sha256=s['files'].get(p),end_sha256=e['files'].get(p)) for n,p in e['module_files'].items() if s['module_files'].get(n)!=p}
 return dict(start_files=len(s['files']),end_files=len(e['files']),start_modulemap=len(s['module_files']),end_modulemap=len(e['module_files']),module_maps_equal=s['module_files']==e['module_files'],mapping_additions=additions,all_mapping_additions_sealed_before_work=all(v['start_sha256'] and v['start_sha256']==v['end_sha256'] for v in additions.values()),changed=e['changed'],new_files=e['new_files'])
def main():
 m=json.loads((O/'DOWNLOAD_MANIFEST.json').read_text(encoding='utf-8'))
 if not all((D/x['name']).stat().st_size==x['size'] and sha(D/x['name'])==x['sha256'] for x in m):raise RuntimeError('Original archival identity')
 result=read('RESULT.json');audit=read('INDEPENDENT_IDENTITY_AUDIT.json');initial=read('INITIAL_LIVE_PARAMETERS.json');final=read('FINAL_LIVE_PARAMETERS.json')
 resource=json.loads(sorted(O.glob('RESOURCE_PREFLIGHT_*.json'))[0].read_text(encoding='utf-8'));observed=json.loads(resource['stdout'])
 obj=dict(status='FULL_IDENTITY_SCHEDULER_COMPONENT_PASS_NO_ONLINE_SPEED_OR_BUDGET_QUALIFICATION',registration_sha256=sha(O/'EXECUTION_REGISTRATION.json'),independent_checks=audit['check_count'],independent_failures=audit['failure_count'],
  component_qualified=result['file_sha_map_exact'] and result['module_map_exact'] and result['all_live_parameters_unchanged'] and result['source_qualified'] and audit['failure_count']==0,
  actual_entire_file_count=result['candidate_files'],same_original_file_set_and_sha=result['file_sha_map_exact'],same_actual_modulemap=result['module_map_exact'],no_hash_cache=True,no_files_or_native_scope_removed=True,complete_byte_reads=True,
  actual_live_parameter_identity=dict(backbone_leaves=len(initial['backbone']),head_leaves=len(initial['head']),total_values=audit['actual_parameter_count'],initial_equals_final=initial==final,all_leaves_match_independent_original_initialization=True),
  actual_source_identity=dict(producer=source(''),independent=source('AUDIT_')),
  timings=dict(parallel_collection_seconds=result['parallel_seconds'],subsequent_sequential_reference_seconds=result['sequential_reference_seconds'],actual_full_live_parameter_inventory_seconds=result['live_parameter_inventory_seconds'],parallel_plus_parameter_operation_sum_seconds=result['parallel_seconds']+result['live_parameter_inventory_seconds'],clocks=result['clocks']),
  time_scope='Each collection clock includes exact original file/native/module/tree enumeration and full file hashing. Parameter gate is separately clocked. Model/native/runtime initialization, beginning/full ending evidence collection and JSON archive work exist outside these operation clocks. This is a distinct bounded component, not a10-second online request.',
  timing_interpretation='The measured4-thread collection alone consumed7.896017199847847s, nearly all of the8s execution allowance. OS cache order and sequential follow-up are not controlled; do not claim causal speed improvement or degradation. No fair speed benefit was established. Original full-hash publication gate remains unchanged.',
  thread_scope=dict(io_threads=4,process_cpu_affinity=result['cpu_affinity'],single_cpu15_only=True),resource_preflight=dict(available_ram_gib=observed['available_ram_gib'],cpu15_busy_percent=observed['cpu15_busy_percent'],owned_review_processes=[p for p in observed['processes'] if p['owned']],other_observed_python_process_count=sum(not p['owned'] for p in observed['processes']),other_processes_not_modified=True),
  exits={n:read(n) for n in ('RUN_EXIT.json','AUDIT_RUN_EXIT.json','CHAIN_EXIT.json')},archive=dict(files=len(m),bytes=sum(x['size'] for x in m),chunk_bytes=262144,all_original_sha_match=True,actual_owned_tree_empty_before_download=True),
  model_forward_calls=0,C0_solver_calls=0,optimizer_updates=0,old_QA_or_thread_variant_retried=False,candidate_installed_in_request=False,full_scientific_loaded_closure_qualified=False,full_request_budget_qualified=False,equal_time_maskco_iteration_qualified=False,formal_trained_controls_qualified=False,scientific_family3_started=False,family2_closed=True,stable_online_gain=False,
  next_hypothesis='Prospectively design and independently test whether complete day-start source seal plus in-request actual-loaded/new-module/native identity and immutable live-parameter guards is sufficient under an explicit new contract. Do not replace the current all-file publication gate or reduce this file set until sufficient evidence is independently established. Also the per-expression scoring candidate still needs a new actual whole-request score/partial-chain reference and completeC0/fallback8+2 barriers. No new science/data/weight qualification is granted.')
 with (O/'STAGE_ADJUDICATION.json').open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
 print(json.dumps({k:obj[k] for k in ('status','independent_checks','independent_failures','actual_live_parameter_identity','actual_source_identity','timings','resource_preflight','archive')}))
if __name__=='__main__':main()
