"""Preserve saved semantics result and timing scope; no execution/replay."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'research/review_3080_20261010/delivery/primitive_schedule_semantics_root_v1';D=O/'downloaded'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(n):return json.loads((D/n).read_text(encoding='utf-8'))
def source(prefix):
 s=load(prefix+'SOURCE_START.json');e=load(prefix+'SOURCE_END.json')
 additions={n:dict(path=p,start_sha256=s['files'].get(p),end_sha256=e['files'].get(p)) for n,p in e['module_files'].items() if s['module_files'].get(n)!=p}
 return dict(start_files=len(s['files']),end_files=len(e['files']),start_modulemap=len(s['module_files']),end_modulemap=len(e['module_files']),maps_equal=s['module_files']==e['module_files'],mapping_changes=additions,all_mapping_additions_sealed_at_start=all(x['start_sha256'] and x['start_sha256']==x['end_sha256'] for x in additions.values()),changed=e['changed'],new_files=e['new_files'])
def main():
 manifest=json.loads((O/'DOWNLOAD_MANIFEST.json').read_text(encoding='utf-8'))
 if not all((D/x['name']).stat().st_size==x['size'] and sha(D/x['name'])==x['sha256'] for x in manifest):raise RuntimeError('Saved original archive identity')
 result=load('RESULT.json');audit=load('INDEPENDENT_SEMANTICS_AUDIT.json');reg=json.loads((O/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
 timings=[]
 for r in result['records']:
  c=r['candidate'];rows=c['operation_rows']
  timings.append(dict(case_index=r['case']['index'],nodes=r['case']['nodes'],actions=r['actions_count'],input_preparation_seconds=r['input_preparation_seconds'],original=r['original'],candidate=dict(preparation_seconds=c['preparation_seconds'],dispatch_and_possible_compile_sum_seconds=sum(v['dispatch_and_possible_compile_seconds'] for v in rows),sync_sum_seconds=sum(v['sync_seconds'] for v in rows),total_before_host_seconds=c['total_before_host_seconds'],final_slice_and_host_seconds=c['host_seconds'],operation_rows=rows)))
 obs=sorted(O.glob('RESOURCE_PREFLIGHT_*.json'))
 resource=json.loads(obs[0].read_text(encoding='utf-8'));observed=json.loads(resource['stdout'])
 obj=dict(status='NEW_THREE_SYNTHETIC_INPUT_STRICT_COMPONENT_PASS_NO_SCIENTIFIC_INSTALL',registration_sha256=sha(O/'EXECUTION_REGISTRATION.json'),method=reg['method'],
  exact_case_count=result['exact_equal_count'],total_cases=result['count'],total_action_outputs=sum(r['actions_count'] for r in result['records']),all_float32_score_mismatch_counts=[r['bitwise_mismatch_count'] for r in result['records']],independent_checks=audit['check_count'],independent_failures=audit['failure_count'],
  original_four_parameters_exact=True,semantics_component_qualified=result['synthetic_semantics_qualified'] and result['source_qualified'] and audit['failure_count']==0,
  strict_scope='Only fixed3 new synthetic current-skeleton/latent inputs and four original seeded float32 head leaves. Original NNX input/output layers unchanged; no full-head compiler combination. This is observed exact output on these inputs, not a proof for all inputs or general graph identity.',
  source=dict(producer=source(''),independent=source('AUDIT_')),resource=dict(available_ram_gib=observed['available_ram_gib'],cpu15_busy_percent=observed['cpu15_busy_percent'],owned_research_processes_at_preflight=[p for p in observed['processes'] if p['owned']]),
  timing=timings,timing_limit='One cold baseline and one candidate call per new input; original precedes candidate, caches and extra synchronization differ. No warmup. Dispatch includes possible compilation, sync separate; final candidate column-slice may finish in final_slice_and_host_seconds. Scope excludes source enumeration/archive preparation; no performance or8+2 whole-request claim.',
  exits={n:load(n) for n in ('RUN_EXIT.json','AUDIT_RUN_EXIT.json','CHAIN_EXIT.json')},archive=dict(files=len(manifest),bytes=sum(x['size'] for x in manifest),all_original_sha_match=True,chunk_bytes=262144),
  candidate_installed=False,old34_compiled_FAIL_unchanged=True,old_intermediate8892_QA_repeated=False,backbone_executions=0,C0_solver_calls=0,optimizer_updates=0,family2_closed=True,scientific_family3_started=False,
  formal_controls_qualified=False,full_scientific_loaded_closure_qualified=False,full_request_budget_qualified=False,equal_time_maskco_iteration_qualified=False,stable_online_gain=False,
  next='Keep these3 outputs/source evidence sealed, do not repeat component QA. Prospectively bind this per-expression schedule to a new bounded whole-request engineering transaction, original current-state action identities/completeC0/original fallback and all actual source/parameters before8-second publication gate, with2-second reaping reserve. Any old-input/full-chain semantic or budget gap remains a distinct gate. Formal trained same-initCE/task-random/strongdynamic and family3science remain closed.')
 with (O/'STAGE_ADJUDICATION.json').open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
 print(json.dumps({k:obj[k] for k in ('status','exact_case_count','total_action_outputs','independent_checks','independent_failures','source','resource','archive')}))
if __name__=='__main__':main()
