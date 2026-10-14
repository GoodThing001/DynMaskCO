"""Register exactly two new phase-instrumented model requests, before execution."""
import ast
import hashlib
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'research/review_3080_20261010'
D=R/'delivery'
PRE=D/'early_launch_lifecycle_root_v3'
OLD=D/'timed_actual_request_execution_root_v1'
OUT=D/'phase_timed_model_execution_root_v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def put(p,x):
    p=Path(p)
    if isinstance(x,str):
        if p.suffix=='.py':ast.parse(x,filename=str(p))
        with p.open('x',encoding='utf-8',newline='\n') as f:f.write(x)
    else:
        with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    old=json.loads((OLD/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
    sealed=json.loads((PRE/'ENGINEERING_REGISTRATION.json').read_text(encoding='utf-8'))
    for n,h in sealed['files'].items():assert sha(ROOT/n)==h,n
    assert json.loads((PRE/'INDEPENDENT_LIFECYCLE_AUDIT.json').read_text())['failures']==[]
    OUT.mkdir(exist_ok=False)
    copied=['delivery_phase_actual_engine_v1.py','delivery_phase_shared_worker_v2.py',
        'delivery_phase_shared_parent_v3.py','phase_failure_cleanup_v2.py','startup_native_identity_v1.py',
        'bounded_phase_trace_v1.py','independent_phase_trace_parser_v1.py',
        'delivery_actual_action_head_v1.py','delivery_actual_score_chain_v1.py',
        'delivery_feasible_action_options_v1.py','delivery_retained_edge_action_guard_v1.py',
        'features_objective.py','action_head_runtime_v1.py','timed_evidence_io_v1.py']
    for n in copied:shutil.copy2(PRE/n,OUT/n)
    shutil.copy2(OLD/'CHAIN_INPUT.json',OUT/'CHAIN_INPUT.json')
    shutil.copy2(D/'phase_timing_wiring_preparation_root_v1/PHASE_DESIGN.json',OUT/'PHASE_DESIGN.json')
    driver=(ROOT/'tools/run_delivery_timed_request_v1.py').read_text(encoding='utf-8')
    driver=driver.replace('from delivery_timed_shared_parent_v3 import request','from delivery_phase_shared_parent_v3 import request')
    put(OUT/'run_delivery_phase_timed_request_v1.py',driver)
    audit=(ROOT/'tools/audit_delivery_timed_request_v2.py').read_text(encoding='utf-8')
    audit=audit.replace('import copy,gc,hashlib,io,json,sys,time,traceback','import copy,gc,hashlib,io,json,sys,time,traceback\nfrom fractions import Fraction\nfrom independent_phase_trace_parser_v1 import parse\nfrom startup_native_identity_v1 import prime_native_identity')
    audit=audit.replace('    import numpy as np,psutil,jax,jax.numpy as jnp','    native_handles=prime_native_identity()\n    import numpy as np,psutil,jax,jax.numpy as jnp',1)
    audit=audit.replace("for f,h in st['evidence_files'].items():ck(tag+':stage_artifact:'+f,sha(root/f)==h)",
        "for f,h in st['evidence_files'].items():ck(tag+':stage_artifact:'+f,sha(root/Path(f))==h)")
    audit=audit.replace("all(f in st['evidence_files'] for f in required_files.get(st['stage'],[]))",
        "all(f in {Path(k).as_posix() for k in st['evidence_files']} for f in required_files.get(st['stage'],[]))")
    audit=audit.replace("    checks=[];unexercised=[];partials=[];missing_source_end=[];largest=0.;forward_count=0",
        "    checks=[];unexercised=[];partials=[];missing_source_end=[];largest=0.;forward_count=0;phase_summaries=[]")
    anchor="            prod_start=read(root/'SOURCE_START.json')"
    injection='''            # Parse only these newly saved buffers after the exact writer reap.
            # Closed trace is a byte status, not candidate/source/scientific eligibility.
            for role in ('WORKER','CALLER'):
                trace_path=BASE/(tag+'_'+role+'_TRACE.bin')
                parsed=parse(trace_path.read_bytes(),1,s,e,receipt['worker_reaped'])
                ck(tag+':trace_schema:'+role,len(parsed['records'])>0)
                phase={}
                for item in parsed['records']:
                    label=str(item['phase']);group=phase.setdefault(label,dict(completed=0,open=0,exception=0,exact_completed_seconds=[0,1]))
                    if item['status']==0:group['open']+=1;continue
                    group['completed']+=1
                    if item['status']==2:group['exception']+=1
                    value=Fraction(*group['exact_completed_seconds'])+Fraction(*item['exact_operation_seconds'])
                    group['exact_completed_seconds']=[value.numerator,value.denominator]
                ck(tag+':trace_union_bounded:'+role,Fraction(*parsed['exact_operation_union_seconds'])<=Fraction(*parsed['exact_outer_seconds']))
                ck(tag+':caller_trace_closed:'+role,role!='CALLER' or parsed['closed'])
                phase_summaries.append(dict(tag=tag,role=role,**parsed,phase_totals=phase,
                    phases_are_combined_as_registered=True,nested_phase_totals_not_added_as_complete_request=True,
                    writer_end_dependency_closure_available=(q/'SOURCE_END.json').exists()))
'''
    assert audit.count(anchor)==1
    audit=audit.replace(anchor,injection+anchor)
    audit=audit.replace("new_timed_evidence_forward_replays=forward_count,max_abs_numpy_head_error=largest,",
        "new_timed_evidence_forward_replays=forward_count,max_abs_numpy_head_error=largest,phase_summaries=phase_summaries,")
    put(OUT/'audit_delivery_phase_timed_request_v1.py',audit)
    supervisor=(ROOT/'tools/supervise_delivery_timed_request_v1.py').read_text(encoding='utf-8')
    supervisor=supervisor.replace('run_delivery_timed_request_v1.py','run_delivery_phase_timed_request_v1.py').replace('audit_delivery_timed_request_v2.py','audit_delivery_phase_timed_request_v1.py')
    put(OUT/'supervise_delivery_phase_timed_request_v1.py',supervisor)
    launcher=(ROOT/'tools/timed_request_remote_launcher_v1.py').read_text(encoding='utf-8').replace('supervise_delivery_timed_request_v1.py','supervise_delivery_phase_timed_request_v1.py')
    put(OUT/'phase_timed_request_remote_launcher_v1.py',launcher)
    shutil.copy2(ROOT/'tools/timed_request_remote_status_v1.py',OUT/'phase_timed_request_remote_status_v1.py')
    transport=(ROOT/'tools/transport_delivery_timed_request_v1.py').read_text(encoding='utf-8')
    transport=transport.replace('timed_actual_request_execution_root_v1','phase_timed_model_execution_root_v1').replace('timed_request_remote_launcher_v1.py','phase_timed_request_remote_launcher_v1.py').replace('timed_request_remote_status_v1.py','phase_timed_request_remote_status_v1.py')
    tp=ROOT/'tools/transport_delivery_phase_timed_request_v1.py';put(tp,transport)
    names=copied+['CHAIN_INPUT.json','PHASE_DESIGN.json','run_delivery_phase_timed_request_v1.py',
        'audit_delivery_phase_timed_request_v1.py','supervise_delivery_phase_timed_request_v1.py',
        'phase_timed_request_remote_launcher_v1.py','phase_timed_request_remote_status_v1.py']
    row=json.loads((OUT/'CHAIN_INPUT.json').read_text(encoding='utf-8'))
    assert row['key']==[12,3] and len(row['remaining'])==25
    assert not any(k in row for k in ('teacher_routes','oracle','labels','target_rank','cost'))
    imports=[]
    for node in ast.walk(ast.parse(audit)):
        if isinstance(node,ast.Import):imports += [x.name for x in node.names]
        elif isinstance(node,ast.ImportFrom):imports.append(node.module or '')
    forbidden=['delivery_phase_actual_engine','delivery_phase_shared','delivery_actual_action_head',
        'delivery_actual_score_chain','delivery_feasible_action_options','delivery_retained_edge_action_guard','features_objective','bounded_phase_trace']
    assert not any(m.startswith(f) for m in imports for f in forbidden)
    for n in names:
        if n.endswith('.py'):ast.parse((OUT/n).read_text(encoding='utf-8'),filename=n)
    reg={k:v for k,v in old.items() if k not in ('package_files','transport_tool_sha256','source_review_sha256')}
    reg.update(created_utc=datetime.now(timezone.utc).isoformat(),status='PROSPECTIVE_PHASE_INSTRUMENTED_TWO_INITIAL_MODEL_REQUESTS',
        execution_authorized=True,model_execution_authorized=True,independent_timed_request_checker_sealed=True,
        early_launch_component_only=False,prior_registration_sha256=sha(OLD/'EXECUTION_REGISTRATION.json'),
        new_startup_registration_sha256=sha(PRE/'ENGINEERING_REGISTRATION.json'),
        preparation_tool_sha256=sha(Path(__file__)),
        package_files={n:sha(OUT/n) for n in names},requests=[dict(arm=a,fault_stage=0) for a in ['pretrained_initial','random888_initial']],
        selection_rule='Exactly two new phase-instrumented requests, one per fixed initial arm; first old key12,3 unchanged; no faults or retries, no scientific utility selection.',
        maximum_proposed_encodes=2,maximum_proposed_decodes=50,
        stage_execution=['run_delivery_phase_timed_request_v1.py','audit_delivery_phase_timed_request_v1.py'],
        remote_directory='research/review_3080_20261010/phase_timed_model_execution_root_v1',
        transport_tool_sha256=sha(tp),actual_phase_design_sha256=sha(OUT/'PHASE_DESIGN.json'),
        new_telemetry_observation=True,old_ten_requests_not_replayed=True,
        independent_trace_scope='Saved new worker/caller bytes only; independent parser and exact binary64 rational unions. Open/torn/exception records retained; malformed buffers fail without rerun.',
        unchanged_original_engine_sha256=sha(OUT/'delivery_phase_actual_engine_v1.py'),
        numeric_contract_unchanged=True,stage_faults_not_retested=True,
        full_request_budget_qualified=False,full_scientific_loaded_closure_qualified=False,
        formal_controls_qualified=False,learned_output_qualified=False,scientific_family3_started=False,stable_online_gain=False)
    put(OUT/'EXECUTION_REGISTRATION.json',reg)
    put(OUT/'LOCAL_EXECUTION_BINDING.json',dict(registration_sha256=sha(OUT/'EXECUTION_REGISTRATION.json'),
        transport_tool_sha256=sha(tp),package_files=reg['package_files'],shared_MD_upload_authorized=False,model_executed_at_registration=False))
    put(OUT/'SOURCE_REVIEW_BEFORE_EXECUTION.json',dict(independent_imports=imports,forbidden_imports=forbidden,
        engine_bytes_unchanged=True,source_only=True,old_scientific_numeric_contract_unchanged=True,
        limits=['Prepared public snapshot only; no live installation or formal trained controls',
            'Receipt creation remains outside caller measured elapsed',
            'Parent synchronous bounded copy/hash/archive is timed but not arbitrarily preemptible',
            'Phase8 NPZ and phase14 enumeration/hash are combined; no inference about their internal counterfactual speed',
            'Trace validity and initial untrained scores are not online gain or complete scientific dependency qualification']))
    print(json.dumps(dict(directory=str(OUT),registration_sha256=sha(OUT/'EXECUTION_REGISTRATION.json'),requests=2,model_executed=False,science_started=False)))


if __name__=='__main__':main()
