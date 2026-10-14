"""Exact engineering execution registration, before new chain/forward/C0."""
import ast,json,shutil
from pathlib import Path
from datetime import datetime,timezone
from prepare_delivery_actual_score_chain_v3 import sha,read,create
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'research/review_3080_20261010/delivery'
SOURCE=D/'actual_score_chain_preparation_root_v3';O=D/'actual_score_chain_execution_root_v1'

def main():
    plan=read(SOURCE/'INTEGRATION_PLAN_PREPARATION.json')
    assert not plan['execution_authorized'] and not plan['model_executed']
    assert all(sha(SOURCE/n)==h for n,h in plan['package_files'].items())
    extra=['audit_delivery_actual_score_chain_v1.py','supervise_delivery_actual_score_chain_v1.py',
           'score_chain_remote_launcher_v1.py','score_chain_remote_status_v1.py']
    for name in list(plan['package_files'])+extra:
        p=SOURCE/name if name in plan['package_files'] else ROOT/'tools'/name
        if name.endswith('.py'):ast.parse(p.read_text(encoding='utf-8'))
    transport=ROOT/'tools/transport_delivery_actual_score_chain_v1.py';ast.parse(transport.read_text(encoding='utf-8'))
    O.mkdir(exist_ok=False);package={}
    for name in list(plan['package_files'])+extra:
        p=SOURCE/name if name in plan['package_files'] else ROOT/'tools'/name
        shutil.copy2(p,O/name);package[name]=sha(O/name)
    shutil.copy2(transport,O/'LOCAL_FROZEN_TRANSPORT.py')
    reg={k:v for k,v in plan.items() if k not in ('created_utc','status','package_files','outstanding')}
    reg.update(created_utc=datetime.now(timezone.utc).isoformat(),status='PROSPECTIVE_TWO_INITIAL_ARM_ACTUAL_CHAIN_ENGINEERING',
        execution_authorized=True,independent_full_chain_checker_sealed=True,package_files=package,
        source_plan_sha256=sha(SOURCE/'INTEGRATION_PLAN_PREPARATION.json'),transport_tool_sha256=sha(transport),
        local_transport_helpers={str(ROOT/'tools'/n):sha(ROOT/'tools'/n) for n in ['_srv.py','_srv_exec_v2.py']},
        remote_directory='research/review_3080_20261010/actual_score_chain_execution_root_v1',allowed_upload_types=['.py','.json'],
        stage_execution=['run_delivery_actual_score_chain_v1.py','audit_delivery_actual_score_chain_v1.py'],
        stage_timeout_seconds=900,reap_timeout_seconds=10,
        download_max_file_bytes=16*1024**2,download_max_total_bytes=256*1024**2,download_chunk_bytes=262144,
        resource_policy=dict(minimum_available_ram_gib=8,affinity=[15],maximum_preflight_cpu15_busy_percent=20,
            refuse_observed_owned_research_tree=True,gpu_used=False),
        independent_replay_workload=dict(new_initial_encodes=2,new_context_full_encode_decode_replays_maximum=48,
            original_initial_00_full_forward_repeated=False,
            score_choice='Exact independently reconstructed JAX head outputs required; NumPy2e-5 applies only separate head arithmetic, no ranking relaxation.',
            geometry='Independent direct NumPy original operation grouping; all boolean local-route/retained filters and saved distances/competition exact; no science threshold relaxation.'),
        qualification_rule='Raw independent source/parameter/features/cache/score/option/transition/C0/fallback evidence must all pass. If no feasible action, preserve incomplete chain and incumbent fallback and report domain/status explicitly; never label incomplete construction complete. No benefit criterion or model selection.',
        preloaded_scope='Interpreter/import/backend/parameter construction/feature preparation and closed initial score artifacts outside engineering arm timer; no new encode/decode warmup. IncumbentC0/encoder/newdecodes/trace serialization/finalC0 inside arm timer. Stage900s is not10s request.',
        original_initial_component_QA_repeated=False,formal_controls_qualified=False,full_request_budget_qualified=False,
        full_scientific_loaded_closure_qualified=False,scientific_family3_started=False,stable_online_gain=False,
        failures='Preserve all outputs/source/exit; no automatic retry. Inspect actual process tree after errors. Any repair is separately registered new exclusive tree with same scientific parameters; closedQA not rerun.')
    create(O/'EXECUTION_REGISTRATION.json',reg)
    create(O/'LOCAL_EXECUTION_BINDING.json',dict(registration_sha256=sha(O/'EXECUTION_REGISTRATION.json'),
        transport_tool_sha256=sha(transport),package_files=package,model_executed_at_registration=False,
        shared_MD_upload_authorized=False))
    assert all(sha(O/n)==h for n,h in package.items())
    print(json.dumps(dict(registration_sha256=sha(O/'EXECUTION_REGISTRATION.json'),package_files=len(package),
        independent_audit_source_sealed=True,model_executed=False,scientific_family3_started=False)))

if __name__=='__main__':main()
