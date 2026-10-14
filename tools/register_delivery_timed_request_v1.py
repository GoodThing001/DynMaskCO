"""Accurate prospective initial-model timed engineering registration, no runtime."""
import ast,hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research/review_3080_20261010';D=R/'delivery'
PREP=D/'timed_request_independent_preparation_root_v1';OLD=D/'actual_score_chain_execution_root_v1';O=D/'timed_actual_request_execution_root_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    plan=read(PREP/'PREPARATION_REGISTRATION.json');old=read(OLD/'EXECUTION_REGISTRATION.json')
    assert sha(PREP/'PREPARATION_REGISTRATION.json')=='ffe65084b94ccd913a0cd198cd6e6aa9b524851f68b4fc507a83048b5c9f307a'
    assert all(sha(PREP/n)==h for n,h in plan['package_files'].items())
    deps=['delivery_actual_score_chain_v1.py','delivery_feasible_action_options_v1.py','delivery_retained_edge_action_guard_v1.py','delivery_actual_action_head_v1.py','action_head_runtime_v1.py','features_objective.py','CHAIN_INPUT.json']
    new=['delivery_timed_actual_engine_v3.py','delivery_timed_shared_worker_v3.py','delivery_timed_shared_parent_v3.py',
         'audit_delivery_timed_request_v2.py','timed_evidence_io_v1.py','run_delivery_timed_request_v1.py',
         'supervise_delivery_timed_request_v1.py','timed_request_remote_launcher_v1.py','timed_request_remote_status_v1.py']
    transport=ROOT/'tools/transport_delivery_timed_request_v1.py'
    for p in [ROOT/'tools'/n for n in new]+[PREP/n for n in deps if n.endswith('.py')]+[transport]:ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    row=read(PREP/'CHAIN_INPUT.json');assert row['key']==[12,3] and len(row['remaining'])==25
    assert not any(k in row for k in ('teacher_routes','oracle','target_rank','labels','cost'))
    for n in deps:assert sha(PREP/n)==old['package_files'][n]
    audit=(ROOT/'tools/audit_delivery_timed_request_v2.py').read_text(encoding='utf-8');imports=[]
    for node in ast.walk(ast.parse(audit)):
        if isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
        elif isinstance(node,ast.ImportFrom):imports.append(node.module or '')
    assert not any(m.startswith(f) for m in imports for f in ['delivery_timed_actual_engine','delivery_timed_shared','delivery_actual_action_head','delivery_actual_score_chain','delivery_feasible_action_options','delivery_retained_edge_action_guard','features_objective'])
    requests=plan['proposed_requests'];assert len(requests)==10 and len({(x['arm'],x['fault_stage']) for x in requests})==10
    O.mkdir(exist_ok=False)
    for n in deps:shutil.copy2(PREP/n,O/n)
    for n in new:shutil.copy2(ROOT/'tools'/n,O/n)
    shutil.copy2(transport,O/'LOCAL_FROZEN_TRANSPORT.py')
    review=dict(source_only=True,created_utc=datetime.now(timezone.utc).isoformat(),preparation_sha256=sha(PREP/'PREPARATION_REGISTRATION.json'),
        prior_missing_reader_path='tools/prepare_delivery_actual_score_chain_execution_v1.py',prior_missing_reader_did_not_start_model_or_server=True,
        static_review_changes=['Atomic completed artifact names, .part raw files retained on cancellation; no numeric/algorithm change.',
            'Reject entry marker when already cancelled; reached marker and actual delay entry still separate.',
            'Caller selected archive SHA moved before end clock; cleanup included, receipt creation outside explicitly.',
            'Natural nonzero exit distinguished from own kill, actual returncodes retained.',
            'Audit reports missing completed dependency closure separately; killed native computation cannot fabricate source-end.',
            'Exact10 unique requests bound, no execution retry; observed owned-descendant cleanup on900s supervisor timeout.',
            'Recursive exclusive archive uses relative path containment/file64MiB/total512MiB/readv256KiB bounds.'],
        independent_audit_imports=sorted(set(imports)),runtime_validation=False,no_old_QA_execution=True,
        fixed_scientific_parameters=True,source_preparation_original_files_not_changed=True,
        remaining_limits=plan['limitations'])
    create(O/'SOURCE_REVIEW_BEFORE_EXECUTION.json',review)
    reg=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='PROSPECTIVE_INITIAL_MODEL_PUBLIC_SNAPSHOT_TIMED_ENGINEERING',
        execution_authorized=True,independent_timed_request_checker_sealed=True,
        source_review_sha256=sha(O/'SOURCE_REVIEW_BEFORE_EXECUTION.json'),source_preparation_sha256=sha(PREP/'PREPARATION_REGISTRATION.json'),
        prior_completed_component_registration_sha256=sha(OLD/'EXECUTION_REGISTRATION.json'),
        package_files={n:sha(O/n) for n in deps+new},input_digest=old['input_digest'],input_sha256=sha(O/'CHAIN_INPUT.json'),key=[12,3],masked_count=25,
        requests=requests,total_request_seconds=10,child_seconds=8,reap_reserve_seconds=2,
        cpu_affinity=[15],minimum_available_ram_gib=8,head_seed=20261012,random_seed=888,optimizer_steps=0,
        old_remote_root=old['old_remote_root'],checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],
        contract_hash=old['contract_hash'],normalization_sources=old['normalization_sources'],numpy_head_max_abs_error=0.00002,
        selection_rule='Immutable first observed recordkey12,3; two initial arms normal once each plus pretrained primary arm actual entrance faults1..8 once each, regardless reach or outputs; no resampling.',
        old_initial_scores_reused=False,maximum_proposed_encodes=10,maximum_proposed_decodes=250,
        initialization_outside_request=['interpreter/libraries/checkpoint parameters and fixed head','shared allocation','prepared public input raw','original incumbent fallback certificate already exists'],
        initialization_no_inference_warmup=True,
        timed_scope='Caller bounded input copy; worker JSON/full10D/incumbentC0/encode/all current decode/head/local eligibility/choices/transitions evidence/candidateC0/actual file and full parameter check/serialize/publish; caller bounded SHA/JSON/C0 source selection/selected archive/SHA/kill-reap/handle cleanup. Receipt creation outside measured elapsed. Public snapshot only, no live installation.',
        registered_qualification='Separate raw replay/timing/fault reach/missing closure. Deadline8/total10 never extended to finish chain/fault. No qualified response returns original plan. Unreached fault means unexercised; missing or torn artifacts retained and cannot prove completion. Selected before deadline output survives later postpublish kill. No benefit criterion.',
        stage_execution=['run_delivery_timed_request_v1.py','audit_delivery_timed_request_v2.py'],stage_timeout_seconds=900,reap_timeout_seconds=10,
        remote_directory='research/review_3080_20261010/timed_actual_request_execution_root_v1',allowed_upload_types=['.py','.json'],
        transport_tool_sha256=sha(transport),local_transport_helpers={str(ROOT/'tools'/n):sha(ROOT/'tools'/n) for n in ['_srv.py','_srv_exec_v2.py']},
        download_max_file_bytes=64*1024**2,download_max_total_bytes=512*1024**2,download_chunk_bytes=262144,
        resource_policy=dict(minimum_available_ram_gib=8,affinity=[15],maximum_preflight_cpu15_busy_percent=20,refuse_observed_owned_research_tree=True,gpu_used=False),
        independent_scope='Independent original parameter/head/features/anchors/full encode/decode exact JAX scores; NumPy2e-5 only head arithmetic. Independent local geometry/regret/retained/customer competition and actual new partial/full evidence. Original JAX/Transformer/C0 physics shared, not another model/physics implementation.',
        full_request_budget_qualified=False,full_scientific_loaded_closure_qualified=False,formal_controls_qualified=False,learned_output_qualified=False,
        scientific_family3_started=False,stable_online_gain=False,old_QA_repeated=False,
        limitations=plan['limitations'],failures='Keep every raw failure/exit/.part/source/input. Never retry production or oldQA; any engineering recovery new exclusive directory after actual owned tree exit with prospective same scientific parameters.')
    create(O/'EXECUTION_REGISTRATION.json',reg)
    create(O/'LOCAL_EXECUTION_BINDING.json',dict(registration_sha256=sha(O/'EXECUTION_REGISTRATION.json'),transport_tool_sha256=sha(transport),package_files=reg['package_files'],model_executed_at_registration=False,shared_MD_upload_authorized=False))
    print(json.dumps(dict(directory=str(O),registration_sha256=sha(O/'EXECUTION_REGISTRATION.json'),package_files=len(reg['package_files']),requests=len(requests),science_started=False)))
if __name__=='__main__':main()
