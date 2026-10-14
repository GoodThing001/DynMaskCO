"""Prospective bounded engineering registration, before any actual forward."""
import ast,json,shutil
from pathlib import Path
from datetime import datetime,timezone
from record_pickup_inactive_slot_cancel_20261012 import sha,read,create
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'research/review_3080_20261010/delivery';SOURCE=D/'action_head_independent_replay_preparation_root_v1';O=D/'actual_action_head_execution_root_v1'

def main():
    plan=read(SOURCE/'EXECUTION_PLAN_PREPARATION.json')
    assert not plan['execution_authorized'] and all(sha(SOURCE/n)==h for n,h in plan['package_files'].items())
    O.mkdir(exist_ok=False);package={}
    for n in plan['package_files']:
        if n=='supervise_delivery_action_head_v2.py':continue
        shutil.copy2(SOURCE/n,O/n);package[n]=sha(O/n)
    for n in ['supervise_delivery_action_head_v3.py','action_head_remote_launcher_v1.py','action_head_remote_status_v1.py']:
        p=ROOT/'tools'/n;ast.parse(p.read_text(encoding='utf-8'));shutil.copy2(p,O/n);package[n]=sha(O/n)
    transport=ROOT/'tools/transport_delivery_action_head_v1.py';ast.parse(transport.read_text(encoding='utf-8'));shutil.copy2(transport,O/'LOCAL_FROZEN_TRANSPORT.py')
    helpers={str(ROOT/'tools'/n):sha(ROOT/'tools'/n) for n in ['_srv.py','_srv_exec_v2.py']}
    reg={k:v for k,v in plan.items() if k not in ['created_utc','status','outstanding','package_files','execution_authorized','resource_observation','resource_observation_sha256']}
    reg.update(created_utc=datetime.now(timezone.utc).isoformat(),status='PROSPECTIVE_INITIAL_ACTION_SCORING_ENGINEERING',execution_authorized=True,package_files=package,source_plan_sha256=sha(SOURCE/'EXECUTION_PLAN_PREPARATION.json'),transport_tool_sha256=sha(transport),local_transport_helpers=helpers,remote_directory='research/review_3080_20261010/actual_action_head_execution_root_v1',allowed_upload_types=['.py','.json'],stage_execution=['run_delivery_actual_action_head_v2.py','audit_delivery_actual_action_head_v2.py'],resource_policy=dict(minimum_available_ram_gib=8,affinity=[15],maximum_preflight_cpu15_busy_percent=20,refuse_observed_owned_research_tree=True,gpu_used=False),download_max_file_bytes=16*1024**2,download_max_total_bytes=256*1024**2,download_chunk_bytes=262144,scope='Exactly60 new initial-model component production forwards and60 independent full replay forwards on fixed all30 old first skeletons; zero optimizer/teacher/solver/C0/dynamic day. This is not scientific family3 or online10second qualification.',formal_controls_qualified=False,full_C0_qualified=False,full_request_budget_qualified=False,scientific_model_qualified=False,stable_online_gain=False,failures='Preserve all actual source/attempts/logs/output/exit; stop automatic retry; actual process inspection required; any repair uses new exclusive engineering registration and identical scientific parameters.')
    create(O/'EXECUTION_REGISTRATION.json',reg)
    create(O/'LOCAL_EXECUTION_BINDING.json',dict(registration_sha256=sha(O/'EXECUTION_REGISTRATION.json'),transport_tool_sha256=sha(transport),package_files=package,prepared_utc=datetime.now(timezone.utc).isoformat(),no_model_execution_at_registration=True))
    assert all(sha(O/n)==h for n,h in package.items()) and all(sha(SOURCE/n)==h for n,h in plan['package_files'].items())
    print(json.dumps(dict(registration_sha256=sha(O/'EXECUTION_REGISTRATION.json'),files=len(package),model_executed=False,scientific_family3_started=False)))

if __name__=='__main__':main()
