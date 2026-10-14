"""Seal exact new startup workload before generating any case observations."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/review_3080_20261010/delivery/early_launch_lifecycle_root_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not (OUT/'ENGINEERING_REGISTRATION.json').exists()
    files={str(p.relative_to(ROOT)):sha(p) for p in OUT.glob('*.py')}
    for name in ('prepare_delivery_early_launch_v1.py','run_delivery_early_launch_v1.py',
                 'supervise_delivery_early_launch_v1.py','register_delivery_early_launch_v1.py',
                 'pickup_loaded_closure_v1.py'):
        p=ROOT/'tools'/name;files[str(p.relative_to(ROOT))]=sha(p)
    for name in files:ast.parse((ROOT/name).read_text(encoding='utf-8'))
    reg=dict(registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        files=files,cases=['mapping_failure','log_collision','popen_failure','identity_failure',
                         'init_exit','startup_endpoint','receipt_archive_failure'],
        execution_authorized_for_early_launch_component=True,model_execution_authorized=False,
        old_QA_repeated=False,old_model_worker_not_imported=True,
        early_launch_component_only=True,total_request_seconds=10,child_seconds=8,reap_reserve_seconds=2,
        original_source_failure_preserved=True,native_priming_not_sample_warmup=True,
        outer_supervisor_seconds=60,fixture_child_ready_seconds=4,fixture_stop_before_request_sequence=True,
        full_request_budget_qualified=False,full_scientific_loaded_closure_qualified=False,
        formal_controls_qualified=False,scientific_family3_started=False,stable_online_gain=False,
        local_only=True,remote_upload=False,
        observed_process_inventory='Prior Get-CimInstance returned AccessDenied; no absence inference. Only direct launched Popen handles are killed.',
        caveat='These are new startup lifecycle boundaries. Fixture worker neither imports model nor calls telemetry kernel/C0/solver. Native preload qualifies only actual component snapshots.')
    with (OUT/'ENGINEERING_REGISTRATION.json').open('x',encoding='utf-8') as f:json.dump(reg,f,indent=2)
    print('Registered seven new startup boundaries; actual model execution prohibited.')


if __name__=='__main__':main()
