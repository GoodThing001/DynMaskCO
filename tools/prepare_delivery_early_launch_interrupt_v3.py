"""Seal eager startup imports and one new BaseException boundary, no old replay."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'research/review_3080_20261010/delivery/early_launch_lifecycle_root_v2'
OUT=ROOT/'research/review_3080_20261010/delivery/early_launch_lifecycle_root_v3'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    for p in OLD.glob('*.py'):(OUT/p.name).open('xb').write(p.read_bytes())
    p=OUT/'delivery_phase_shared_parent_v3.py'
    text=p.read_text(encoding='utf-8').replace('from pathlib import Path\n',
        'from pathlib import Path\nfrom bounded_phase_trace_v1 import PhaseTrace,MAX_RECORDS\n',1)
    p.write_text(text,encoding='utf-8',newline='\n')
    runner=(ROOT/'tools/run_delivery_early_launch_v2.py').read_text(encoding='utf-8')
    runner=runner.replace('early_launch_lifecycle_root_v2','early_launch_lifecycle_root_v3')
    runner=runner.replace("('identity_failure','receipt_archive_failure')", "('identity_failure','receipt_archive_failure','identity_interrupt')")
    runner=runner.replace("            raise PermissionError('Registered child identity query failure')",
        "            if name=='identity_interrupt':raise SystemExit('Registered startup identity interruption')\n            raise PermissionError('Registered child identity query failure')")
    runner=runner.replace("receipt_archive_failure=('PermissionError',4,1))",
        "receipt_archive_failure=('PermissionError',4,1),identity_interrupt=('SystemExit',4,1))")
    path=ROOT/'tools/run_delivery_early_launch_v3.py';ast.parse(runner);path.open('x',encoding='utf-8',newline='\n').write(runner)
    supervisor=(ROOT/'tools/supervise_delivery_early_launch_v2.py').read_text(encoding='utf-8')
    supervisor=supervisor.replace('early_launch_lifecycle_root_v2','early_launch_lifecycle_root_v3').replace('run_delivery_early_launch_v2.py','run_delivery_early_launch_v3.py')
    path=ROOT/'tools/supervise_delivery_early_launch_v3.py';ast.parse(supervisor);path.open('x',encoding='utf-8',newline='\n').write(supervisor)
    files={str(p.relative_to(ROOT)):sha(p) for p in OUT.glob('*.py')}
    for name in ('prepare_delivery_early_launch_interrupt_v3.py','run_delivery_early_launch_v3.py',
                 'supervise_delivery_early_launch_v3.py','pickup_loaded_closure_v1.py'):
        p=ROOT/'tools'/name;files[str(p.relative_to(ROOT))]=sha(p)
    for name in files:ast.parse((ROOT/name).read_text(encoding='utf-8'))
    reg=dict(registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,
        prior_registration_sha256=sha(OLD/'ENGINEERING_REGISTRATION.json'),
        prior_audit_sha256=sha(OLD/'INDEPENDENT_LIFECYCLE_AUDIT.json'),
        cases=['identity_interrupt'],previous_seven_cases_not_replayed=True,
        execution_authorized_for_early_launch_component=True,model_execution_authorized=False,
        early_launch_component_only=True,total_request_seconds=10,child_seconds=8,reap_reserve_seconds=2,
        outer_supervisor_seconds=60,fixture_child_ready_seconds=4,
        revisions=['Import bounded_phase_trace_v1 before source start without calling its kernel',
            'One new real-child SystemExit/except BaseException startup boundary'],
        helper_v2_bytes_unchanged=sha(OUT/'phase_failure_cleanup_v2.py')==sha(OLD/'phase_failure_cleanup_v2.py'),
        engine_bytes_unchanged=sha(OUT/'delivery_phase_actual_engine_v1.py')==sha(OLD/'delivery_phase_actual_engine_v1.py'),
        old_QA_repeated=False,full_request_budget_qualified=False,
        full_scientific_loaded_closure_qualified=False,formal_controls_qualified=False,
        scientific_family3_started=False,stable_online_gain=False,remote_upload=False)
    with (OUT/'ENGINEERING_REGISTRATION.json').open('x',encoding='utf-8') as f:json.dump(reg,f,indent=2)
    print('One new startup interruption registered; all seven prior cases excluded.')


if __name__=='__main__':main()
