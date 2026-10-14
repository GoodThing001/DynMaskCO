"""Append a recovery for four unreached child boundaries; no passed-case replay."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'research/review_3080_20261010/delivery/early_launch_lifecycle_root_v1'
OUT=ROOT/'research/review_3080_20261010/delivery/early_launch_lifecycle_root_v2'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    for p in OLD.glob('*.py'):
        (OUT/p.name).open('xb').write(p.read_bytes())
    fixture=OUT/'early_launch_fixture_worker_v1.py'
    text=fixture.read_text(encoding='utf-8').replace("telemetry[:17]=b'EARLY-LAUNCH-ONLY!'",
        "marker=b'EARLY-LAUNCH-ONLY!';telemetry[:len(marker)]=marker")
    fixture.write_text(text,encoding='utf-8',newline='\n')
    native=OUT/'startup_native_identity_v1.py'
    text=native.read_text(encoding='utf-8').replace("    native = ctypes.WinDLL(str(path.resolve()))",
        "    native = (ctypes.WinDLL(str(path.resolve())),\n              ctypes.WinDLL(str((path.parent/'ntmarta.dll').resolve())))")
    native.write_text(text,encoding='utf-8',newline='\n')
    runner=(ROOT/'tools/run_delivery_early_launch_v1.py').read_text(encoding='utf-8')
    runner=runner.replace('early_launch_lifecycle_root_v1','early_launch_lifecycle_root_v2')
    runner=runner.replace('import hashlib\n','import ast\nimport hashlib\n',1)
    runner=runner.replace("sys.path.insert(0,str(OUT))\nsys.path.insert(0,str(ROOT/'tools'))",
        "sys.path.insert(0,str(ROOT/'tools'))\nsys.path.insert(0,str(OUT))")
    runner=runner.replace('import timed_evidence_io_v1','import action_head_runtime_v1\nimport timed_evidence_io_v1',1)
    runner=runner.replace("identity_failure=('PermissionError',4,1),", "identity_failure=('PermissionError',4,1),")
    runner=runner.replace("    for name,(typ,nmaps,nchildren) in expected.items():", "    for name in reg['cases']:\n        typ,nmaps,nchildren=expected[name]")
    path=ROOT/'tools/run_delivery_early_launch_v2.py'
    ast.parse(runner);path.open('x',encoding='utf-8',newline='\n').write(runner)
    supervisor=(ROOT/'tools/supervise_delivery_early_launch_v1.py').read_text(encoding='utf-8')
    supervisor=supervisor.replace('early_launch_lifecycle_root_v1','early_launch_lifecycle_root_v2').replace('run_delivery_early_launch_v1.py','run_delivery_early_launch_v2.py')
    path=ROOT/'tools/supervise_delivery_early_launch_v2.py'
    ast.parse(supervisor);path.open('x',encoding='utf-8',newline='\n').write(supervisor)
    files={str(p.relative_to(ROOT)):sha(p) for p in OUT.glob('*.py')}
    for name in ('prepare_delivery_early_launch_recovery_v2.py','run_delivery_early_launch_v2.py',
                 'supervise_delivery_early_launch_v2.py','pickup_loaded_closure_v1.py'):
        p=ROOT/'tools'/name;files[str(p.relative_to(ROOT))]=sha(p)
    for n in files:ast.parse((ROOT/n).read_text(encoding='utf-8'))
    reg=dict(registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,
        prior_registration_sha256=sha(OLD/'ENGINEERING_REGISTRATION.json'),
        prior_actual_result_sha256=sha(OLD/'RESULT.json'),
        cases=['identity_failure','init_exit','startup_endpoint','receipt_archive_failure'],
        completed_mapping_log_popen_cases_not_replayed=True,
        execution_authorized_for_early_launch_component=True,model_execution_authorized=False,
        early_launch_component_only=True,total_request_seconds=10,child_seconds=8,reap_reserve_seconds=2,
        outer_supervisor_seconds=60,fixture_child_ready_seconds=4,
        parent_v3_bytes_unchanged=sha(OUT/'delivery_phase_shared_parent_v3.py')==sha(OLD/'delivery_phase_shared_parent_v3.py'),
        helper_v2_bytes_unchanged=sha(OUT/'phase_failure_cleanup_v2.py')==sha(OLD/'phase_failure_cleanup_v2.py'),
        revisions=['Fix fixture marker slice to len(marker); no change to parent/model',
            'Explicitly prime observed ntmarta.dll in addition to SHCore.dll; no unique loader-cause claim',
            'Import ast and correct package action_head_runtime before source start; no model/sample warmup'],
        old_QA_repeated=False,full_request_budget_qualified=False,
        full_scientific_loaded_closure_qualified=False,formal_controls_qualified=False,
        scientific_family3_started=False,stable_online_gain=False,remote_upload=False)
    with (OUT/'ENGINEERING_REGISTRATION.json').open('x',encoding='utf-8') as f:json.dump(reg,f,indent=2)
    print('Four previously unreached child boundaries registered; three completed boundaries excluded.')


if __name__=='__main__':main()
