"""Build a new parent from sealed bytes; never change the old failure tree."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'research/review_3080_20261010/delivery/phase_cleanup_boundary_engineering_root_v1'
OUT = ROOT/'research/review_3080_20261010/delivery/early_launch_lifecycle_root_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create(name, value):
    if name.endswith('.py'):
        ast.parse(value)
    with (OUT/name).open('x', encoding='utf-8', newline='\n') as f:
        f.write(value)


def main():
    reg = json.loads((OLD/'ENGINEERING_REGISTRATION.json').read_text(encoding='utf-8'))
    original = {}
    for name, expected in reg['package_files'].items():
        if name.endswith('.py') or name == 'PHASE_DESIGN.json':
            assert sha(OLD/name) == expected
            original[name] = expected
            if name == 'delivery_phase_shared_parent_v2.py':
                continue
            with (OUT/name).open('xb') as f:
                f.write((OLD/name).read_bytes())
    parent = (OLD/'delivery_phase_shared_parent_v2.py').read_text(encoding='utf-8')
    parent = parent.replace('from phase_failure_cleanup_v1 import cleanup',
                            'from phase_failure_cleanup_v2 import cleanup')
    start = parent.index("    ctrl=mmap.mmap(-1,128")
    stop = parent.index("        while struct.unpack_from('<I',ctrl,12)[0]!=1:")
    allocations = parent[start:parent.index('    caller_trace=None;failure_traceback=None', start)]
    initialization = '''    ctrl=inp=out=telemetry=stdout=log=child=None
    child_identity=None;caller_trace=None;failure_traceback=None
    selected=None;reason='no_qualified_response';started=None;init=time.perf_counter()
    writer_clock=None;validation_clock=None;selection_clock=None;kill_clock=None;reap_clock=None
    try:
        from startup_native_identity_v1 import prime_native_identity
        native_dependency_handle=prime_native_identity()
'''
    parent = parent[:start]+initialization+''.join('    '+line+'\n' for line in allocations.splitlines())+parent[stop:]
    parent = parent.replace("if any(not handle.closed for handle in (stdout,log,ctrl,inp,out,telemetry)):",
                            "if failure_traceback is not None or any(handle is not None and not handle.closed for handle in (stdout,log,ctrl,inp,out,telemetry)):")
    parent = parent.replace("(stdout,log,ctrl,inp,out,telemetry),retain,started,stop)",
                            "(('stdout',stdout),('stderr',log),('control',ctrl),('input',inp),('output',out),('telemetry',telemetry)),retain,started,stop)")
    parent = parent.replace("            save(base,tag+'_CALLER_FAILURE_AFTER_CLEANUP.json',final)",
                            """            try:
                save(base,tag+'_CALLER_FAILURE_AFTER_CLEANUP.json',final)
            except BaseException as archive_error:
                if failure_traceback is None:
                    raise
                # Preserve the original startup exception even if the disk fails.
                pending=sys.exception()
                if pending is not None:
                    pending.add_note('Cleanup receipt failed: '+repr(archive_error))""")
    create('delivery_phase_shared_parent_v3.py', parent)
    worker = (OLD/'delivery_phase_shared_worker_v1.py').read_text(encoding='utf-8')
    worker = worker.replace('    engine=InitialEngine(BASE,reg,spec[\'arm\'])',
                            "    from startup_native_identity_v1 import prime_native_identity\n    native_dependency_handle=prime_native_identity()\n    engine=InitialEngine(BASE,reg,spec['arm'])", 1)
    # The actual model worker is prepared only and never used by this local stage.
    create('delivery_phase_shared_worker_v2.py', worker)
    parent_path=OUT/'delivery_phase_shared_parent_v3.py'
    parent_text=parent_path.read_text(encoding='utf-8').replace("base/'delivery_phase_shared_worker_v1.py'", "base/'delivery_phase_shared_worker_v2.py'")
    parent_path.write_text(parent_text,encoding='utf-8',newline='\n')
    manifest=dict(old_registered_source_files=original,
                  old_registration_sha256=sha(OLD/'ENGINEERING_REGISTRATION.json'),
                  new_parent_sha256=sha(parent_path),
                  engine_bytes_unchanged=sha(OUT/'delivery_phase_actual_engine_v1.py')==sha(OLD/'delivery_phase_actual_engine_v1.py'),
                  actual_model_execution_authorized=False,
                  full_request_budget_qualified=False,scientific_family3_started=False)
    create('PREPARATION_IDENTITY.json',json.dumps(manifest,indent=2)+'\n')
    print('New early-launch lifecycle source prepared; model execution stays disabled.')


if __name__ == '__main__':
    main()
