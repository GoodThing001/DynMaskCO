"""Seal new cleanup source and a bounded trace-only plan, before data creation."""
import ast,hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'research/review_3080_20261010'
PRE=R/'delivery/phase_timing_wiring_preparation_root_v1'
OUT=R/'delivery/phase_cleanup_boundary_engineering_root_v1'

HELPER='''"""Retain original trace only after the launched process handle is reaped.

This component does not certify arbitrary-load timing or model/scientific state.
Selected output is never revoked by this cleanup helper.
"""
import hashlib,time,traceback

def cleanup(child,worker_trace,caller_trace,handles,save_raw,start_clock,total_deadline):
    errors=[];kill_clock=None;reap_clock=None;raw_hash=None;caller_hash=None
    def perform():
        nonlocal kill_clock,reap_clock,raw_hash
        if child.poll() is None:
            kill_clock=time.perf_counter();child.kill()
        # The Popen process handle identifies our launched child even if a PID
        # is later reused. Never look up an unrelated process to terminate it.
        child.wait(timeout=max(.001,total_deadline-time.perf_counter()))
        reap_clock=time.perf_counter()
        raw=bytes(worker_trace)
        save_raw('worker',raw);raw_hash=hashlib.sha256(raw).hexdigest()
    try:
        if caller_trace is None:perform()
        else:caller_trace.measure(19,1,perform)
    except BaseException:
        errors.append(traceback.format_exc())
    # If wait failed, no raw worker read is permitted. Retain any caller trace
    # and record the failure; do not label worker exited/reaped or invent bytes.
    try:
        if caller_trace is not None:
            caller_trace.close()
            raw=bytes(caller_trace.storage)
            save_raw('caller',raw);caller_hash=hashlib.sha256(raw).hexdigest()
            caller_trace.storage.release()
    except BaseException:errors.append(traceback.format_exc())
    for handle in handles:
        try:
            if handle is not None and not handle.closed:handle.close()
        except BaseException:errors.append(traceback.format_exc())
    end=time.perf_counter()
    return dict(start_clock=start_clock,end_clock=end,
        elapsed=None if start_clock is None else end-start_clock,
        kill_clock=kill_clock,reap_clock=reap_clock,
        actual_worker_returncode=child.poll(),worker_reaped=reap_clock is not None,
        worker_trace_sha256=raw_hash,caller_trace_sha256=caller_hash,
        cleanup_errors=errors,within_declared_window=end<=total_deadline,
        archive_and_handle_cleanup_in_elapsed=True,receipt_creation_in_elapsed=False,
        selected_output_revoked=False,full_request_budget_qualified=False)
'''

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def create(n,v):
    p=OUT/n
    if isinstance(v,str):
        if p.suffix=='.py':ast.parse(v)
        with p.open('x',encoding='utf-8',newline='\n') as f:f.write(v)
    else:
        with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)

def main():
    OUT.mkdir(exist_ok=False)
    prior=json.loads((PRE/'PREPARATION_REGISTRATION.json').read_text(encoding='utf-8'))
    for n,h in prior['package_files'].items():
        assert sha(PRE/n)==h
        shutil.copy2(PRE/n,OUT/n)
    create('phase_failure_cleanup_v1.py',HELPER)
    parent=(PRE/'delivery_phase_shared_parent_v1.py').read_text(encoding='utf-8')
    parent=parent.replace('    selected=None;reason=', '    caller_trace=None;failure_traceback=None\n    selected=None;reason=',1)
    cut=parent.index('    except Exception:\n        import traceback')
    parent=parent[:cut]+'''    except BaseException:
        import traceback
        failure_traceback=traceback.format_exc()
        raise
    finally:
        if any(not handle.closed for handle in (stdout,log,ctrl,inp,out,telemetry)):
            from phase_failure_cleanup_v1 import cleanup
            # This path does not replace an already-selected return value.
            def retain(kind,raw):
                suffix='_WORKER_TRACE.bin' if kind=='worker' else '_CALLER_TRACE.bin'
                path=base/(tag+suffix)
                if path.exists():
                    # Partial original archive is retained; never overwritten.
                    path=base/(tag+'_EXCEPTION_'+kind.upper()+'_TRACE.bin')
                save_bytes(path,raw)
            stop=started+10. if started is not None else time.perf_counter()+2.
            final=cleanup(child,telemetry,caller_trace,(stdout,log,ctrl,inp,out,telemetry),retain,started,stop)
            final.update(traceback=failure_traceback,child_identity=child_identity,
                selected=selected is not None,selection_clock=selection_clock,
                qualification=False,initialization_outside_request=started is None)
            save(base,tag+'_CALLER_FAILURE_AFTER_CLEANUP.json',final)
'''
    # Include helper before initial SOURCE_START closure; no lazy helper source.
    parent=parent.replace('    from bounded_phase_trace_v1 import PhaseTrace,MAX_RECORDS',
        '    from bounded_phase_trace_v1 import PhaseTrace,MAX_RECORDS\n    from phase_failure_cleanup_v1 import cleanup',1)
    create('delivery_phase_shared_parent_v2.py',parent)
    create('ENGINEERING_REGISTRATION.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        kind='Bounded trace and exception cleanup only, no actual model request',
        preparation_tool_sha256=sha(Path(__file__)),
        prior_registration_sha256=sha(PRE/'PREPARATION_REGISTRATION.json'),
        package_files={p.name:sha(p) for p in sorted(OUT.iterdir())},
        cases=['nested_success','recorded_exception','capacity_overflow','actual_open_writer_kill_and_cleanup'],
        validator_mutations=['magic','version','capacity','count','closed','header_padding','sequence','phase','status','record_padding','nonfinite_clock','clock_order','unused_tail'],
        writer_ready_limit_seconds=5,kill_reap_and_archive_limit_seconds=2,
        outer_supervisor_seconds=60,maximum_buffers=4,
        allowed_imports=['bounded_phase_trace_v1','independent_phase_trace_parser_v1','phase_failure_cleanup_v1','stdlib','psutil'],
        model_engine_worker_parent_not_imported=True,old_QA_repeated=False,
        execution_authorized_for_trace_only=True,model_execution_authorized=False,
        full_request_budget_qualified=False,full_scientific_loaded_closure_qualified=False,family3_started=False,
        source_only_remaining=['Parent process creation/identity outside try still needs separate review; copied model parent is not runtime qualified',
            'Full new request replay and exact model registration/supervision/transport/resource owner checks']))
    print('Trace-only preregistration sealed; model/parent workload remains disabled.')
if __name__=='__main__':main()
