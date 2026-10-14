"""Prepared public-snapshot cancellation caller; not a scientific online adapter.

Prepared public bytes and model/library startup are outside the request window.
Bounded caller copy/SHA/JSON/guard/archive are timed but synchronous, so this
source alone cannot establish arbitrary-load or full hard realtime qualification.
"""
import hashlib,json,mmap,struct,subprocess,sys,time,uuid
from pathlib import Path
from bounded_phase_trace_v1 import PhaseTrace,MAX_RECORDS
SIZE=2*2**20;HEADER=64

def request(base,python,arm,prepared_input,incumbent,fault_stage=0):
    import psutil
    from action_head_runtime_v1 import sha
    from timed_evidence_io_v1 import save,save_bytes
    from bounded_phase_trace_v1 import PhaseTrace,MAX_RECORDS
    from phase_failure_cleanup_v2 import cleanup
    base=Path(base);reg=json.loads((base/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
    startup_only=reg.get('early_launch_component_only') is True
    if startup_only:
        if reg.get('execution_authorized_for_early_launch_component') is not True or reg.get('model_execution_authorized') is not False:
            raise RuntimeError('Startup fixture lacks its separate component registration')
        worker_name='early_launch_fixture_worker_v1.py'
        worker_path=Path(reg['fixture_worker_path'])
        if worker_path.name!=worker_name or sha(worker_path)!=reg.get('fixture_worker_sha256'):
            raise RuntimeError('Startup fixture bytes differ from registration')
    else:
        worker_name='delivery_score_split_worker_v1.py'
        worker_path=base/worker_name
    if not startup_only and (reg.get('execution_authorized') is not True or reg.get('independent_timed_request_checker_sealed') is not True):
        raise RuntimeError('No actual execution without separate registration/checker')
    if reg.get('total_request_seconds')!=10 or reg.get('child_seconds')!=8 or reg.get('reap_reserve_seconds')!=2:
        raise ValueError('Fixed10/8/2 engineering contract')
    if not isinstance(prepared_input,bytes) or not 0<len(prepared_input)<=SIZE:raise ValueError('Bounded prepared input')
    nonce='maskco_timed_'+uuid.uuid4().hex;tag=arm+'_fault%d'%fault_stage
    spec=dict(arm=arm,fault_stage=fault_stage,control=nonce+'_c',input=nonce+'_i',output=nonce+'_o',output_directory=tag+'_worker',trace=nonce+'_t')
    specpath=base/(tag+'_SPEC.json');save(base,specpath.name,spec)
    ctrl=inp=out=telemetry=stdout=log=child=None
    child_identity=None;caller_trace=None;failure_traceback=None
    selected=None;reason='no_qualified_response';started=None;init=time.perf_counter()
    writer_clock=None;validation_clock=None;selection_clock=None;kill_clock=None;reap_clock=None
    try:
        from startup_native_identity_v1 import prime_native_identity
        native_dependency_handle=prime_native_identity()
        ctrl=mmap.mmap(-1,128,tagname=spec['control']);inp=mmap.mmap(-1,SIZE,tagname=spec['input']);out=mmap.mmap(-1,SIZE+HEADER,tagname=spec['output'])
        telemetry=mmap.mmap(-1,64+64*MAX_RECORDS,tagname=spec['trace'])
        caller_buffer=bytearray(64+64*MAX_RECORDS)
        log=open(base/(tag+'_stderr.txt'),'xb');stdout=open(base/(tag+'_stdout.txt'),'xb')
        child=subprocess.Popen([str(python),'-B',str(worker_path),str(specpath)],stdout=stdout,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        process=psutil.Process(child.pid)
        child_identity=dict(pid=child.pid,create_time=process.create_time(),cmdline=process.cmdline(),username=process.username())
        while struct.unpack_from('<I',ctrl,12)[0]!=1:
            if child.poll() is not None:raise RuntimeError('Initialization exited: %s'%child.returncode)
            if time.perf_counter()-init>=300:raise TimeoutError('Bounded resident initialization')
            time.sleep(.002)
        if startup_only:
            raise RuntimeError('Registered startup-only endpoint before request')
        initial_identity=json.loads((base/spec['output_directory']/'INITIAL_PARAMETERS.json').read_text(encoding='utf-8'))
        expected_source=initial_identity['source_identity']
        # Input copy and all subsequent caller handling are in this new window.
        started=time.perf_counter();deadline=started+8.;total_deadline=started+10.
        caller_trace=PhaseTrace(caller_buffer,1,MAX_RECORDS)
        inp[:len(prepared_input)]=prepared_input
        struct.pack_into('<I',ctrl,8,len(prepared_input));struct.pack_into('<d',ctrl,16,deadline);struct.pack_into('<Q',ctrl,0,1)
        while time.perf_counter()<deadline:
            if struct.unpack_from('<Q',ctrl,32)[0]==1:
                def validate_select():
                    nonlocal selected,reason,writer_clock,validation_clock,selection_clock
                    seq,size,h,writer_clock=struct.unpack_from('<QI4x32sd',out,0)
                    if seq!=1 or not 0<size<=SIZE:reason='response_header_rejected';return
                    raw=bytes(out[HEADER:HEADER+size])
                    if hashlib.sha256(raw).digest()!=h or struct.unpack_from('<Q',ctrl,32)[0]!=1:
                        reason='response_identity_rejected';return
                    report=json.loads(raw.decode('utf-8'))
                    if report['arm']!=arm or report['key']!=[12,3] or report.get('source_identity')!=expected_source or report.get('source_qualified') is not True or report.get('learned_output_qualified') is not False:
                        reason='response_source_guard_rejected';return
                    candidate=report['candidate'];choose=bool(candidate and candidate['hard'] and candidate['cost']<report['incumbent']['cost'])
                    if not report['incumbent']['hard'] or report['candidate_selected']!=choose or report['selected_routes']!=(report['routes'] if choose else incumbent):
                        reason='response_strict_C0_selection_guard_rejected';return
                    if time.perf_counter()>=deadline or writer_clock>=deadline:
                        reason='validation_missed_child_deadline';return
                    # Selection is only a public-snapshot return value. There is no
                    # live environment/vehicle pointer installation in this draft.
                    validation_clock=time.perf_counter()
                    if validation_clock>=deadline:reason='validation_missed_child_deadline';return
                    selection_clock=time.perf_counter()
                    if selection_clock>=deadline:reason='selection_missed_child_deadline';return
                    selected=report;reason='qualified_response_selected'
                    save_bytes(base/(tag+'_SELECTED_RAW.json'),raw)
                    return
                caller_trace.measure(18,0,validate_select)
                break
            if child.poll() is not None:break
            time.sleep(.001)
        def reap_collect():
            nonlocal kill_clock,reap_clock
            if child.poll() is None:
                # Selected output remains selected if the child stalls after publish.
                while time.perf_counter()<deadline and child.poll() is None:time.sleep(.001)
                if child.poll() is None:
                    struct.pack_into('<I',ctrl,24,1);kill_clock=time.perf_counter();child.kill()
            child.wait(timeout=max(.001,total_deadline-time.perf_counter()))
            reap_clock=time.perf_counter()
            archived_hash=sha(base/(tag+'_SELECTED_RAW.json')) if selected else None
            # Close the caller's mappings and log handles before recording the end.
            # Receipt-file creation itself remains explicitly outside this measurement.
            actual_stage=struct.unpack_from('<I',ctrl,28)[0]
            save_bytes(base/(tag+'_WORKER_TRACE.bin'),bytes(telemetry))
            return archived_hash,actual_stage
        archived_hash,actual_stage=caller_trace.measure(19,0,reap_collect)
        caller_trace.close()
        save_bytes(base/(tag+'_CALLER_TRACE.bin'),bytes(caller_buffer))
        caller_trace.storage.release()
        telemetry.close();stdout.close();log.close();ctrl.close();inp.close();out.close()
        finished=time.perf_counter()
        receipt=dict(arm=arm,fault_stage=fault_stage,start_clock=started,end_clock=finished,elapsed=finished-started,
            selected=selected is not None,reason=reason,selected_routes=selected['selected_routes'] if selected else incumbent,
            raw_identity=archived_hash,
            actual_worker_returncode=child.returncode,natural_exit=kill_clock is None,
            child_pid=child.pid,child_identity=child_identity,worker_reaped=True,within_total_window=finished<=total_deadline,
            actual_stage=actual_stage,total_request_seconds=10,child_seconds=8,reap_reserve_seconds=2,
            writer_clock=writer_clock,validation_clock=validation_clock,selection_clock=selection_clock,
            kill_clock=kill_clock,reap_clock=reap_clock,child_deadline=deadline,total_deadline=total_deadline,
            caller_cleanup_in_elapsed=True,telemetry_collection_archive_in_elapsed=True,receipt_file_creation_in_elapsed=False,
            prepared_public_input_only=True,live_installation=False,formal_controls_qualified=False,
            full_request_budget_qualified=False,scientific_family3_started=False,stable_online_gain=False)
        save(base,tag+'_CALLER_RECEIPT.json',receipt);return receipt
    except BaseException:
        import traceback
        failure_traceback=traceback.format_exc()
        raise
    finally:
        if failure_traceback is not None or any(handle is not None and not handle.closed for handle in (stdout,log,ctrl,inp,out,telemetry)):
            from phase_failure_cleanup_v2 import cleanup
            # This path does not replace an already-selected return value.
            def retain(kind,raw):
                suffix='_WORKER_TRACE.bin' if kind=='worker' else '_CALLER_TRACE.bin'
                path=base/(tag+suffix)
                if path.exists():
                    # Partial original archive is retained; never overwritten.
                    path=base/(tag+'_EXCEPTION_'+kind.upper()+'_TRACE.bin')
                save_bytes(path,raw)
            stop=started+10. if started is not None else time.perf_counter()+2.
            final=cleanup(child,telemetry,caller_trace,(('stdout',stdout),('stderr',log),('control',ctrl),('input',inp),('output',out),('telemetry',telemetry)),retain,started,stop)
            final.update(traceback=failure_traceback,child_identity=child_identity,
                selected=selected is not None,selection_clock=selection_clock,
                qualification=False,initialization_outside_request=started is None)
            pending=sys.exception()
            try:
                save(base,tag+'_CALLER_FAILURE_AFTER_CLEANUP.json',final)
            except BaseException as archive_error:
                if failure_traceback is None:
                    raise
                # Preserve the original startup exception even if the disk fails.
                if pending is not None:
                    pending.add_note('Cleanup receipt failed: '+repr(archive_error))
