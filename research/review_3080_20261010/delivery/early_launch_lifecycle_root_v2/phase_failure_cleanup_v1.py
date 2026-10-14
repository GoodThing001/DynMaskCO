"""Retain original trace only after the launched process handle is reaped.

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
