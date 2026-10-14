"""Clean every acquired startup resource, including failures before Popen.

Only the exact launched Popen handle may be terminated. Worker bytes may be
read only after wait returns; absent, closed or unreaped storage is not read.
Original exceptions and selected outputs remain owned by the caller.
"""
import hashlib
import time
import traceback


def cleanup(child, worker_trace, caller_trace, handles, save_raw,
            start_clock, total_deadline):
    errors = []
    kill_clock = reap_clock = raw_hash = caller_hash = None
    closed = {}

    def perform():
        nonlocal kill_clock, reap_clock, raw_hash
        if child is None:
            return
        if child.poll() is None:
            kill_clock = time.perf_counter()
            child.kill()
        child.wait(timeout=max(.001, total_deadline - time.perf_counter()))
        reap_clock = time.perf_counter()
        if worker_trace is not None and not worker_trace.closed:
            raw = bytes(worker_trace)
            save_raw('worker', raw)
            raw_hash = hashlib.sha256(raw).hexdigest()

    try:
        if caller_trace is None:
            perform()
        else:
            caller_trace.measure(19, 1, perform)
    except BaseException:
        errors.append(traceback.format_exc())
    if caller_trace is not None:
        try:
            caller_trace.close()
            raw = bytes(caller_trace.storage)
            save_raw('caller', raw)
            caller_hash = hashlib.sha256(raw).hexdigest()
        except BaseException:
            errors.append(traceback.format_exc())
        finally:
            try:
                caller_trace.storage.release()
            except BaseException:
                errors.append(traceback.format_exc())
    for name, handle in handles:
        if handle is None:
            closed[name] = None
            continue
        try:
            if not handle.closed:
                handle.close()
            closed[name] = handle.closed
        except BaseException:
            closed[name] = False
            errors.append(traceback.format_exc())
    end = time.perf_counter()
    return dict(start_clock=start_clock, end_clock=end,
                elapsed=None if start_clock is None else end-start_clock,
                kill_clock=kill_clock, reap_clock=reap_clock,
                child_launched=child is not None,
                actual_worker_returncode=None if child is None else child.poll(),
                worker_reaped=reap_clock is not None,
                worker_trace_sha256=raw_hash, caller_trace_sha256=caller_hash,
                handles_closed=closed, cleanup_errors=errors,
                within_declared_window=end <= total_deadline,
                archive_and_handle_cleanup_in_elapsed=True,
                receipt_creation_in_elapsed=False, selected_output_revoked=False,
                full_request_budget_qualified=False)
