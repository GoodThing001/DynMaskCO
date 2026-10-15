"""Shared bounded worker supervision; no model imports or policy decisions."""
import hashlib
import json
import math
import os
import subprocess
import time
import uuid
from pathlib import Path

from d3_l1_p0c_v18 import ProcessTree
from d3_l1_v112_phase_journal import latest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, record):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    with open(tmp, 'xb') as fh:
        fh.write(json.dumps(record, indent=2, sort_keys=True).encode())
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def identity_check(identity):
    rows = {}
    for path, expected in identity.items():
        try:
            actual = sha(path)
            rows[path] = {'expected': expected, 'actual': actual, 'ok': actual == expected}
        except OSError as exc:
            rows[path] = {'expected': expected, 'ok': False, 'error': repr(exc)}
    return {'rows': rows, 'ok': bool(rows) and all(r['ok'] for r in rows.values())}


def timeline_check(out, started, ended, slot_cap):
    """Audit ALL durable phase records, including phases too short for the polling loop to see."""
    events = [json.loads(p.read_text()) for p in sorted((out / 'phases').glob('phase_*.json'))]
    previous = started
    opened, finished, errors = {}, {}, []
    for i, event in enumerate(events):
        mono = event.get('mono')
        if event.get('sequence') != i or not isinstance(mono, (int, float)) or not math.isfinite(mono):
            errors.append('invalid_phase_identity_or_clock')
            continue
        if not previous <= mono <= ended:
            errors.append('nonmonotonic_or_out_of_window_phase')
        previous = mono
        if event.get('stage') in ('slot', 'slot_finished'):
            key = (event.get('day_id'), event.get('position'))
            if key not in [(d, p) for d in range(16) for p in range(1, 5)]:
                errors.append('phase_slot_out_of_domain')
            if event['stage'] == 'slot':
                if key in opened:
                    errors.append('duplicate_slot_start')
                opened[key] = mono
            else:
                if key not in opened or key in finished:
                    errors.append('orphan_or_duplicate_slot_finish')
                else:
                    finished[key] = {'start': opened[key], 'finish': mono, 'span_s': mono - opened[key]}
                    if mono - opened[key] > slot_cap:
                        errors.append('late_durable_slot_finish')
    return {'n_events': len(events), 'errors': errors, 'opened': [list(k) for k in opened],
            'finished': [{'day_id': k[0], 'position': k[1], **v} for k, v in finished.items()],
            'completed_phase': bool(events) and events[-1]['stage'] == 'completed',
            'ok': not errors}


def run_contained(command, out, identity, *, init_cap=60., slot_cap=10., batch_cap=180., env=None,
                  tail_publisher=None):
    """One process attempt per directory, authentic monotonic deadlines and independent end identity."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    token = out / 'CONTROL_STARTED.json'
    with open(token, 'x', encoding='utf-8') as fh:
        json.dump({'command': command, 'pid': os.getpid(), 'one_attempt': True}, fh)
        fh.flush();os.fsync(fh.fileno())
    started = time.monotonic()
    start_identity = identity_check(identity)
    result = {'start_identity': start_identity, 'command': command,
        'caps': {'initialize': init_cap, 'slot': slot_cap, 'batch': batch_cap},
        'started_mono': started, 'pid': None, 'reason': None, 'cleanup': None,
        'returncode': None, 'tree_active_after': None, 'timeline': None}
    publish(out / 'CONTROL_IDENTITY_START.json', start_identity)
    if not start_identity['ok']:
        result.update(reason='HARD_STOP_IDENTITY', end_identity=identity_check(identity), ok=False)
        publish(out / 'CONTROL_RESULT.json', result)
        return result
    process, tree = None, None
    try:
        with open(out / 'WORKER.stdout', 'wb') as stdout, open(out / 'WORKER.stderr', 'wb') as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env,
                                       start_new_session=(os.name != 'nt'))
            result['pid'] = process.pid
            tree = ProcessTree(process)
            publish(out / 'GO', {'contained': True, 'pid': process.pid})
            while process.poll() is None:
                current = latest(out / 'phases')
                now = time.monotonic()  # includes the bounded phase-read wait
                if now - started > batch_cap:
                    result['reason'] = 'batch_deadline'
                elif not (out / 'READY.json').exists() and now - started > init_cap:
                    result['reason'] = 'initialization_compile_deadline'
                elif current.get('stage') == 'slot' and now - current['mono'] > slot_cap:
                    result['reason'] = 'slot_span_deadline'
                if result['reason']:
                    result['cleanup'] = tree.terminate()
                    break
                time.sleep(.02)
            process.wait(timeout=5)
            if tree.active():
                result['reason'] = result['reason'] or 'descendants_outlived_worker'
                result['cleanup'] = tree.terminate()
            result['tree_active_after'] = tree.active()
            result['returncode'] = process.returncode
            result['worker_ended_mono'] = time.monotonic()
            result['timeline'] = timeline_check(out, started, result['worker_ended_mono'], slot_cap)
            if not result['timeline']['ok']:
                result['reason'] = result['reason'] or 'invalid_or_late_durable_phase'
            if process.returncode:
                result['reason'] = result['reason'] or 'worker_error'
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result['reason'] = result['reason'] or 'supervisor_io_or_protocol_error'
        result['error'] = '%s: %s' % (type(exc).__name__, exc)
    finally:
        if tree is not None:
            if process.poll() is None or tree.active():
                result['cleanup'] = tree.terminate()
            result['tree_active_after'] = tree.active()
            result['returncode'] = process.returncode
            tree.close()
        elif process is not None and process.poll() is None:
            process.kill();process.wait(timeout=5)
            result['returncode'] = process.returncode
    result['end_identity'] = identity_check(identity)
    try:
        if tail_publisher is None:
            publish(out / 'CONTROL_IDENTITY_END.json', result['end_identity'])
        else:
            # V1.18: the trailing identity record may be published to a FRESH destination name so
            # that no fixed name is ever replaced while something might hold it open.  The legacy
            # fixed-name publication above is kept byte-identically for the V1.16 acceptance.
            result['tail_publication'] = tail_publisher(result['end_identity'])
    except OSError as exc:
        result['tail_identity_persist_error'] = repr(exc)
    if not result['end_identity']['ok']:
        result['reason'] = 'HARD_STOP_IDENTITY_END'
    if time.monotonic() - started > batch_cap:
        result['reason'] = result['reason'] or 'batch_deadline'
    result['ok'] = not result['reason'] and result['returncode'] == 0 and result['tree_active_after'] == 0
    result['worker_control_wall_s'] = time.monotonic() - started
    try:
        publish(out / 'CONTROL_RESULT.json', result)
    except OSError as exc:
        result.update(ok=False, reason='supervisor_final_persistence_failed', error=repr(exc))
        publish(out / 'CONTROL_EMERGENCY.json', result)
    # A late final fsync is not a successful run even if the optimistic file was written.
    if time.monotonic() - started > batch_cap:
        result.update(ok=False, reason=result['reason'] or 'batch_deadline_final_persistence')
        publish(out / 'CONTROL_FINAL_BUDGET_FAILURE.json', result)
    return result


def reconcile_rows(out, control, *, verify_record, rows_dir=None, row_glob=None):
    """Each durable row requires a matching durable phase finish and independent verification.
    NOTE: the per-row verdicts below are PROVISIONAL.  Call round_verdict() afterwards -- the round
    is only qualified if the trailing identity record is valid and the control result is ok.

    `rows_dir` selects the directory the row artifacts live in, RELATIVE to `out`.  Its historical
    default is `scores`, which is where the V1.16 harness writes them; passing `'.'` reads rows
    that sit directly in the run root, which is what the V1.11 layout did.

    `row_glob` is a format string with `{day}` and `{pos}` placeholders.  Its default reproduces the
    historical FIXED file name `day%02d_pos%d.json`.  V1.18 passes a glob instead, because a
    uniquely named immutable record (never a fixed name that can be replaced under a reader) cannot
    be located by an exact name.  Exactly one match is required: zero matches is "not attempted"
    and more than one is an error -- a duplicate row record must never be silently merged.
    """
    out = Path(out)
    rows_root = out / (rows_dir if rows_dir is not None else 'scores')
    pattern = row_glob or 'day{day:02d}_pos{pos}.json'
    timeline = control.get('timeline') or {}
    finished = {(r['day_id'], r['position']) for r in timeline.get('finished', [])}
    opened = {tuple(k) for k in timeline.get('opened', [])}
    # If a kill prevented the final timeline audit, durable starts still provide failure accounting.
    if not opened:
        try:
            for p in sorted((out / 'phases').glob('phase_*.json')):
                r = json.loads(p.read_text())
                if r.get('stage') == 'slot':
                    opened.add((r.get('day_id'), r.get('position')))
        except (OSError, ValueError):
            pass
    rows = []
    for day in range(16):
        for pos in range(1, 5):
            key = (day, pos)
            row = {'day_id': day, 'position': pos, 'qualified': False}
            matches = sorted(rows_root.glob(pattern.format(day=day, pos=pos)))
            path = matches[0] if len(matches) == 1 else None
            if len(matches) > 1:
                row.update(status='FAILED_DUPLICATE_ROW_RECORDS',
                           error='%d records match the row pattern' % len(matches),
                           matches=[p.name for p in matches[:8]])
            elif path is None:
                row['status'] = 'FAILED_TERMINATED_MID_SLOT' if key in opened else 'NOT_ATTEMPTED_AFTER_TERMINATION'
            else:
                try:
                    record = json.loads(path.read_text())
                    valid, detail = verify_record(record, out)
                    valid = valid and record.get('day_id') == day and record.get('position') == pos
                    row.update(record=record, verification=detail)
                    row['qualified'] = bool(valid and key in finished)
                    row['status'] = 'DURABLE_ROW_VERIFIED' if row['qualified'] else 'FAILED_ROW_OR_PUBLICATION'
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    row.update(status='FAILED_CORRUPT_ROW', error=repr(exc))
            rows.append(row)
    return rows


def round_verdict(out, control, rows, *, expected_rows=64, tail_path=None):
    """ROUND-LEVEL adjudication with revocation.

    The per-slot PASS files stay on disk as immutable evidence, but they are NOT the qualification.
    The round qualifies only when every expected row is verified AND the trailing identity record
    (CONTROL_IDENTITY_END.json) is present and valid AND the control result is ok.  If the tail
    record is missing/corrupt, or the end identity mismatched, or the row count is incomplete, the
    whole round is REVOKED: every row is downgraded to qualified=False and a durable
    ROUND_REVOKED.json is published, so a tail failure can never leave a passing round behind.
    """
    out = Path(out)
    tail_target = Path(tail_path) if tail_path is not None else (out / 'CONTROL_IDENTITY_END.json')
    report = {'expected_rows': expected_rows,
              'rows_qualified_provisionally': sum(1 for r in rows if r.get('qualified')),
              'tail_path': str(tail_target), 'revoked': False, 'reason': None,
              'control_ok': bool((control or {}).get('ok')),
              'control_reason': (control or {}).get('reason')}
    try:
        tail = json.loads(tail_target.read_text())
        report['tail_present'] = True
        report['tail_ok'] = bool(tail.get('ok')) and bool(tail.get('rows'))
        report['tail_rows'] = sorted(tail.get('rows', {}))
    except (OSError, ValueError) as exc:
        report['tail_present'], report['tail_ok'] = False, False
        report['tail_error'] = repr(exc)
    if not report['tail_ok']:
        report.update(revoked=True, reason='TAIL_IDENTITY_MISSING_OR_INVALID')
    if not report['control_ok']:
        report.update(revoked=True,
                      reason=report['reason'] or ('CONTROL_NOT_OK:%s' % report['control_reason']))
    if report['rows_qualified_provisionally'] != expected_rows:
        report.update(revoked=True, reason=report['reason'] or 'INCOMPLETE_ROWS')
    if report['revoked']:
        rows = [dict(r, qualified=False, revoked=True,
                     status='REVOKED_' + str(r.get('status') or 'UNKNOWN')) for r in rows]
        report['status'] = 'ROUND_REVOKED'
        report['rows_revoked'] = len(rows)
        publish(out / 'ROUND_REVOKED.json', report)
    else:
        report['status'] = 'ROUND_PASS'
        report['rows_qualified'] = report['rows_qualified_provisionally']
        publish(out / 'ROUND_PASS.json', report)
    return report, rows
