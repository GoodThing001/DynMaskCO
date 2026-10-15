"""V1.8 supervisor; V1.9 adds scoped human-authorization consumption for one pass."""
import ctypes
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

SEEDS = ('d3_l1_collection_runner', 'd3_l1_causal_perturbation', 'd3_l1_bitwise_parity',
         'd3_l1_state_collector', 'd3_l1_p0c_v18')
FROZEN_PARAMETERS = {'capacity': 50.0, 'num_vehicles': 25, 'tw_speed': 1.0,
                     'booking_horizon': 24.0, 'K': 4, 'days': 16, 'denominator': 64,
                     'slot_cap_s': 60.0, 'day_rollout_cap_s': 120.0, 'batch_cap_s': 3600.0,
                     'model_identity': 'F_FREE_NO_MODEL', 'scalar_encoding': 'lossless_scalar_v18'}


def write_json(R, path, value):
    R.atomic_write_bytes(path, json.dumps(value, indent=2, sort_keys=True).encode())


def runtime_inventory(R):
    """Static lazy-import closure PLUS actual loaded local modules, with resolved origins."""
    mods, edges, external = R.closure(SEEDS, runtime=True)
    for name, module in list(sys.modules.items()):
        origin = getattr(getattr(module, '__spec__', None), 'origin', None)
        if not origin or not Path(origin).is_file():
            continue
        p = Path(origin).resolve()
        if p.is_relative_to(R.EXT / 'scripts') and name != '__main__':
            rec = mods.setdefault(name, {'path': p.relative_to(R.EXT).as_posix(),
                                         'sha256_at_scan': R.sha_file(p)})
            rec.update(runtime_file=str(p), runtime_sha256=R.sha_file(p),
                       runtime_matches_scan=R.sha_file(p) == rec['sha256_at_scan'])
    result, errors = {}, []
    for name, rec in sorted(mods.items()):
        origin = rec.get('runtime_file')
        path = (Path(origin).resolve() if origin else None)
        expected = (R.EXT / rec['path']).resolve()
        if path != expected or not rec.get('runtime_matches_scan'):
            errors.append({'module': name, 'record': rec})
        module = sys.modules.get(name)
        spec = getattr(getattr(module, '__spec__', None), 'origin', None)
        if not spec or Path(spec).resolve() != expected:
            errors.append({'module': name, 'spec_origin': spec})
        result[name] = {'path': rec['path'], 'sha256': rec.get('runtime_sha256'),
                        'resolved_origin': str(path), 'spec_origin': spec}
    R.assert_f_free()
    return {'modules': result, 'errors': errors, 'edges': edges, 'external': external}


def execution_identity(R, args):
    from coldchain_contract import ObjectiveProfile, apply_objective_profile, default_pilot_contract
    profile_path = R.EXT / 'results/o0cc/scale_v2/objective_profile.json'
    doc = json.loads(profile_path.read_text(encoding='utf-8'))
    fields = {f.name for f in dataclasses.fields(ObjectiveProfile)}
    if set(doc) - fields - {'profile_hash', 'provenance', 'hard_gate'}:
        raise ValueError('unrecognised profile fields')
    profile = ObjectiveProfile(**{k: v for k, v in doc.items() if k in fields})
    base = default_pilot_contract()
    eff = apply_objective_profile(base, profile)
    inv = runtime_inventory(R)
    frozen = json.loads(Path(args.identity).read_text(encoding='utf-8'))['records']['identity']
    frozen_checks = {
        'historical_identity': R.sha_file(args.identity) ==
        'f44e3456db5fac67c85829e5f28648174b4c50b81d1af85a288dc78a7795172c',
        'base': base.contract_hash == frozen['l1_base_contract_sha256'],
        'effective': eff.contract_hash == frozen['l1_effective_contract_sha256'],
        'profile_file': R.sha_file(profile_path) == frozen['profile_file_sha256'],
        'profile_hash': profile.profile_hash == frozen['objective_profile_sha256']
        and doc['profile_hash'] == profile.profile_hash,
        'dataset': R.sha_file(args.dataset) == R.DEV_CHECK_SHA,
    }
    # No historical per-file SHA is silently rewritten: new SHA identity is separate.
    params = dict(FROZEN_PARAMETERS, slot_cap_s=args.slot_cap_s,
                  day_rollout_cap_s=args.day_rollout_cap_s, batch_cap_s=args.batch_cap_s)
    versions = {}
    for name in ('numpy', 'jax', 'jaxlib', 'flax'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'schema': 'p0c_execution_identity_v18', 'dataset_sha256': R.sha_file(args.dataset),
            'teacher_manifest_sha256': R.sha_file(Path(args.dataset).parent / 'M0_TEACHER_MANIFEST.json'),
            'profile_file_sha256': R.sha_file(profile_path), 'profile_hash': profile.profile_hash,
            'base_contract_hash': base.contract_hash, 'effective_contract_hash': eff.contract_hash,
            'historical_identity_sha256': R.sha_file(args.identity),
            'runner_sha256': R.sha_file(R.HERE / 'd3_l1_collection_runner.py'),
            'parameters': params, 'python': sys.version, 'packages': versions,
            'platform': sys.platform, 'modules': inv['modules'], 'origin_errors': inv['errors'],
            'frozen_checks': frozen_checks}


def comparable(identity):
    return {k: ({n: {f: v[f] for f in ('path', 'sha256')} for n, v in value.items()}
                 if k == 'modules' else value) for k, value in identity.items()}


def authorization_errors(R, args):
    """Formal execution requires the recorded human instruction and a pinned scope.

    This checks an explicit input artifact; it never manufactures user approval.
    The supervisor consumes its one-pass token, while workers only verify it.
    """
    errors = []
    path = getattr(args, 'authorization', None)
    expected = getattr(args, 'expect_authorization_sha256', None)
    if not path or not expected or R.sha_file(path) != expected:
        return ['FORMAL_AUTHORIZATION_MISSING_OR_CHANGED']
    try:
        auth = json.loads(Path(path).read_text(encoding='utf-8'))
        if auth.get('authority') != 'human_user_conditional_instruction':
            errors.append('FORMAL_AUTHORITY_NOT_HUMAN_INSTRUCTION')
        if not auth.get('user_quote') or not auth.get('prior_request_scope'):
            errors.append('FORMAL_AUTHORIZATION_PROVENANCE_MISSING')
        if auth.get('authorized_action') != 'ONE_F_FREE_16_DAY_4_SLOT_COLLECTION':
            errors.append('FORMAL_AUTHORIZATION_SCOPE_MISMATCH')
        if auth.get('dataset_sha256') != R.DEV_CHECK_SHA or auth.get('parameters') != FROZEN_PARAMETERS:
            errors.append('FORMAL_AUTHORIZATION_FROZEN_PARAMETERS_MISMATCH')
        if auth.get('platform') != sys.platform or auth.get('python') != sys.version:
            errors.append('FORMAL_AUTHORIZATION_RUNTIME_MISMATCH')
        if auth.get('runtime_identity_sha256') != R.sha_file(args.runtime_identity):
            errors.append('FORMAL_AUTHORIZATION_SOURCE_IDENTITY_MISMATCH')
        if Path(auth['output_directory']).resolve() != Path(args.outdir).resolve():
            errors.append('FORMAL_AUTHORIZATION_OUTPUT_MISMATCH')
        for review in auth.get('required_reviews', []):
            review_path = Path(path).parent / review['path']
            if R.sha_file(review_path) != review['sha256']:
                errors.append('FORMAL_AUTHORIZATION_REVIEW_CHANGED')
            else:
                reviewed = json.loads(review_path.read_text(encoding='utf-8'))
                key, expected_value = review['pass_key'], review['pass_value']
                if key not in ('n_fail', 'ok') or type(reviewed.get(key)) is not type(expected_value) \
                        or reviewed.get(key) != expected_value:
                    errors.append('FORMAL_AUTHORIZATION_REVIEW_NOT_PASSED')
        if len(auth.get('required_reviews', [])) < 2:
            errors.append('FORMAL_AUTHORIZATION_REVIEW_MISSING')
        if args.days != 16 or not 0 <= args.day_id < 16:
            errors.append('FORMAL_AUTHORIZATION_DAY_SCOPE_MISMATCH')
        if args.fault != 'none' or args.inject_block_s != 0:
            errors.append('FORMAL_AUTHORIZATION_FAULT_INJECTION_FORBIDDEN')
    except (ValueError, OSError, KeyError, TypeError) as exc:
        errors.append('FORMAL_AUTHORIZATION_UNREADABLE:%s' % type(exc).__name__)
    return errors


def identity_gate(R, args, end=False):
    try:
        expected = json.loads(Path(args.runtime_identity).read_text(encoding='utf-8'))
        actual = execution_identity(R, args)
        mismatches = [k for k in comparable(actual)
                      if comparable(actual)[k] != comparable(expected).get(k)]
        if actual['origin_errors'] or not all(actual['frozen_checks'].values()):
            mismatches.append('frozen_or_origin_check')
        if actual['teacher_manifest_sha256'] != 'b84e171676f8afc02e5ac416538965dda224671d719863a1412f659464bcaf04':
            mismatches.append('teacher_manifest')
        caps = (args.slot_cap_s, args.day_rollout_cap_s, args.batch_cap_s)
        if args.purpose != 'synthetic' and caps != (60.0, 120.0, 3600.0):
            mismatches.append('frozen_caps')
        if args.expect_dataset_sha256 != R.DEV_CHECK_SHA:
            mismatches.append('dataset_override')
        for attr, field in (('expect_runner_sha256', 'runner_sha256'),
                            ('expect_profile_sha256', 'profile_file_sha256'),
                            ('expect_effective_contract', 'effective_contract_hash')):
            if getattr(args, attr, None) is not None and getattr(args, attr) != actual[field]:
                mismatches.append(attr)
        if getattr(args, 'day_cap_s', 120.0) != 120.0 and args.purpose != 'synthetic':
            mismatches.append('obsolete_day_cap_override')
        if args.purpose == 'engineering' and (args.days != 1 or args.day_id != 0):
            mismatches.append('engineering_dev_day0_only')
        if args.purpose == 'formal':
            mismatches.extend(authorization_errors(R, args))
        return {'ok': not mismatches, 'mismatches': mismatches, 'identity': actual,
                'runtime_identity_sha256': R.sha_file(args.runtime_identity), 'end_check': end}
    except Exception as exc:
        return {'ok': False, 'mismatches': ['missing_or_unreadable_identity'],
                'error': '%s: %s' % (type(exc).__name__, exc), 'end_check': end}


class ProcessTree:
    """Windows job containment / POSIX private process group; bounded kill and reap."""
    def __init__(self, process):
        self.process = process
        self.handle = None
        if os.name == 'nt':
            from ctypes import wintypes
            self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
            self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            self.kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                            ctypes.c_void_p, wintypes.DWORD]
            self.kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                             ctypes.c_void_p, wintypes.DWORD,
                                                             ctypes.c_void_p]
            self.handle = self.kernel.CreateJobObjectW(None, None)
            class BasicLimits(ctypes.Structure):
                _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                            ('flags', wintypes.DWORD), ('min_ws', ctypes.c_size_t),
                            ('max_ws', ctypes.c_size_t), ('active_limit', wintypes.DWORD),
                            ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD),
                            ('scheduling', wintypes.DWORD)]
            class ExtendedLimits(ctypes.Structure):
                _fields_ = [('basic', BasicLimits), ('io', ctypes.c_uint64 * 6),
                            ('memory', ctypes.c_size_t * 4)]
            limits = ExtendedLimits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits),
                                                       ctypes.sizeof(limits)):
                process.kill()
                process.wait(timeout=5)
                raise OSError('cannot set job kill-on-close containment')
            if not self.handle or not self.kernel.AssignProcessToJobObject(
                    self.handle, wintypes.HANDLE(int(process._handle))):
                process.kill()
                process.wait(timeout=5)
                raise OSError(ctypes.get_last_error(), 'cannot contain worker in Windows job')

    def active(self):
        if os.name == 'nt':
            class Accounting(ctypes.Structure):
                _fields_ = [('times', ctypes.c_int64 * 4), ('faults', ctypes.c_uint32),
                            ('total', ctypes.c_uint32), ('active', ctypes.c_uint32),
                            ('terminated', ctypes.c_uint32)]
            row = Accounting()
            if not self.kernel.QueryInformationJobObject(self.handle, 1, ctypes.byref(row),
                                                         ctypes.sizeof(row), None):
                raise OSError('cannot verify job process count')
            return int(row.active)
        try:
            os.killpg(self.process.pid, 0)
            return 1
        except ProcessLookupError:
            return 0

    def terminate(self):
        started = time.monotonic()
        before = self.active()
        if os.name == 'nt':
            if not self.kernel.TerminateJobObject(self.handle, 137):
                raise OSError('TerminateJobObject failed')
        else:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.process.wait(timeout=5)
        while self.active() and time.monotonic() - started < 5:
            time.sleep(0.01)
        after = self.active()
        return {'pid': self.process.pid, 'tree_active_before': before, 'tree_active_after': after,
                'returncode': self.process.returncode, 'kill_reap_seconds': time.monotonic() - started,
                'mechanism': 'TerminateJobObject' if os.name == 'nt' else 'killpg(SIGKILL)',
                'cleanup_ok': after == 0}

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def finish_day(R, ledger_path, day, args, reason):
    """Supervisor owns terminal accounting even when capture was never entered."""
    rows = R.read_ledger(ledger_path)
    for pos in range(1, 5):
        event, rec = R.slot_state(rows, day, pos)
        complete = any(r.get('event') == 'SLOT_FINISHED' and r.get('day_id') == day
                       and r.get('position') == pos for r in rows)
        if event == 'ATTEMPT_STARTED' or (event == 'PUBLISHED' and not complete):
            R.append_jsonl(ledger_path, {'event': 'FAILED', 'day_id': day, 'position': pos,
                                         'run_id': args.run_id, 'reason': 'terminated_mid_slot',
                                         'termination_reason': reason})
        elif event is None:
            R.append_jsonl(ledger_path, {'event': 'NOT_ATTEMPTED_DUE_TO_TERMINATION',
                                         'day_id': day, 'position': pos, 'run_id': args.run_id,
                                         'reason': reason})
    R.append_jsonl(ledger_path, {'event': 'DAY_TERMINATED', 'day_id': day,
                                 'run_id': args.run_id, 'reason': reason})


def day_blocked(R, rows, day):
    return any(r.get('day_id') == day and r.get('event') in
               ('DAY_TERMINATED', 'DAY_COMPLETED', 'DAY_WORKER_STARTED') for r in rows)


def worker_command(R, args, day, go):
    cmd = [sys.executable, str(R.HERE / 'd3_l1_collection_runner.py'), '--role', 'worker',
           '--outdir', str(Path(args.outdir)), '--run-id', args.run_id, '--day-id', str(day),
           '--days', str(args.days if args.purpose == 'formal' else 1),
           '--dataset', args.dataset, '--identity', args.identity,
           '--runtime-identity', args.runtime_identity, '--purpose', args.purpose,
           '--supervisor-go', str(go), '--fault', args.fault,
           '--inject-block-s', str(args.inject_block_s)]
    if args.purpose == 'formal':
        cmd.extend(['--authorization', args.authorization, '--expect-authorization-sha256',
                    args.expect_authorization_sha256])
    return cmd


def claim_authorization(R, args, gate):
    token = Path(args.authorization).with_suffix('.CONSUMED.json')
    with open(token, 'x', encoding='utf-8') as fh:
        json.dump({'run_id': args.run_id, 'output_directory': str(Path(args.outdir).resolve()),
                   'authorization_sha256': R.sha_file(args.authorization),
                   'runtime_identity_sha256': gate['runtime_identity_sha256'],
                   'started_at_utc_epoch': time.time(), 'one_pass_only': True}, fh,
                  indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())


def supervise(R, args, synthetic_command=None):
    """An OS-owned lock prevents competing supervisors from spending the same pass."""
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'SUPERVISOR.lock', 'a+b') as lock:
        lock.seek(0)
        lock.write(b'0')
        lock.flush()
        lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            try:
                return _supervise(R, args, synthetic_command)
            except OSError as exc:
                # Any child has been reaped by the inner loop before its final writes.
                # If the result disk fails, retain an emergency failure when possible;
                # stderr and a nonzero exit remain authoritative if that disk also fails.
                failure = {'verdict': 'SUPERVISOR_PERSISTENCE_FAILED', 'run_id': args.run_id,
                           'slots_in_denominator': 64, 'error': '%s: %s' % (type(exc).__name__, exc)}
                print(json.dumps(failure), file=sys.stderr, flush=True)
                for path in (out / 'SUPERVISOR_RESULT.json',
                             out / 'runs' / args.run_id / 'SUPERVISOR_EMERGENCY.json'):
                    try:
                        write_json(R, path, failure)
                    except OSError:
                        pass
                return 1
        finally:
            lock.seek(0)
            if os.name == 'nt':
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def _supervise(R, args, synthetic_command=None):
    """Same loop for production and isolated synthetic acceptance; never runs a 64-slot batch."""
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if args.purpose == 'formal' and synthetic_command is not None:
        raise ValueError('formal collection cannot use a synthetic command')
    start = time.monotonic()
    ledger = out / 'STATE_LEDGER.jsonl'
    run_archive = out / 'runs' / args.run_id
    run_archive.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        write_json(R, run_archive / name, value)
        write_json(R, out / name, value)  # convenience view; immutable run copy retained
    gate = identity_gate(R, args)
    save('SUPERVISOR_IDENTITY_START.json', gate)
    if not gate['ok']:
        save('SUPERVISOR_RESULT.json', {'verdict': 'HARD_STOP_IDENTITY',
                   'identity': gate, 'workers_started': 0, 'slots_in_denominator': 64})
        return 2
    if args.purpose == 'formal':
        # This token is global to the authorization, not to a changeable run_id/outdir.
        try:
            claim_authorization(R, args, gate)
        except FileExistsError:
            save('SUPERVISOR_RESULT.json', {'verdict': 'HARD_STOP_AUTHORIZATION_ALREADY_CONSUMED',
                  'workers_started': 0, 'slots_in_denominator': 64})
            return 2
    results, stop = [], None
    existing = R.read_ledger(ledger)
    if R.ledger_integrity(existing)[0]:
        for day in range(args.days):
            if (any(r.get('day_id') == day and r.get('event') == 'DAY_WORKER_STARTED' for r in existing)
                    and not any(r.get('day_id') == day and r.get('event') in
                                ('DAY_TERMINATED', 'DAY_COMPLETED') for r in existing)):
                finish_day(R, ledger, day, args, 'previous_supervisor_interrupted')
    for day in range(args.days):
        rows = R.read_ledger(ledger)
        if not R.ledger_integrity(rows)[0]:
            stop = 'LEDGER_CORRUPT'
            break
        if day_blocked(R, rows, day):
            results.append({'day_id': day, 'status': 'SKIPPED_DAY_ALREADY_STARTED'})
            continue
        if time.monotonic() - start >= args.batch_cap_s:
            stop = 'batch_deadline'
            break
        # Recheck before EVERY worker (including the runtime manifest's own bytes).
        pre = identity_gate(R, args)
        if not pre['ok'] or pre['runtime_identity_sha256'] != gate['runtime_identity_sha256']:
            stop = 'HARD_STOP_IDENTITY'
            break
        go = out / ('GO_DAY%02d' % day)
        cmd = synthetic_command(day, go) if synthetic_command else worker_command(R, args, day, go)
        R.append_jsonl(ledger, {'event': 'DAY_WORKER_STARTED', 'day_id': day,
                                'run_id': args.run_id, 'command': cmd})
        status, cleanup = 'worker_error', None
        worker_start = time.monotonic()
        tree, process = None, None
        stdout_path = out / ('WORKER_DAY%02d.stdout' % day)
        stderr_path = out / ('WORKER_DAY%02d.stderr' % day)
        try:
            with open(stdout_path, 'wb') as stdout, open(stderr_path, 'wb') as stderr:
                process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr,
                                           start_new_session=(os.name != 'nt'))
                tree = ProcessTree(process)
                R.atomic_write_bytes(go, b'contained')  # child may now run/spawn
                while True:
                    now = time.monotonic()
                    rows = R.read_ledger(ledger)
                    if not R.ledger_integrity(rows)[0]:
                        status, stop = 'ledger_corrupt', 'LEDGER_CORRUPT'
                        break
                    if now - start >= args.batch_cap_s:
                        status, stop = 'batch_deadline_killed', 'batch_deadline'
                        break
                    started = [r for r in rows if r.get('event') == 'DAY_ROLLOUT_STARTED'
                               and r.get('day_id') == day]
                    finished = [r for r in rows if r.get('event') == 'DAY_ROLLOUT_FINISHED'
                                and r.get('day_id') == day]
                    rollout_age = ((finished[-1]['mono'] if finished else now) - started[-1]['mono']
                                   if started else 0)
                    slow = [s for s in R.open_slots(rows, now, args.slot_cap_s, monotonic=True)
                            if s[0][0] == day]
                    # Completed-but-late records cannot bypass a hard deadline between polls.
                    late = [r for r in rows if r.get('day_id') == day
                            and r.get('slot_span_wall_s', 0) > args.slot_cap_s]
                    if rollout_age > args.day_rollout_cap_s:
                        status = 'rollout_deadline_killed'
                        break
                    if slow or late:
                        status = 'slot_deadline_killed'
                        break
                    if now - worker_start > args.day_rollout_cap_s + args.batch_cap_s:
                        status = 'lifetime_guard_killed'
                        break
                    if process.poll() is not None:
                        # A zero exit is not evidence of publication or identity completion.
                        stat = out / ('WORKER_IDENTITY_END_DAY%02d.json' % day)
                        end = json.loads(stat.read_text(encoding='utf-8')) if stat.exists() else {}
                        terminals = [R.slot_state(rows, day, pos)[0] for pos in range(1, 5)]
                        durable = {r.get('position') for r in rows if r.get('day_id') == day
                                   and r.get('event') == 'SLOT_FINISHED'}
                        status = ('completed' if process.returncode == 0 and end.get('ok')
                                  and all(e == 'PUBLISHED' for e in terminals)
                                  and durable == {1, 2, 3, 4} else 'worker_error')
                        if process.returncode == 2 or (end and not end.get('ok')):
                            status, stop = 'identity_mismatch', 'HARD_STOP_IDENTITY'
                        break
                    time.sleep(0.02)
                if process.poll() is None or tree.active():
                    cleanup = tree.terminate()
                    if not cleanup['cleanup_ok']:
                        stop = 'PROCESS_CLEANUP_FAILED'
        except BaseException as exc:
            status = 'supervisor_exception'
            cleanup = {'exception': '%s: %s' % (type(exc).__name__, exc)}
            if tree is not None:
                cleanup['termination'] = tree.terminate()
            elif process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
        finally:
            if tree:
                tree.close()
        if status != 'completed':
            R.append_jsonl(ledger, {'event': status.upper(), 'day_id': day,
                                    'run_id': args.run_id, 'pid': process.pid if process else None})
            finish_day(R, ledger, day, args, status)
        else:
            R.append_jsonl(ledger, {'event': 'DAY_COMPLETED', 'day_id': day, 'run_id': args.run_id})
        results.append({'day_id': day, 'status': status, 'pid': process.pid if process else None,
                        'returncode': process.returncode if process else None,
                        'seconds': time.monotonic() - worker_start, 'cleanup': cleanup})
        save('WORKER_DAY%02d.json' % day, results[-1])
        if stop:
            break
    if stop:
        for day in range(args.days):
            if not day_blocked(R, R.read_ledger(ledger), day):
                finish_day(R, ledger, day, args, stop)
    end = identity_gate(R, args, end=True)
    save('SUPERVISOR_IDENTITY_END.json', end)
    if not end['ok'] or end.get('runtime_identity_sha256') != gate['runtime_identity_sha256']:
        stop = 'HARD_STOP_IDENTITY'
    rows = R.read_ledger(ledger)
    # Publication rows are never erased on a later audit/identity failure.
    published = [r for r in rows if r.get('event') == 'PUBLISHED'
                 and R.slot_state(rows, r['day_id'], r['position'])[0] == 'PUBLISHED'
                 and any(q.get('event') == 'SLOT_FINISHED' and q.get('day_id') == r['day_id']
                         and q.get('position') == r['position'] for q in rows)]
    failed = (stop or any(r['status'] not in ('completed', 'SKIPPED_DAY_ALREADY_STARTED') for r in results)
              or any(r.get('event') in ('FAILED', 'NOT_ATTEMPTED_DUE_TO_TERMINATION',
                                        'DAY_TERMINATED') for r in rows))
    result = {'run_id': args.run_id, 'purpose': args.purpose,
              'verdict': stop or ('BATCH_FAILED' if failed else 'BATCH_COMPLETE'),
              'slots_published': len(published), 'slots_in_denominator': 64,
              'days': results, 'batch_wall_s': time.monotonic() - start,
              'caps': {'rollout': args.day_rollout_cap_s, 'slot': args.slot_cap_s,
                       'batch': args.batch_cap_s}, 'source_stable': end['ok'],
              'model_identity': 'F_FREE_NO_MODEL',
              'formal_slots_collected': len(published) if args.purpose == 'formal' else 0,
              'authorization_sha256': R.sha_file(args.authorization) if args.purpose == 'formal' else None}
    result['source_stability'] = {
        name: {'sha256_at_start': row['sha256'],
               'sha256_at_end': end.get('identity', {}).get('modules', {}).get(name, {}).get('sha256'),
               'source_stable': row['sha256'] == end.get('identity', {}).get('modules', {}).get(name, {}).get('sha256')}
        for name, row in gate['identity']['modules'].items()}
    save('SUPERVISOR_RESULT.json', result)
    # Final persistence itself belongs to the batch window.
    if time.monotonic() - start > args.batch_cap_s and result['verdict'] == 'BATCH_COMPLETE':
        result.update(verdict='batch_deadline_after_persistence',
                      batch_wall_s=time.monotonic() - start)
        save('SUPERVISOR_RESULT.json', result)
    return 1 if result['verdict'] != 'BATCH_COMPLETE' else 0
