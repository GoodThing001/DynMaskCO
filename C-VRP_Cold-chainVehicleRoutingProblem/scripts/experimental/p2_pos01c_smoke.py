"""One-shot, fixed-identity P2-C CPU diagnostic. No training or production install.

The generic collect_real_labels guard remains locked. This entry can only consume
the R0 roots and the explicit user-approved contract below, once per run directory.
All computations run in a reclaimable Windows job; only the supervisor writes.
"""
from __future__ import annotations

import argparse
import ctypes as C
from ctypes import wintypes as W
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
RUN_ID = 'p2_pos01c_smoke_20261019_v1'
OUT = ROOT / 'research' / RUN_ID
R0 = ROOT / 'configs/p2_pos01c_r0_20261019_v1'
GIB = 1024**3
MAX_PACKET = 8*1024**2
MAX_OUTPUT = 256*1024**2

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
os.environ.update(CUDA_VISIBLE_DEVICES='-1', JAX_PLATFORMS='cpu',
                  JAX_PLATFORM_NAME='cpu', JAX_ENABLE_COMPILATION_CACHE='false',
                  XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1',
                  PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def plain(x):
    if is_dataclass(x):
        return plain(asdict(x))
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [plain(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return {'nonfinite_float': repr(x)}
    if hasattr(x, 'tolist'):
        return plain(x.tolist())
    return x


def encoded(x):
    return json.dumps(plain(x), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(',', ':')).encode('utf-8')


class Store:
    def __init__(self, root, cap=MAX_OUTPUT):
        self.root, self.cap = Path(root), cap

    def used(self):
        return sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())

    def write(self, name, data, final=False):
        target = (self.root/name).resolve()
        if not target.is_relative_to(self.root.resolve()) or target.exists():
            raise RuntimeError('unsafe or repeated output path')
        # Reserve 8 MiB for supervisor's final failure receipt and summary.
        limit = self.cap if final else self.cap-8*1024**2
        if self.used()+len(data) > limit:
            raise RuntimeError('OUTPUT_LIMIT; no partial write')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f:
            f.write(data)


class IO(C.Structure):
    _fields_ = [(n, C.c_ulonglong) for n in ('r', 'w', 'o', 'rb', 'wb', 'ob')]


class Basic(C.Structure):
    _fields_ = [('process_time', C.c_longlong), ('job_time', C.c_longlong),
                ('flags', W.DWORD), ('min_ws', C.c_size_t), ('max_ws', C.c_size_t),
                ('active_processes', W.DWORD), ('affinity', C.c_size_t),
                ('priority', W.DWORD), ('scheduling', W.DWORD)]


class Extended(C.Structure):
    _fields_ = [('basic', Basic), ('io', IO), ('process_memory', C.c_size_t),
                ('job_memory', C.c_size_t), ('peak_process', C.c_size_t), ('peak_job', C.c_size_t)]


class ProcessMemory(C.Structure):
    _fields_ = [('cb', W.DWORD), ('faults', W.DWORD)]+[
        (n, C.c_size_t) for n in ('peak_ws', 'ws', 'peak_paged', 'paged',
                                'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile', 'private')]


class Job:
    def __init__(self, limit=7*GIB):
        self.k = C.WinDLL('kernel32', use_last_error=True)
        self.k.CreateJobObjectW.restype = W.HANDLE
        self.k.CreateJobObjectW.argtypes = (C.c_void_p, W.LPCWSTR)
        self.k.SetInformationJobObject.argtypes = (W.HANDLE, C.c_int, C.c_void_p, W.DWORD)
        self.k.AssignProcessToJobObject.argtypes = (W.HANDLE, W.HANDLE)
        self.k.OpenProcess.argtypes = (W.DWORD, W.BOOL, W.DWORD)
        self.k.OpenProcess.restype = W.HANDLE
        self.k.QueryInformationJobObject.argtypes = (W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.c_void_p)
        self.k.TerminateJobObject.argtypes = (W.HANDLE, W.UINT)
        self.k.CloseHandle.argtypes = (W.HANDLE,)
        self.handle = self.k.CreateJobObjectW(None, None)
        if not self.handle:
            raise C.WinError(C.get_last_error())
        info = Extended()
        info.basic.flags = 0x200 | 0x2000  # aggregate committed memory + kill tree on close
        info.job_memory = limit
        if not self.k.SetInformationJobObject(self.handle, 9, C.byref(info), C.sizeof(info)):
            raise C.WinError(C.get_last_error())

    def assign(self, pid):
        process = self.k.OpenProcess(0x100 | 1 | 0x1000, False, pid)
        if not process:
            raise C.WinError(C.get_last_error())
        try:
            if not self.k.AssignProcessToJobObject(self.handle, process):
                raise C.WinError(C.get_last_error())
        finally:
            self.k.CloseHandle(process)

    def peak(self):
        info = Extended()
        if not self.k.QueryInformationJobObject(self.handle, 9, C.byref(info), C.sizeof(info), None):
            raise C.WinError(C.get_last_error())
        return int(info.peak_job)

    def close(self):
        if self.handle:
            self.k.TerminateJobObject(self.handle, 99)
            self.k.CloseHandle(self.handle)
            self.handle = None


def supervisor_memory():
    k = C.WinDLL('kernel32', use_last_error=True)
    k.GetCurrentProcess.restype = W.HANDLE
    ps = C.WinDLL('psapi', use_last_error=True)
    ps.GetProcessMemoryInfo.argtypes = (W.HANDLE, C.c_void_p, W.DWORD)
    info = ProcessMemory()
    info.cb = C.sizeof(info)
    if not ps.GetProcessMemoryInfo(k.GetCurrentProcess(), C.byref(info), info.cb):
        raise C.WinError(C.get_last_error())
    return max(info.ws, info.private)


def send(conn, kind, payload=None, raw=b''):
    packet = encoded({'kind': kind, 'payload': payload})+b'\n'+raw
    if len(packet) > MAX_PACKET:
        raise RuntimeError('IPC_PACKET_LIMIT')
    conn.send_bytes(packet)


def receive(conn):
    packet = conn.recv_bytes(MAX_PACKET)
    header, raw = packet.split(b'\n', 1)
    obj = json.loads(header)
    return obj['kind'], obj['payload'], raw


def worker(conn, gate):
    gate.wait()  # no imports/computation/descendants until assigned to the job
    sink = open(os.devnull, 'w')
    sys.stdout = sys.stderr = sink
    state = {}
    while True:
        try:
            kind, command, _ = receive(conn)
            if command['op'] == 'probe-memory':
                try:
                    bytearray(128*1024**2)
                    result = {'allocation_denied': False}
                except MemoryError:
                    result = {'allocation_denied': True}
            elif command['op'] == 'probe-timeout':
                time.sleep(10)
                result = {}
            else:
                result = compute(command, state, conn)
            send(conn, 'RESULT', result)
        except EOFError:
            return
        except BaseException as exc:
            send(conn, 'ERROR', {'error': type(exc).__name__+': '+str(exc),
                                 'traceback': traceback.format_exc()[-16000:]})
            return


def save_raw(conn, name, value):
    send(conn, 'SAVE', name, pickle.dumps(value, protocol=5))
    kind, _, _ = receive(conn)
    if kind != 'ACK':
        raise RuntimeError('supervisor refused raw artifact')


def imports():
    sys.path.insert(0, str(ROOT/'scripts'))
    from project_paths import configure_imports
    configure_imports(('experimental', 'models', 'training', 'simulation', 'evaluation', 'coldchain', 'tests'))
    import numpy as np
    import p2_pos01_teacher as t
    import p2_pos01_reference as r
    return np, t, r


def compute(cmd, state, conn):
    np, t, r = imports()
    op = cmd['op']
    if op == 'identity':
        import p2_pos01c_r0_prepare as prep
        reg, manifest = prep.verify()
        from scenario_saa import make_c0_contract_v2, build_history
        hist = []
        for day in manifest['days']:
            if day['role'] == 'HIST_TRAIN':
                with np.load(ROOT/day['relative_path'], allow_pickle=False) as z:
                    hist.extend(build_history({k: z[k] for k in z.files}))
        spec = t.SaaSpec(contract=make_c0_contract_v2(), history=tuple(tuple(d) for d in hist),
            booking_horizon=16., budget=186.06639500089943, capacity=50., tw_speed=1.5,
            K=4, n_neighbors=10, time_limit=6., cooling_share=4.452,
            reject_penalty=(5., 10., 15.), stream_id=RUN_ID, stream_seed=2026101903,
            shadow_mode='ls', energy_pricing='marginal', standby_orders=8.)
        assert spec.contract.contract_hash == reg['contract_hash']
        assert len(spec.history) == 16
        state.update(spec=spec, manifest=manifest, captures=[], prepared={}, direct={})
        try:
            t.collect_real_labels()
        except PermissionError:
            locked = True
        else:
            raise RuntimeError('generic real collector protection was removed')
        return {'spec_signature': spec.signature, 'contract_hash': spec.contract.contract_hash,
                'generic_real_collector_locked': locked, 'history_days': 16,
                'SaaSpec': {k: plain(v) for k, v in asdict(spec).items() if k not in ('history', 'contract')}}
    if op == 'initialize':
        from p2_pos01_mask_value import FrozenBackbone, MaskValueModel
        import jax
        backbone = FrozenBackbone(WORKSPACE/'MASKCO_code/ckpts/cvrp100.ckpt')
        assert backbone.manifest['parameter_sha256'] == '7b43ece321619b7cea8d88b35c3053e8e6cd82f40f968631380adef4a0f88a2e'
        state.update(backbone=backbone, model=MaskValueModel(backbone))
        return {'backbone': backbone.manifest, 'devices': [d.platform for d in jax.devices()],
                'head': 'UNTRAINED_ZERO_OUTPUT_NO_OPTIMIZER', 'python': sys.version,
                'numpy': np.__version__, 'jax': jax.__version__}
    if op == 'preflight':
        import test_p2_pos01_reference as tests
        from dataclasses import replace
        source, spec = tests.saa_synthetic_dataset()
        spec = replace(spec, cooling_share=4.452)
        captures, *_ = tests.run_actual_saa(source, spec, True)
        capture = captures[0]
        t.validate_saa_capture(capture, spec)
        actions = t.prepare_mask_actions(capture, spec, state['backbone'])
        assert len({r.plan_hash(a['plan']) for a in actions if a['status'] == 'OK'}) >= 2
        tapes = t.make_future_tapes(capture, actions, spec, K=2)
        keep = t.run_branch(capture, None, tapes[0], spec)
        assert keep['status'] == 'OK'
        _, traces, served, policy, ev = tests.run_actual_saa(tapes[0]['dataset'], spec)
        assert keep['trace_identity'] == t.snapshot_state_hash({'traces': [asdict(x) for x in traces]})
        assert keep['evaluation'] == ev and keep['ledger'] == policy.export_state()
        tests.test_saa_hidden_future_cannot_change_current_state_or_action(state['backbone'])
        tests.test_saa_broken_ledger_prefix_and_cutoff_rejected((capture, actions, tapes, spec, {}))
        exception = t.run_branch(capture, actions[0], tapes[0], spec, synthetic_fault='raise_after_install')
        timeout = t.run_branch(capture, actions[0], tapes[0], spec, seconds=5., synthetic_fault='hang_after_install')
        assert exception['status'] == 'EXCEPTION' and exception['utility'] is None
        assert timeout['status'] == 'ROLLOUT_TIMEOUT' and timeout['worker_reclaimed']
        assert state['backbone'].parameter_digest() == state['backbone'].manifest['parameter_sha256']
        return {'status': 'PASS', 'checks': ['synthetic_full_KEEP_direct_parity', 'hidden_future_invariance',
            'ledger_prefix_cutoff_rejection', 'conservative_capacity_boundary', 'nondegenerate_repairs',
            'exception_and_timeout_reclaimed', 'frozen_weights'],
            'synthetic_continuations': 3, 'no_real_source_executed': True}
    if op == 'capture':
        descriptor = [d for d in state['manifest']['days'] if d['role'] == 'SOURCE_TRAIN'][cmd['day']]
        with np.load(ROOT/descriptor['relative_path'], allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        assert t.dataset_digest(data) == descriptor['dataset_sha256']
        spec = state['spec']
        policy = t.PublicSaaReplanner(spec)
        env = t.SaaPhaseEnv(data, 50., 1.5, num_vehicles=15, replanner=policy,
                            coldchain_contract=spec.contract, booking_horizon=16.)
        captured, seen = [], set()
        class Captured(Exception):
            pass
        def hook(engine, view, context):
            masks = r.mask_pool(view, r.plans_from_view(view), seed=0)
            identity = r.identity(view)
            if masks and identity not in seen:
                cap = t.capture_saa_phase(engine, view, context, spec)
                t.validate_saa_capture(cap, spec)
                state['captures'].append(cap)
                index = len(state['captures'])-1
                captured.append({'index': index, 'clock': view.clock, 'capture_identity': cap['saa_identity'],
                    'public_identity': cap['public_identity'], 'day_id': descriptor['day_id'],
                    'legal_masks': masks, 'accepted': view.accepted, 'rejected': view.rejected,
                    'served': view.served, 'public_view': view})
                seen.add(identity)
                if len(captured) == 2:
                    raise Captured()
        env.p2_capture = hook
        try:
            env.run(0)
        except Captured:
            pass
        return {'day_id': descriptor['day_id'], 'states': captured,
                'SAA_timeouts': policy.timeouts, 'admission_checks': policy.admission_checks,
                'selection_rule': 'first two distinct post-admission public states with a nonempty legal mask pool; never utility-based'}
    if op == 'prepare':
        index, spec = cmd['state'], state['spec']
        capture = state['captures'][index]
        view = capture['public_view']
        base = r.plans_from_view(view)
        masks = r.mask_pool(view, base, seed=0)[:3]
        scores = state['model'].score_masks(view, base, masks)
        assert np.isfinite(scores).all() and (scores == 0).all()
        actions = t.prepare_mask_actions(capture, spec, state['backbone'], masks=masks, seconds=10.)
        # Exact public projection check after changing every still-hidden row.
        import copy
        from strict_online_env import StrictOnlineEnv
        policy = t.PublicSaaReplanner(spec)
        policy.restore_state(capture['snapshot']['replanner_state'])
        vehicles, _, served = t.restore_recourse_snapshot(capture['snapshot'])
        source = [d for d in state['manifest']['days'] if d['dataset_sha256'] == capture['dataset_identity']][0]
        with np.load(ROOT/source['relative_path'], allow_pickle=False) as z:
            changed = {k: z[k].copy() for k in z.files}
        hidden = changed['reveal_time'][0] > view.clock+r.EVENT_TOL
        changed['coords'][0, hidden] = 123.
        changed['demands'][0, hidden] = 19.
        changed['temp_class'][0, hidden] = 2
        changed['reveal_time'][0, hidden] = max(spec.booking_horizon+10., view.clock+10.)
        e = StrictOnlineEnv(changed, 50., 1.5, num_vehicles=15, coldchain_contract=spec.contract, booking_horizon=16.)
        proxy, _ = r.saa_public_env(e, 0, view.clock)
        other = r.view_from_saa(proxy, policy, view.clock, vehicles, served, view.replan_ids)
        assert r.identity(other) == capture['public_identity'], 'hidden future changed current public input'
        assert tuple(r.wait_or_return(other, v) for v in other.vehicles) == tuple(r.wait_or_return(view, v) for v in view.vehicles)
        np.testing.assert_array_equal(scores, state['model'].score_masks(other, base, masks))
        assert [a['action_identity'] for a in actions] == [t._action_identity(a) for a in actions]
        tapes = t.make_future_tapes(capture, actions, spec, K=2)  # only AFTER all actions frozen
        state['prepared'][index] = (capture, actions, tapes)
        save_raw(conn, 'private/state_%d.pickle' % index, {'capture': capture, 'actions': actions, 'tapes': tapes})
        return {'state_index': index, 'legal_mask_pool': r.mask_pool(view, base), 'attempted_masks': masks,
            'scores': scores, 'actions': actions, 'keep_plan': base,
            'unique_accepted_repair_plans': len({r.plan_hash(a['plan']) for a in actions if a['status'] == 'OK'}),
            'tapes': [{k: v for k, v in tape.items() if k != 'dataset'} for tape in tapes],
            'current_public_invariance': 'PASS', 'phase': t.PHASE}
    if op == 'branch':
        index, k, action_id = cmd['state'], cmd['future'], cmd['action']
        capture, actions, tapes = state['prepared'][index]
        spec, tape = state['spec'], tapes[k]
        if action_id == 'DIRECT':
            # Native continuation, independent of the mask-install/snapshot helper.
            import copy
            from strict_online_env import StrictOnlineEnv
            from coldchain_evaluator_a1 import evaluate_trace_a1
            snap = copy.deepcopy(capture['snapshot'])
            snap['customer_universe'] = np.flatnonzero(tape['dataset']['demands'][0] > 0).astype(np.int32)
            snap['state_hash'] = t.snapshot_state_hash(snap)
            policy = t.PublicSaaReplanner(spec)
            env = StrictOnlineEnv(copy.deepcopy(tape['dataset']), 50., 1.5, num_vehicles=15,
                replanner=policy, coldchain_contract=spec.contract, booking_horizon=16.)
            t.validate_snapshot_against_env(env, snap)
            traces, served = env.run_resumed(snap)
            ev = evaluate_trace_a1(traces, {k: a[0] for k, a in tape['dataset'].items()},
                spec.contract, policy._accepted, policy._rejected, spec.budget, dict(enumerate(spec.reject_penalty)))
            row = {'status': 'DIRECT_OK', 'evaluation': ev, 'ledger': policy.export_state(),
                'trace_identity': t.snapshot_state_hash({'traces': [asdict(v) for v in traces]}),
                'scenario_identity': tape['scenario_identity'], 'utility': ev['utility'] if ev['hard_feasible'] else None}
            state['direct'][(index, k)] = row
        else:
            action = None if action_id == 'KEEP' else actions[int(action_id)]
            row = t.run_branch(capture, action, tape, spec, seconds=54.)
            if action_id == 'KEEP':
                direct = state['direct'][(index, k)]
                assert row.get('trace_identity') == direct['trace_identity'], 'KEEP/direct trace mismatch'
                assert row.get('ledger') == direct['ledger'], 'KEEP/direct ledger mismatch'
                assert plain(row.get('evaluation')) == plain(direct['evaluation']), 'KEEP/direct outcome mismatch'
                assert row['first_entry']['admission_count'] == capture['snapshot']['replanner_state']['p2_saa']['admission_count']
                row['KEEP_direct_parity'] = 'PASS'
        assert state['backbone'].parameter_digest() == state['backbone'].manifest['parameter_sha256']
        return row
    raise ValueError('unknown operation')


class Session:
    def __init__(self, store, deadline, memory=7*GIB):
        self.store, self.deadline = store, deadline
        self.ctx = mp.get_context('spawn')
        self.conn, child = self.ctx.Pipe()
        gate = self.ctx.Event()
        self.job = Job(memory)
        self.process = self.ctx.Process(target=worker, args=(child, gate))
        self.process.start()
        child.close()
        try:
            self.job.assign(self.process.pid)
        except BaseException:
            self.process.terminate()
            self.process.join(2)
            self.job.close()
            raise
        gate.set()
        self.peaks = {'job_committed_bytes': 0, 'supervisor_bytes': 0}

    def call(self, command, seconds):
        started = time.monotonic()
        # Termination reserve is included in the slot and global 90-minute envelope.
        cutoff = min(started+seconds-2., self.deadline-5.)
        send(self.conn, 'COMMAND', command)
        while time.monotonic() < cutoff:
            self.peaks['job_committed_bytes'] = max(self.peaks['job_committed_bytes'], self.job.peak())
            self.peaks['supervisor_bytes'] = max(self.peaks['supervisor_bytes'], supervisor_memory())
            if self.peaks['supervisor_bytes'] > 512*1024**2:
                raise RuntimeError('SUPERVISOR_MEMORY_LIMIT')
            if self.store.used() > MAX_OUTPUT-8*1024**2:
                raise RuntimeError('OUTPUT_LIMIT')
            if self.conn.poll(.05):
                kind, result, raw = receive(self.conn)
                if kind == 'SAVE':
                    self.store.write(result, raw)
                    send(self.conn, 'ACK')
                    continue
                if kind == 'ERROR':
                    raise RuntimeError(json.dumps(result, ensure_ascii=False))
                if kind != 'RESULT':
                    raise RuntimeError('bad worker receipt')
                return result
            if not self.process.is_alive():
                raise RuntimeError('worker exited without receipt')
        raise TimeoutError('slot or global deadline reached')

    def close(self):
        self.job.close()  # kills worker AND every continuation descendant
        self.process.join(1.)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(.5)
        self.conn.close()
        if self.process.is_alive():
            raise RuntimeError('job could not be reclaimed')


def sources():
    # Broad Python closure; includes the new entry, existing algorithms, upstream code.
    files = []
    for base in (ROOT/'scripts', WORKSPACE/'MASKCO_code/models', WORKSPACE/'MASKCO_code/modules',
                 WORKSPACE/'MASKCO_code/training', WORKSPACE/'MASKCO_code/helpers'):
        files.extend(base.rglob('*.py'))
    return {p.relative_to(WORKSPACE).as_posix(): sha(p) for p in sorted(set(files))}


def execute():
    OUT.mkdir(parents=True, exist_ok=False)  # the public fixed run-id is single-use
    store = Store(OUT)
    start = time.monotonic()
    deadline = start+5400.
    registration = {'run_id': RUN_ID, 'user_authorised': True, 'qualification': 'REFERENCE_ONLY',
        'baseline_head': '72d7e000783c0ecabe03890a3cc04a4d51c68ef5',
        'R0_sha256': {p.name: sha(p) for p in R0.iterdir() if p.is_file()},
        'source_sha256': sources(), 'checkpoint_sha256': sha(WORKSPACE/'MASKCO_code/ckpts/cvrp100.ckpt'),
        'B_ref_kwh': 186.06639500089943, 'cooling_share': 4.452, 'reject_penalty': [5, 10, 15],
        'SAA': {'K': 4, 'event_time_limit_s': 6, 'n_neighbors': 10, 'shadow_mode': 'ls',
                'energy_pricing': 'marginal', 'standby_orders': 8, 'future_policy': 'reveal', 'incr_eval': False,
                'stream_id': RUN_ID, 'stream_seed': 2026101903},
        'operating': {'capacity': 50., 'margin': 1e-4, 'vehicles': 15, 'N': 50,
                      'booking_horizon_h': 16., 'depot_deadline_h': 22., 'tw_speed': 1.5},
        'selection': 'first two distinct legal-mask post-admission states per registered source day',
        'mask_selection': 'first up to 3 of existing deduplicated mask_pool(seed=0), never label-dependent',
        'queue': {'state_slots': 4, 'future_K': 2, 'actions_per_future': ['DIRECT', 'KEEP', '0', '1', '2'],
                  'complete_continuation_cap': 40, 'DIRECT_is_in_cap': True},
        'resource': {'CPU_workers': 1, 'BLAS_threads': 1, 'GPU': False,
            'RAM_bytes': 8*GIB, 'worker_tree_job_hard_bytes': 7*GIB,
            'supervisor_stop_bytes': 512*1024**2, 'reserve_bytes': 512*1024**2,
            'output_hard_bytes': MAX_OUTPUT, 'max_IPC_bytes': MAX_PACKET,
            'planned_s': 3900, 'hard_s': 5400, 'initialization_s': 300, 'synthetic_preflight_s': 300,
            'source_day_s': 120, 'state_prepare_s': 120, 'continuation_s': 60,
            'completion_budget': '300+300+2*120+4*120+40*60+180=3900 seconds; no extra slots'},
        'stop': 'any identity, projection, KEEP, capacity, supervision failure => BLOCKED; no repair/retry, seed/B changes or training'}
    store.write('registration.json', encoded(registration))
    rows = [{'state_slot': s, 'future': k, 'action': a, 'status': 'NOT_ATTEMPTED', 'utility': None}
            for s in range(4) for k in range(2) for a in ('DIRECT', 'KEEP', '0', '1', '2')]
    result = {'status': 'BLOCKED', 'qualification': 'REFERENCE_ONLY', 'phase': 'PREFLIGHT',
              'states': [], 'days': [], 'rows': rows, 'labels': [], 'commands': [], 'source_days_attempted': 0}
    session = None
    try:
        # Supervisor self-checks use no data, no model, and expected faults only.
        test_dir = OUT/'output_probe'
        test_dir.mkdir()
        tiny = Store(test_dir, cap=9*1024**2)
        try:
            tiny.write('refused.bin', b'x'*(1024**2+1))
        except RuntimeError:
            assert not (test_dir/'refused.bin').exists()
        else:
            raise RuntimeError('output guard failed')
        probe = Session(store, deadline, memory=64*1024**2)
        try:
            assert probe.call({'op': 'probe-memory'}, 10)['allocation_denied']
        finally:
            probe.close()
        probe = Session(store, deadline)
        try:
            try:
                probe.call({'op': 'probe-timeout'}, 3)
            except TimeoutError:
                pass
            else:
                raise RuntimeError('timeout guard failed')
        finally:
            probe.close()
        result['supervision_preflight'] = 'PASS: memory allocation denied; timeout tree reclaimed; output refusal atomic'
        session = Session(store, deadline)
        def call(command, seconds):
            t0 = time.monotonic()
            answer = session.call(command, seconds)
            result['commands'].append({'command': command, 'elapsed_s': time.monotonic()-t0})
            return answer
        result['identity'] = call({'op': 'identity'}, 60)
        result['model'] = call({'op': 'initialize'}, 300)
        result['synthetic_preflight'] = call({'op': 'preflight'}, 300)
        result['phase'] = 'SOURCE_CAPTURE'
        for day in range(2):
            result['source_days_attempted'] += 1
            day_record = call({'op': 'capture', 'day': day}, 120)
            result['days'].append(day_record)
            store.write('day_%d.json' % day, encoded(day_record))
        for day, day_record in enumerate(result['days']):
            for ordinal in range(2):
                slot = 2*day+ordinal
                matching = [row for row in rows if row['state_slot'] == slot]
                if ordinal >= len(day_record['states']):
                    for row in matching:
                        row['status'] = 'NO_STATE'
                    continue
                capture_record = day_record['states'][ordinal]
                index = capture_record['index']
                result['phase'] = 'PREPARE_STATE_%d' % slot
                prepared = call({'op': 'prepare', 'state': index}, 120)
                prepared.update(state_slot=slot, day_id=day_record['day_id'])
                result['states'].append(prepared)
                store.write('state_%d.json' % slot, encoded(prepared))
                for row in matching:
                    if row['action'] not in ('DIRECT', 'KEEP') and int(row['action']) >= len(prepared['actions']):
                        row['status'] = 'NO_MASK'
                        continue
                    result['phase'] = 'CONTINUATION_%d_%d_%s' % (slot, row['future'], row['action'])
                    command = {'op': 'branch', 'state': index, 'future': row['future'], 'action': row['action']}
                    row.update(status='RUNNING', attempted=True)
                    outcome = call(command, 60)
                    row.update(outcome)
                    store.write('row_%d_%d_%s.json' % (slot, row['future'], row['action']), encoded(row))
        if sources() != registration['source_sha256'] or any(sha(R0/name) != h for name, h in registration['R0_sha256'].items()):
            raise RuntimeError('source or R0 identity changed during run')
        result.update(status='COMPLETE', phase='FINISHED')
    except BaseException as exc:
        result['stop_reason'] = type(exc).__name__+': '+str(exc)
        result['traceback'] = traceback.format_exc()[-16000:]
        for row in rows:
            if row['status'] == 'RUNNING':
                row.update(status='ROLLOUT_TIMEOUT' if isinstance(exc, TimeoutError) else 'BLOCKED',
                           failures=[result['stop_reason']], utility=None)
    finally:
        if session is not None:
            result['resource_peaks'] = session.peaks
            try:
                session.close()
            except BaseException as exc:
                result.update(status='BLOCKED', supervision_cleanup_error=str(exc))
        # All registered candidate/state slots remain in the denominator.
        for s in range(4):
            for a in ('0', '1', '2'):
                pair = []
                for k in range(2):
                    candidate = next(r for r in rows if (r['state_slot'], r['future'], r['action']) == (s, k, a))
                    keep = next(r for r in rows if (r['state_slot'], r['future'], r['action']) == (s, k, 'KEEP'))
                    valid = candidate['status'] == keep['status'] == 'OK' and candidate.get('utility') is not None and keep.get('utility') is not None
                    pair.append({'valid': valid, 'delta': candidate['utility']-keep['utility'] if valid else None,
                                 'candidate_status': candidate['status'], 'keep_status': keep['status']})
                valid = all(p['valid'] for p in pair)
                result['labels'].append({'state_slot': s, 'action': a, 'valid': valid,
                    'delta_utility': math.fsum(p['delta'] for p in pair)/2 if valid else None, 'paired': pair})
        result.update(elapsed_s=time.monotonic()-start, output_bytes_before_final=store.used(),
                      training_steps=0, full_certified=False, final_test_used=False)
        store.write('result.json', encoded(result), final=True)
        print(json.dumps({'status': result['status'], 'phase': result['phase'], 'stop_reason': result.get('stop_reason'),
                          'source_days_attempted': result['source_days_attempted'], 'elapsed_s': result['elapsed_s']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-authorised-fixed-run', action='store_true', required=True)
    parser.parse_args()
    execute()
