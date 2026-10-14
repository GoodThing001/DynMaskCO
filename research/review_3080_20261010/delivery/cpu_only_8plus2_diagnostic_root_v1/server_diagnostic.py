"""CPU-only, no-publication 8 s + 2 s engineering diagnostic (runs on the 3080 host).

This script is the *executor* of a pre-registered protocol.  It never publishes
anything, never writes into a scientific result tree, and never converts a failure
into a pass: it writes exactly one of

    DIAG_STATE.json      (complete run; gate_pass computed from the declared gates)
    DIAG_FAILURE.json    (raised before completion; traceback + gate_pass=false)

The CPU environment is locked BEFORE any JAX import.  The closure verdict comes from
`unified_closure_protocol`, i.e. the exact module that `local_unified_protocol_test.py`
verified LOCALLY (6 pass / 0 fail); no second copy of the gating logic exists here.

Declared unit of work: one model chain = encode_cvrp + decode_edge_logits on a fixed
51-node instance of the delivery contract.  This is a *model-level* unit.  It is NOT
the complete delivery request, which stays unqualified.
"""
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. lock the CPU environment BEFORE importing jax / anything numpy-heavy
# ---------------------------------------------------------------------------
PACKAGE = Path(__file__).resolve().parent


def _find_root(start):
    """Repo root = nearest ancestor (or self) holding MASKCO_code/."""
    for candidate in (start, *start.parents):
        if (candidate / 'MASKCO_code').is_dir():
            return candidate
    raise RuntimeError('repo root with MASKCO_code/ not found above %s' % start)


ROOT = _find_root(PACKAGE)
OUT = Path(os.environ['DIAG_OUT'])
CORE = int(os.environ.get('DIAG_CORE', '15'))
BUDGET_S = float(os.environ.get('DIAG_BUDGET_S', '8.0'))
RESERVE_S = float(os.environ.get('DIAG_RESERVE_S', '2.0'))
RAM_GATE_GIB = float(os.environ.get('DIAG_RAM_GATE_GIB', '8.0'))
CORE_IDLE_GATE = float(os.environ.get('DIAG_CORE_IDLE_GATE', '25.0'))

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

cpu_lock = {}
try:
    os.sched_setaffinity(0, {CORE})
    cpu_lock['affinity'] = sorted(os.sched_getaffinity(0))
    cpu_lock['affinity_enforced'] = cpu_lock['affinity'] == [CORE]
except Exception as exc:                                   # pragma: no cover
    cpu_lock['affinity_error'] = repr(exc)
    cpu_lock['affinity_enforced'] = False
cpu_lock['thread_env'] = {k: os.environ.get(k) for k in
                          ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                           'NUMEXPR_NUM_THREADS', 'JAX_PLATFORMS', 'CUDA_VISIBLE_DEVICES',
                           'XLA_PYTHON_CLIENT_PREALLOCATE')}
cpu_lock['enforced'] = bool(cpu_lock['affinity_enforced']
                            and cpu_lock['thread_env']['JAX_PLATFORMS'] == 'cpu'
                            and cpu_lock['thread_env']['CUDA_VISIBLE_DEVICES'] == '')

import psutil  # noqa: E402  (numpy-free)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(name, payload):
    (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
                            encoding='utf-8', newline='\n')


IDENTITY_FILES = [
    PACKAGE / 'server_diagnostic.py',
    PACKAGE / 'unified_closure_protocol.py',
    PACKAGE / 'supervise_diagnostic.py',
    ROOT / 'tools' / 'actual_loaded_closure_candidate_r20.py',
    ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem' / 'scripts' / 'models' / 'mpre.py',
    ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem' / 'scripts' / 'models' / 'cvrptw_utils.py',
    ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem' / 'scripts' / 'project_paths.py',
    ROOT / 'MASKCO_code' / 'ckpts' / 'cvrp100.ckpt',
]
REGISTRATION = PACKAGE / 'REGISTRATION.json'


def main():
    t_start = time.time()
    result = {'phase': 'start', 'pid': os.getpid(), 'python': sys.version.split()[0],
              'host': os.uname().nodename, 'ROOT': str(ROOT), 'cpu_lock': cpu_lock,
              'declared_budget_seconds': BUDGET_S, 'declared_reserve_seconds': RESERVE_S,
              'publication_performed': False}
    missing = [str(p) for p in IDENTITY_FILES if not p.is_file()]
    if missing:
        raise FileNotFoundError('identity files missing: %s' % missing)
    result['source_identity'] = {str(p): sha(p) for p in IDENTITY_FILES}
    result['registration_sha256'] = sha(REGISTRATION)
    result['source_identity_files'] = len(IDENTITY_FILES)
    write('DIAG_STATE.json', result)

    # ---- preconditions (all read-only, before any heavy import) ------------
    vm = psutil.virtual_memory()
    own = []
    for proc in psutil.process_iter(['pid', 'username', 'cmdline']):
        try:
            if proc.info['pid'] in (os.getpid(), os.getppid()):
                continue
            user = proc.info['username'] or ''
            cmd = ' '.join(proc.info['cmdline'] or [])
            if user.endswith('hzeng') and ('python' in cmd or 'MASKCO_env' in cmd):
                own.append(dict(pid=proc.info['pid'], cmd=cmd[:160]))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    per_core = psutil.cpu_percent(interval=1.0, percpu=True)
    core_usage = per_core[CORE] if CORE < len(per_core) else None
    load_avg = list(os.getloadavg()) if hasattr(os, 'getloadavg') else None
    result['preconditions'] = dict(
        measured_at_unix=time.time(),
        available_ram_gib=round(vm.available / 2 ** 30, 3), ram_gate_gib=RAM_GATE_GIB,
        ram_pass=bool(vm.available >= RAM_GATE_GIB * 2 ** 30),
        total_ram_gib=round(vm.total / 2 ** 30, 3),
        owned_leftover_processes=own, owned_leftovers_checked=True,
        owned_leftovers_pass=bool(len(own) == 0),
        core_index=CORE, core_usage_percent=core_usage,
        core_idle_gate_percent=CORE_IDLE_GATE,
        core_idle_pass=bool(core_usage is not None and core_usage < CORE_IDLE_GATE),
        cpu_count=psutil.cpu_count(),
        load_avg=[round(v, 2) for v in load_avg] if load_avg else None,
        gpu_stage_declared=False, gpu_devices_requested=0)
    result['phase'] = 'preconditions'
    write('DIAG_STATE.json', result)

    # ---- closure protocol: the SAME module verified by the local test ------
    sys.path.insert(0, str(PACKAGE))
    import unified_closure_protocol as U
    explicit = [str(p) for p in IDENTITY_FILES]
    picks = []
    t0 = time.perf_counter()
    p0 = U.snapshot(explicit)
    picks.append(dict(point='P0_before_declared_warmup', seconds=time.perf_counter() - t0,
                      files_hashed=len(p0['file_sha256'])))
    result['closure'] = dict(protocol_module=U.__file__, points=list(U.POINTS),
                             warmup=list(U.WARMUP),
                             gated_adjacencies=[list(a) for a in U.GATED_ADJACENCIES],
                             informational_adjacencies=[list(a) for a in U.INFORMATIONAL_ADJACENCIES],
                             captures=list(picks), explicit_files=len(explicit))
    result['phase'] = 'captured_P0'
    write('DIAG_STATE.json', result)

    # ---- declared warm-up: JAX import, model load, first JAX, first native --
    import numpy as np
    warm = {}
    t0 = time.perf_counter()
    import jax
    warm['jax_import_seconds'] = time.perf_counter() - t0
    warm['jax_version'] = jax.__version__
    warm['devices'] = [str(d) for d in jax.devices()]
    warm['backend'] = jax.default_backend()
    cvd = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem' / 'scripts'
    sys.path.insert(0, str(cvd))
    sys.path.insert(0, str(cvd / 'models'))
    from mpre import load_cvrp_model, encode_cvrp, decode_edge_logits
    t0 = time.perf_counter()
    model, cfg, step = load_cvrp_model(str(ROOT / 'MASKCO_code' / 'ckpts' / 'cvrp100.ckpt'))
    warm['model_load_seconds'] = time.perf_counter() - t0
    warm['ckpt_step'] = int(step) if step is not None else None
    warm['model_class'] = type(model).__name__

    N, B = 51, 1
    rng = np.random.default_rng(20261014)
    coords = rng.random((B, N, 2), dtype=np.float32)
    demands = rng.integers(1, 4, size=(B, N)).astype(np.float32)
    adj = np.ones((B, N, N), dtype=np.float32)
    ts = np.zeros((B,), dtype=np.float32)

    def unit():
        h = encode_cvrp(model, coords, demands, 50.0)
        return decode_edge_logits(model, h, ts, adj)

    t0 = time.perf_counter()
    logits = unit()
    warm['first_chain_seconds'] = time.perf_counter() - t0
    warm['logits_shape'] = list(np.shape(logits))
    warm['logits_finite'] = bool(np.isfinite(np.asarray(logits)).all())
    warm['rss_after_warmup_mib'] = round(psutil.Process().memory_info().rss / 2 ** 20, 2)

    t0 = time.perf_counter()
    p1 = U.snapshot(explicit)
    picks.append(dict(point='P1_after_declared_warmup', seconds=time.perf_counter() - t0,
                      files_hashed=len(p1['file_sha256'])))
    result['warmup'] = warm
    result['closure']['captures'] = picks
    result['phase'] = 'warmed_up'
    write('DIAG_STATE.json', result)
    del logits

    # ---- timed execution segment: BUDGET_S, unit repeated ------------------
    unit_times, exec_start = [], time.perf_counter()
    while True:
        u0 = time.perf_counter()
        out = unit()
        unit_times.append(time.perf_counter() - u0)
        if time.perf_counter() - exec_start >= BUDGET_S:
            break
    exec_seconds = time.perf_counter() - exec_start
    del out

    # ---- recovery reserve: REGISTERED as gc + release, measured ------------
    rss_peak = psutil.Process().memory_info().rss
    r0 = time.perf_counter()
    gc.collect()
    recovery_seconds = time.perf_counter() - r0
    rss_after = psutil.Process().memory_info().rss

    t0 = time.perf_counter()
    p2 = U.snapshot(explicit)
    picks.append(dict(point='P2_after_timed_execution', seconds=time.perf_counter() - t0,
                      files_hashed=len(p2['file_sha256'])))
    t0 = time.perf_counter()
    p3 = U.snapshot(explicit)
    picks.append(dict(point='P3_before_publication', seconds=time.perf_counter() - t0,
                      files_hashed=len(p3['file_sha256'])))
    verdict = U.evaluate(p0, p1, p2, p3)

    result['execution'] = dict(
        budget_seconds=BUDGET_S, execution_seconds=exec_seconds, units=len(unit_times),
        unit_seconds_first=unit_times[0],
        unit_seconds_median=float(np.median(unit_times)),
        unit_seconds_min=float(np.min(unit_times)), unit_seconds_max=float(np.max(unit_times)),
        first_unit_fits_budget=bool(unit_times[0] <= BUDGET_S),
        steady_unit_fits_budget=bool(float(np.median(unit_times)) <= BUDGET_S),
        reserve_seconds=RESERVE_S, recovery_seconds=recovery_seconds,
        recovery_within_reserve=bool(recovery_seconds <= RESERVE_S),
        rss_peak_mib=round(rss_peak / 2 ** 20, 2), rss_after_recovery_mib=round(rss_after / 2 ** 20, 2),
        rss_released_mib=round((rss_peak - rss_after) / 2 ** 20, 2),
        threads=psutil.Process().num_threads())
    result['closure']['captures'] = picks
    result['closure']['capture_seconds_total'] = round(sum(c['seconds'] for c in picks), 3)
    result['closure']['gated'] = {
        k: dict(qualified=v['qualified'], changed_files=len(v['changed_files']),
                new_files=len(v['new_files']), missing_files=len(v['missing_files']),
                module_map_changed=v['module_map_changed'],
                native_map_changed=v['native_map_changed'],
                explicit_map_changed=v['explicit_map_changed'],
                detail_new=[Path(f).name for f in v['new_files']][:8])
        for k, v in verdict['gated'].items()}
    result['closure']['informational'] = {
        k: dict(qualified=v['qualified'], new_files=len(v['new_files']),
                changed_files=len(v['changed_files']))
        for k, v in verdict['informational'].items()}
    result['closure']['publishable'] = verdict['publishable']
    result['closure']['refusal_reasons'] = verdict['refusal_reasons']

    pre = result['preconditions']
    result['gates'] = {
        'cpu_environment_locked': cpu_lock['enforced'],
        'available_ram_gate': pre['ram_pass'],
        'no_owned_leftover_processes': pre['owned_leftovers_pass'],
        'target_core_idle': pre['core_idle_pass'],
        'cpu_only_no_gpu_device': bool(len(warm['devices']) >= 1 and
                                       all('cpu' in d.lower() for d in warm['devices']))
                                  and warm['backend'] == 'cpu',
        'registration_hash_recorded': bool(result['registration_sha256']),
        'model_loaded': bool(warm.get('model_load_seconds') is not None),
        'unit_fits_budget': result['execution']['first_unit_fits_budget'],
        'recovery_within_reserve': result['execution']['recovery_within_reserve'],
        'closure_gated_adjacencies_qualified': bool(verdict['publishable']),
    }
    result['gate_pass'] = all(result['gates'].values())
    result['failed_gates'] = sorted(k for k, v in result['gates'].items() if not v)
    result['phase'] = 'complete'
    result['wall_seconds'] = round(time.time() - t_start, 3)
    write('DIAG_STATE.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2)[:6000])
    return 0 if result['gate_pass'] else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        tb = traceback.format_exc()
        try:
            write('DIAG_FAILURE.json', {'phase': 'failure', 'traceback': tb,
                                        'gate_pass': False, 'publication_performed': False,
                                        'pid': os.getpid(),
                                        'registration_sha256': sha(REGISTRATION)
                                        if REGISTRATION.is_file() else None,
                                        'source_identity': {str(p): sha(p)
                                                            for p in IDENTITY_FILES
                                                            if p.is_file()}})
            (OUT / 'DIAG_FAILURE.traceback.txt').write_text(tb, encoding='utf-8', newline='\n')
        finally:
            traceback.print_exc()
        raise SystemExit(2)
