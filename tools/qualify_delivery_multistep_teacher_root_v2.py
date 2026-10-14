"""Predeclared engineering on two observed fixtures; no new scientific sample."""
import ctypes, hashlib, importlib.metadata, json, os, platform, queue
import subprocess, sys, threading, time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
D = ROOT / 'research/review_3080_20261010/delivery'
OUT = D / 'multistep_insertable_teacher_engineering_root_v2'
FIXTURE = D / 'finite_training_preparation_v1/ENGINEERING_TEACHER_QA_v1/TEACHER_STATES.jsonl'
for sub in ('delivery', 'coldchain', 'research_20261010'):
    sys.path.insert(0, str(E / 'scripts' / sub))
sys.path.insert(0, str(OUT))
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def write(p, x):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(x, f, indent=2, allow_nan=False)

def affinity():
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    k.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    h = k.GetCurrentProcess()
    if not k.SetProcessAffinityMask(h, 1 << 7): raise ctypes.WinError(ctypes.get_last_error())
    a, b = ctypes.c_size_t(), ctypes.c_size_t()
    if not k.GetProcessAffinityMask(h, ctypes.byref(a), ctypes.byref(b)): raise ctypes.WinError(ctypes.get_last_error())
    if a.value != 1 << 7: raise RuntimeError('Actual CPU7 required')
    return [7]

def closure():
    import numpy, ortools, google.protobuf
    versions = {n: importlib.metadata.version(n) for n in ('numpy', 'ortools', 'protobuf')}
    machine = platform.platform()
    files = {Path(__file__).resolve(), FIXTURE, OUT / 'retained_gls.py', Path(sys.executable).resolve()}
    # Pin actual loaded sources, cached bytecode and extension modules as well.
    for m in list(sys.modules.values()):
        for attr in ('__file__', '__cached__'):
            p = getattr(m, attr, None)
            if p and Path(p).is_file(): files.add(Path(p).resolve())
    for m in (numpy, ortools, google.protobuf):
        for p in Path(m.__file__).parent.rglob('*'):
            if p.is_file(): files.add(p.resolve())
    for name in ('numpy.libs', 'ortools.libs'):
        folder = Path(numpy.__file__).parent.parent / name
        if folder.exists(): files.update(p.resolve() for p in folder.rglob('*') if p.is_file())
    for folder in (Path(sys.base_prefix) / 'DLLs',):
        if folder.exists(): files.update(p.resolve() for p in folder.rglob('*') if p.is_file())
    files.update(p.resolve() for p in Path(sys.base_prefix).glob('python*.dll'))
    return {'files': {str(p): sha(p) for p in sorted(files)},
            'python': sys.version, 'executable': sys.executable, 'platform': machine,
            'packages': versions,
            'scope': 'actual imported sources/bytecode, NumPy/ORTools/protobuf trees, interpreter and bundled DLLs; not OS or future JAX training closure'}

def worker():
    cpu = affinity()
    import numpy as np
    from retained_gls import solve
    from ortools.constraint_solver import pywrapcp
    from delivery_equal_schedule import reconstruct
    from delivery_maskco import partial_adjacency
    from delivery_evaluator import evaluate
    from delivery_contract import DeliveryContract
    pywrapcp.DefaultRoutingSearchParameters()
    contract = DeliveryContract()
    start = closure()
    write(OUT / 'WORKER_SOURCE_START.json', start)
    print(json.dumps({'kind': 'ready', 'cpu': cpu, 'seal_sha256': sha(OUT / 'WORKER_SOURCE_START.json')}), flush=True)
    for line in sys.stdin:
        q = json.loads(line)
        if q.get('kind') == 'seal':
            changed = [p for p, h in start['files'].items() if not Path(p).is_file() or sha(p) != h]
            end = closure(); extra = sorted(set(end['files']) - set(start['files']))
            write(OUT / 'WORKER_SOURCE_END.json', {'closure': end, 'changed': changed, 'new_loaded_files': extra})
            print(json.dumps({'kind': 'sealed', 'changed': changed, 'new_loaded_files': extra}), flush=True)
            break
        try:
            if q.get('delay'): time.sleep(1.)
            a = partial_adjacency(q['initial'], set(q['mask']), len(q['data']['demands']))
            teacher = solve(q['data'], q['initial'], a.tolist(), contract, q['deadline'], q['mask'])
            # Deployment state transitions use H, never a teacher-driven rollout.
            routes, remaining, insertions, status = reconstruct(q['data'], q['initial'], contract,
                set(q['mask']), a, None, 0., q['deadline'], time.perf_counter)
            report = evaluate(q['data'], routes, contract, deadline=q['deadline'])
            accepted = not remaining and report['hard'] and time.perf_counter() < q['deadline'] and report['cost'] < teacher['initial_report']['cost']
            print(json.dumps({'kind': 'response', 'teacher': teacher, 'partial': a.tolist(),
                'deployment_candidate': routes, 'deployment_report': report, 'remaining': remaining,
                'insertions': insertions, 'status': status, 'deployment_accepted': bool(accepted),
                'incumbent_after': routes if accepted else q['initial']}), flush=True)
        except Exception as exc:
            print(json.dumps({'kind': 'error', 'error': type(exc).__name__ + ':' + str(exc)}), flush=True)

def main():
    cpu = affinity()
    from delivery_equal_schedule import make_schedule
    OUT.mkdir(exist_ok=False)
    # Exclusive copy: preserve v3 and repair empty vehicle-slot compression here.
    old = D / 'insertable_teacher_engineering_root_v3/retained_gls.py'
    source = old.read_text()
    assert source.count('if route:routes.append(route)') == 1
    (OUT / 'retained_gls.py').write_text(source.replace('if route:routes.append(route)', 'routes.append(route)'), encoding='utf-8')
    reg = {'created_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'engineering only on two old observed fixtures, four evolving deployment states each',
        'ids': [900000, 900001], 'steps': 4, 'mask_seed': 2026101141, 'mask_fraction': .5,
        'state_transition': 'actual nonlearning distance-regret2 reconstruct with original masks; accept complete fullC0-hard improvement only; teacher never selects rollout state',
        'request_cap_s': 10., 'teacher_search_cap_s': 3., 'parent_worker_cpu': cpu,
        'fixture_sha256': sha(FIXTURE), 'previous_teacher_sha256': sha(old), 'teacher_sha256': sha(OUT / 'retained_gls.py'),
        'launcher_sha256': sha(__file__), 'source_change': 'preserve empty original vehicle slots in extract; no old source altered',
        'retention': 'all eight states and every solver candidate; no resampling, label or improvement selection',
        'new_science_family': False, 'stable_online_gain': False,
        'references': ['https://developers.google.com/optimization/routing/routing_tasks', 'https://proceedings.neurips.cc/paper/2019/file/131f383b434fdf48079bff1e44e2d9a5-Paper.pdf']}
    write(OUT / 'ENGINEERING_REGISTRATION.json', reg)
    p = subprocess.Popen([sys.executable, '-B', str(Path(__file__)), '--worker'], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    inbox, errors = queue.Queue(), []
    def reader():
        for line in p.stdout:
            try: inbox.put(json.loads(line))
            except Exception: errors.append(line)
        inbox.put({'kind': 'closed'})
    def errreader():
        for line in p.stderr: errors.append(line)
    threading.Thread(target=reader, daemon=True).start(); threading.Thread(target=errreader, daemon=True).start()
    records, checks, sealed, ready = [], [], None, None
    def ck(k, v): checks.append({'name': k, 'pass': bool(v)})
    try:
        ready = inbox.get(timeout=60); ck('actual_worker_cpu7', ready.get('kind') == 'ready' and ready.get('cpu') == [7])
        if ready.get('kind') != 'ready': raise RuntimeError('No qualified worker')
        rows = [json.loads(x) for x in FIXTURE.read_text().splitlines()]
        with (OUT / 'STATES.jsonl').open('x', encoding='utf-8') as f:
            for i in reg['ids']:
                old = next(x for x in rows if x['instance_id'] == i and x['mask_ordinal'] == 0)
                incumbent = deepcopy(old['incumbent_routes'])
                schedule = make_schedule(len(old['data']['demands']), reg['steps'], reg['mask_seed'], i, reg['mask_fraction'])
                for step, mask in enumerate(schedule):
                    t = time.perf_counter(); q = {'data': old['data'], 'initial': incumbent, 'mask': mask, 'deadline': t + 10.}
                    p.stdin.write(json.dumps(q) + '\n'); p.stdin.flush()
                    try: answer = inbox.get(timeout=max(0., q['deadline'] - time.perf_counter()))
                    except queue.Empty: answer = {'kind': 'timeout'}
                    got = time.perf_counter()
                    row = {'instance_id': i, 'step': step, 'request': q, 'started': t, 'received': got, 'wall_s': got - t, 'answer': answer}
                    f.write(json.dumps(row, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno()); records.append(row)
                    timely = answer.get('kind') == 'response' and got < q['deadline']
                    ck(str((i, step)) + ':complete_response_in_cap', timely)
                    if not timely: raise RuntimeError('Retained failed state; do not resample')
                    incumbent = deepcopy(answer['incumbent_after'])
        p.stdin.write(json.dumps({'kind': 'seal'}) + '\n'); p.stdin.flush()
        sealed = inbox.get(timeout=60)
        ck('actual_start_end_byte_identity', sealed.get('kind') == 'sealed' and not sealed.get('changed'))
        ck('no_unsealed_lazy_dependency', not sealed.get('new_loaded_files'))
        p.wait(timeout=10); ck('worker_clean_exit', p.returncode == 0)
    except Exception as exc:
        errors.append(type(exc).__name__ + ':' + str(exc)); ck('execution_completed', False)
    finally:
        if p.poll() is None: p.kill()
        p.wait(timeout=10)
    ck('eight_states_all_retained', len(records) == 8)
    ck('fixture_launcher_teacher_unchanged', sha(FIXTURE) == reg['fixture_sha256'] and sha(__file__) == reg['launcher_sha256'] and sha(OUT / 'retained_gls.py') == reg['teacher_sha256'])
    result = {'checks': checks, 'failed': sum(not x['pass'] for x in checks), 'state_count': len(records),
        'registration_sha256': sha(OUT / 'ENGINEERING_REGISTRATION.json'), 'states_sha256': sha(OUT / 'STATES.jsonl') if (OUT / 'STATES.jsonl').exists() else None,
        'ready': ready, 'sealed': sealed, 'errors': errors, 'worker_reaped': p.poll() is not None,
        'scientific_training_closure_qualified': False, 'engineering_only': True, 'stable_online_gain': False}
    write(OUT / 'RESULT.json', result)
    print(json.dumps({'states': len(records), 'failed': result['failed'], 'errors': errors, 'sealed': sealed})); return int(result['failed'] > 0)

if __name__ == '__main__':
    if '--worker' in sys.argv: worker()
    else: raise SystemExit(main())
