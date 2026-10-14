"""Offline archived-clock and AST diagnosis only; never import experiment code."""
import ast
import hashlib
import json
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'research/review_3080_20261010'
OLD = R / 'delivery/timed_actual_request_execution_root_v1'
A = OLD / 'downloaded'
O = R / 'delivery/timed_request_timing_boundary_diagnosis_root_v1'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def create(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def inputs():
    paths = [OLD/'DOWNLOAD_MANIFEST.json', A/'RESULT.json']
    for name in ['delivery_timed_actual_engine_v3.py', 'delivery_timed_shared_worker_v3.py',
                 'action_head_runtime_v1.py', 'timed_evidence_io_v1.py']:
        paths.append(A/name)
    for receipt in read(A/'RESULT.json')['receipts']:
        q = A/(receipt['arm']+'_fault'+str(receipt['fault_stage'])+'_worker')
        paths.extend(sorted(q.glob('STAGE_*.json')))
        paths.extend(sorted(q.glob('FAULT_DELAY_ENTERED.json')))
        paths.extend(sorted(q.glob('FAULT_SOURCE_AT_ENTRY.json')))
    return paths

def exact_difference(a, b):
    value = Fraction.from_float(a)-Fraction.from_float(b)
    return {'seconds': float(value), 'rational': str(value), 'later_hex': a.hex(), 'earlier_hex': b.hex()}

def register():
    O.mkdir(exist_ok=False)
    create(O/'PREREGISTRATION.json', {
        'registered_utc': datetime.now(timezone.utc).isoformat(),
        'kind': 'new_offline_engineering_diagnosis_of_existing_evidence',
        'script_sha256': sha(__file__),
        'inputs': {str(p.relative_to(ROOT)): sha(p) for p in inputs()},
        'method': 'All ten archived receipts, all available stage clocks; exact binary64 rational subtraction, manifest byte accounting, AST statement order. No timing inferred from download timestamps.',
        'forbidden': ['model import/forward', 'solver/C0 replay', 'production/audit/recovery rerun', 'new sample/teacher/training', 'budget relaxation/sample warmup', 'counterfactual speedup claim'],
        'numeric_tolerance': None,
        'execution_authorized': False,
        'family3_started': False,
        'process_observation_limit': 'Earlier local Win32_Process read returned Access Denied; no current local or remote process-absence assertion is made. This program only reads archived files.',
        'read_probe_failure': 'A preliminary read used pretrain_initial instead of actual pretrained_initial directory and raised FileNotFoundError; corrected read only, original artifacts unchanged.'})
    print('Registered offline diagnosis; no experiment execution.')

def diagnose():
    reg = read(O/'PREREGISTRATION.json')
    assert sha(__file__) == reg['script_sha256']
    assert all(sha(ROOT/p) == h for p, h in reg['inputs'].items())
    manifest = read(OLD/'DOWNLOAD_MANIFEST.json')
    rows = []
    for r in read(A/'RESULT.json')['receipts']:
        tag = r['arm']+'_fault'+str(r['fault_stage'])
        q = A/(tag+'_worker')
        stages = sorted((read(p) for p in q.glob('STAGE_*.json')), key=lambda x: x['clock'])
        clocks = [('caller_start', r['start_clock'])]+[(x['name'], x['clock']) for x in stages]+[('worker_deadline', r['child_deadline']), ('kill', r['kill_clock']), ('reap', r['reap_clock']), ('caller_end', r['end_clock'])]
        assert all(clocks[i][1] <= clocks[i+1][1] for i in range(len(clocks)-1))
        parts = [dict(from_name=b[0], to_name=e[0], **exact_difference(e[1], b[1])) for b, e in zip(clocks, clocks[1:])]
        assert sum((Fraction(x['rational']) for x in parts), Fraction()) == Fraction.from_float(r['end_clock'])-Fraction.from_float(r['start_clock'])
        delay_path = q/'FAULT_DELAY_ENTERED.json'
        delay = read(delay_path) if delay_path.exists() else None
        matched = next((s for s in stages if s['stage']==r['fault_stage']), None)
        entries = [x for x in manifest if x['name'].replace('\\','/').startswith(tag+'_worker/request/')]
        scores = [x for x in entries if Path(x['name']).suffix=='.npz' and Path(x['name']).stem.isdigit()]
        transitions = [x for x in entries if Path(x['name']).suffix=='.json' and Path(x['name']).stem.isdigit()]
        rows.append(dict(tag=tag, clock_partition=parts,
            marker_to_delay=exact_difference(delay['clock'], matched['clock']) if delay else None,
            delay_to_deadline=exact_difference(r['child_deadline'], delay['clock']) if delay else None,
            archived_request_bytes=sum(x['size'] for x in entries),
            completed_score_bytes=sum(x['size'] for x in scores), completed_score_files=len(scores),
            completed_transition_bytes=sum(x['size'] for x in transitions), completed_transition_files=len(transitions),
            timing_identifies_model_vs_IO=False,
            manifest_has_original_event_timestamps=any('mtime' in x or 'clock' in x for x in entries)))
    witnesses = {}
    for name, function in [('action_head_runtime_v1.py','closure'), ('delivery_timed_shared_worker_v3.py','stage'), ('delivery_timed_actual_engine_v3.py','request'), ('timed_evidence_io_v1.py','save_npz')]:
        src = (A/name).read_text(encoding='utf-8')
        tree = ast.parse(src)
        node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name==function)
        calls = [{'line': n.lineno, 'call': ast.unparse(n.func)} for n in ast.walk(node) if isinstance(n, ast.Call)]
        witnesses[name] = dict(sha256=sha(A/name), function=function, start_line=node.lineno, end_line=node.end_lineno, calls=sorted(calls,key=lambda x:x['line']))
    engine = (A/'delivery_timed_actual_engine_v3.py').read_text(encoding='utf-8')
    worker = (A/'delivery_timed_shared_worker_v3.py').read_text(encoding='utf-8')
    runtime = (A/'action_head_runtime_v1.py').read_text(encoding='utf-8')
    assert engine.index("stage(6,") < engine.index('self.closure(self.extras,self.trees)')
    assert worker.index("'STAGE_%d.json'") < worker.index('snapshot=engine.closure') < worker.index("'FAULT_DELAY_ENTERED.json'")
    assert 'rglob' in runtime and 'memory_maps' in runtime
    create(O/'DIAGNOSIS.json', dict(
        preregistration_sha256=sha(O/'PREREGISTRATION.json'), input_identity_unchanged=True,
        rows=rows, source_AST_witnesses=witnesses,
        findings=[
            'Marker-to-delay is an aggregate of marker publication, registered-tree/package/stdlib/native enumeration and hashing, delta formation and JSON evidence writing; not closure hashing alone.',
            'Closure seals loaded-module/native identities plus additional registered package/source/stdlib trees. File count is not a count of actually imported modules.',
            'Normal stage2-to3 includes original incumbent C0, encoder and evidence persistence; stage3-to4 includes options/context, decoder/head, synchronization and NPZ persistence.',
            'After first score marker, archived evidence has no per-step compute/IO clocks. Manifest bytes measure stored evidence volume, not IO duration or kernel/JIT timing.',
            'Complete-chain source would additionally enumerate/hash closure and parameter inventories before publishing. No current request reached that segment; its actual duration is unobserved.',
            'No numerical decomposition identifies all-timeout cause or predicts completion after IO changes. Neither subtracting observed fault instrumentation nor extrapolating 16 insertions qualifies the request.'],
        next_design={'scope':'Prospectively add actual phase start/end clocks and bounded evidence records, separating observed computation, serialization/write/hash and source checks while retaining every timed operation and original fixed deadline.',
                     'gates':'Independent source/parameter closure remains required before eligibility; do not simply move its cost outside the request and relabel qualification. Separate engineering preparation and new declared workload from science.',
                     'execution_authorized':False},
        model_forward_calls=0, solver_C0_calls=0, old_QA_repeated=False,
        formal_controls_qualified=False, full_request_budget_qualified=False,
        full_scientific_loaded_closure_qualified=False, family3_started=False, stable_online_gain=False))
    assert all(sha(ROOT/p)==h for p,h in reg['inputs'].items())
    create(O/'RUN_EXIT.json', {'exit_code':0, 'kind':'offline_diagnosis_only','result_sha256':sha(O/'DIAGNOSIS.json')})
    print(json.dumps({'requests_diagnosed':len(rows),'model_execution':False,'budget_qualified':False}))

if __name__ == '__main__':
    {'register':register,'diagnose':diagnose}[sys.argv[1]]()
