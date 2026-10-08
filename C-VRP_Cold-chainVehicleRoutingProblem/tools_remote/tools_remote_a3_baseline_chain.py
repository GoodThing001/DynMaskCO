"""One finite A3 repair chain: smoke validation -> 3 baseline batches -> rejudge.

Runs in its own tmux session, never edits original outputs, never chooses a
baseline from observed utility, never touches final-test seeds or other jobs.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem')
SEEDS = ('20260926', '20260930', '20261001')
BASE = Path('results/a3_baseline_src_frozen_20261007')
COND = Path('results/a3_baseline_condonly_v2_src_frozen_20261007')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, d):
    Path(p).write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding='utf-8')


def completed(d):
    p = Path(d) / 'run.exit'
    if not p.exists():
        return False
    lines = set(p.read_text().splitlines())
    if not {'RUN_EXIT=0', 'SEAL_EXIT=0', 'DONE=1'}.issubset(lines):
        raise RuntimeError('Run or end seal failed: ' + str(d))
    if not (Path(d) / 'gate.json').exists():
        raise RuntimeError('Done without gate: ' + str(d))
    return True


def driver_proof():
    """The task payload and all worker/scoring functions must be identical."""
    rel = 'scripts/evaluation/run_a1_step2_gate.py'
    trees = [ast.parse((p / rel).read_text(encoding='utf-8')) for p in (BASE, COND)]
    fn = [{n.name: n for n in t.body if isinstance(n, ast.FunctionDef)} for t in trees]
    for name in ('run_arm', '_worker', 'summarize', 'paired_all_days'):
        if ast.dump(fn[0][name]) != ast.dump(fn[1][name]):
            raise RuntimeError('Worker or statistic changed: ' + name)
    task_loops = [[n for n in ast.walk(f['main']) if isinstance(n, ast.For)
                   and isinstance(n.iter, ast.Call)
                   and isinstance(n.iter.func, ast.Attribute)
                   and ast.unparse(n.iter.func) == 'samplers.items'] for f in fn]
    if any(len(x) != 1 for x in task_loops) or \
            ast.dump(task_loops[0][0]) != ast.dump(task_loops[1][0]):
        raise RuntimeError('Task payload/random seed changed')
    # Exercise the baseline-only report path without spending online compute.
    source = (COND / rel).read_text(encoding='utf-8')
    lo, hi = source.index('    per_day = {}\n'), source.index('    seal_end(src_seal)')
    block = '\n'.join(line[4:] for line in source[lo:hi].splitlines())
    from types import SimpleNamespace
    rows = [{'i': i, 'utility': 10.0, 'hard_feasible': True} for i in range(40)]
    env = {'task_results': [('p_c=(5,10,15)', 'cond_hist', rows)],
           'args': SimpleNamespace(only_cond=True, gate_seed=20260926, gate_instances=40),
           'summarize': lambda r, seed, n: {'n': len(r)},
           'samplers': {'cond_hist': object()}}
    exec(compile(block, '<cond-only-report-proof>', 'exec'), env)
    report = env['results']['p_c=(5,10,15)']
    if report['per_day']['cond_hist'] != rows or 'main_comparison' in report:
        raise RuntimeError('Baseline-only report not honest or complete')
    return {'worker_and_task_ast_equal': True, 'baseline_only_report_probe': True}


def smoke_check(path, expected_shared, cond_only):
    doc = read(Path(path) / 'gate.json')
    ident = doc['identity']
    if ident.get('source_stable') is not True or \
            ident.get('source_sha256') != ident.get('source_end_sha256'):
        raise RuntimeError('Smoke source changed: ' + path)
    if ident.get('shared_source_sha256') != expected_shared:
        raise RuntimeError('Smoke shared source mismatches A3: ' + path)
    cfg = doc['config']
    expected_cfg = {'train_instances': 200, 'gate_instances': 1,
                    'gate_seed': 20260926, 'K': 10, 'time_limit': 10.0,
                    'future_policy': 'density', 'energy_pricing': 'marginal',
                    'shadow_mode': 'greedy', 'penalty': 'p_c=0'}
    if any(cfg.get(k) != v for k, v in expected_cfg.items()):
        raise RuntimeError('Smoke configuration mismatch: ' + path)
    arms = doc['gate']['p_c=0']['per_day']
    names = {'cond_hist'} if cond_only else {'cond_hist', 'uncond_hist', 'explicit_feat'}
    if set(arms) != names:
        raise RuntimeError('Smoke arm set mismatch: ' + path)
    for name, rows in arms.items():
        if len(rows) != 1 or rows[0]['i'] != 0 or \
                rows[0].get('hard_feasible') is not True or \
                rows[0].get('utility') is None or not math.isfinite(rows[0]['utility']):
            raise RuntimeError('Smoke day/hard/utility invalid: ' + name)
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full-smoke', required=True)
    ap.add_argument('--cond-smoke', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    import os
    os.chdir(ROOT)
    out = Path(args.out)
    out.mkdir(exist_ok=False)
    validator_src = Path('scripts/evaluation/diag_a3_adjudicate.py')
    validator = out / validator_src.name
    shutil.copy2(validator_src, validator)
    validator_sha = hashlib.sha256(validator.read_bytes()).hexdigest()
    originals = read('results/a1_a3/A3_ADJUDICATION_FINAL.json')['per_batch']
    dirs = []
    for name in originals:
        matches = list(Path('results').glob('**/' + name + '/gate.json'))
        if len(matches) != 1:
            raise RuntimeError('Ambiguous original A3 batch: ' + name)
        dirs.append(str(matches[0].parent))
    if len(dirs) != 9:
        raise RuntimeError('Original A3 matrix is not nine batches')
    reference = next(read(Path(d) / 'gate.json') for d in dirs if '_B_20260926_' in d)
    expected_shared = reference['results']['20260926']['_identity']['shared_source_sha256']
    write(out / 'chain_state.json', {'stage': 'await_smokes',
          'full_smoke': args.full_smoke, 'cond_smoke': args.cond_smoke,
          'validator_sha256': validator_sha, 'original_batches': dirs,
          'driver_proof': driver_proof()})
    for frozen in (BASE, COND):
        subprocess.run(['sha256sum', '-c', 'SOURCE_MANIFEST.sha256'],
                       cwd=frozen, check=True, stdout=subprocess.DEVNULL)
    limit = time.monotonic() + 3 * 3600
    while not all(completed(d) for d in (args.full_smoke, args.cond_smoke)):
        if time.monotonic() > limit:
            raise RuntimeError('Smoke completion wait exceeded 3 hours')
        time.sleep(30)
    a = smoke_check(args.full_smoke, expected_shared, False)
    b = smoke_check(args.cond_smoke, expected_shared, True)
    for key in ('contract_sha256', 'data_sha256', 'data_meta_sha256',
                'budget_B', 'cooling_share', 'shared_config', 'seeds'):
        if a['identity'].get(key) != b['identity'].get(key):
            raise RuntimeError('Smoke pair identity mismatch: ' + key)
    ra = a['gate']['p_c=0']['per_day']['cond_hist'][0]
    rb = b['gate']['p_c=0']['per_day']['cond_hist'][0]
    compare = {k: {'all_arms': ra.get(k), 'cond_only': rb.get(k)} for k in
               ('utility', 'served_n', 'reject_n', 'energy_total', 'timeouts')}
    # Wall-clock cutoffs can vary across concurrent schedules. They are reported,
    # never used to select among baseline outputs; worker/task equivalence above
    # establishes the only-cond semantics before any full result is observed.
    write(out / 'smoke_validation.json', {'qualified': True,
          'driver_proof': driver_proof(), 'cond_day_comparison': compare,
          'comparison_note': 'Wall-clock-sensitive diagnostic, no utility selection',
          'shared_source_sha256': expected_shared})
    baseline_dirs = {}
    for seed in SEEDS:
        run = subprocess.run(['bash', 'tools_remote/tools_remote_run_a3_baseline_frozen.sh',
                              'full-cond', seed], check=True, capture_output=True, text=True)
        print(run.stdout.strip(), flush=True)
        starts = [l for l in run.stdout.splitlines() if l.startswith('STARTED: ')]
        if len(starts) != 1:
            raise RuntimeError('Unexpected launch result: ' + seed)
        baseline_dirs[seed] = starts[0].split()[-1]
        write(out / 'baseline_map.json', {'gate_json_by_seed': {
            s: str(Path(d) / 'gate.json') for s, d in baseline_dirs.items()}})
    write(out / 'chain_state.json', {'stage': 'baselines_running',
          'baseline_dirs': baseline_dirs, 'validator_sha256': validator_sha})
    limit = time.monotonic() + 72 * 3600
    while not all(completed(d) for d in baseline_dirs.values()):
        if time.monotonic() > limit:
            raise RuntimeError('Baseline completion wait exceeded 72 hours')
        time.sleep(60)
    spec = importlib.util.spec_from_file_location('a3_sealed_validator', validator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.main(dirs + ['--lock', 'results/A2_LOCK.json',
        '--baseline-map', str(out / 'baseline_map.json'),
        '--out', str(out / 'A3_SAME_SOURCE_ADJUDICATION.json')])
    write(out / 'chain_state.json', {'stage': 'finished',
          'formal_adjudicable': result['formal_adjudicable'],
          'baseline_dirs': baseline_dirs, 'validator_sha256': validator_sha})
    if not result['formal_adjudicable']:
        raise RuntimeError('Independent A3 qualification failed; inspect new report')
    print('A3_REPAIR_CHAIN_FINISHED: qualified; numeric outcomes preserved as observed', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('A3_REPAIR_CHAIN_FAILED:', type(exc).__name__, str(exc), flush=True)
        raise
