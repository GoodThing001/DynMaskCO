"""New fixed-schedule attribution panel; never reads or resumes original D2 rows.

P and R use actual CVRP encoder+decoder. H skips inference and removes the
logit term in the identical insertion decoder. This controls opportunities,
not consumed compute. Every planned attempt, including failed/late, is logged.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS / 'delivery'))
sys.path.insert(0, str(SCRIPTS / 'coldchain'))
from delivery_baselines import matrices, route_ok
from delivery_contract import DeliveryContract
from delivery_evaluator import evaluate
from delivery_maskco import MaskCORouteProvider, partial_adjacency


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def dump_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def make_schedule(nodes, steps, seed, instance_id, fraction=.3):
    if nodes < 2 or steps < 1 or not 0 < fraction <= 1:
        raise ValueError('Invalid fixed schedule')
    rng = np.random.default_rng(np.random.SeedSequence([seed, instance_id]))
    return [sorted(map(int, rng.choice(np.arange(1, nodes),
                max(1, int(math.ceil(fraction * (nodes - 1)))), replace=False)))
            for _ in range(steps)]


def reconstruct(instance, incumbent, contract, removed, adjacency, logits,
                guide_weight, deadline, clock):
    """Same regret-2 customer insertion and immutable original edges for P/R/H."""
    dist = matrices(instance)
    off = ~np.eye(len(dist), dtype=bool)
    dscale = max(1e-9, float(dist[off].mean()))
    lscale = 1.0 if logits is None else max(1e-8, float(np.std(logits[off])))
    candidate = [[i for i in route if i not in removed] for route in incumbent]
    candidate += [[] for _ in range(contract.vehicles - len(candidate))]
    remaining = sorted(removed)
    insertions = []
    while remaining:
        if clock() >= deadline:
            return candidate, remaining, insertions, 'deadline'
        choices = []
        for i in remaining:
            options = []
            for k, route in enumerate(candidate):
                for at in range(len(route) + 1):
                    if clock() >= deadline:
                        return candidate, remaining, insertions, 'deadline'
                    a, b = (route[at-1] if at else 0), (route[at] if at < len(route) else 0)
                    if adjacency[a, b] > 0:
                        continue
                    if not route_ok(route[:at] + [i] + route[at:], instance, contract, dist, 0.):
                        continue
                    delta = dist[a, i] + dist[i, b] - dist[a, b]
                    gain = 0. if logits is None else (
                        logits[a, i] + logits[i, b] - (0. if a == b else logits[a, b])) / lscale
                    options.append((float(delta / dscale - guide_weight * gain), k, at))
            options.sort()
            if options:
                # Preserve the original regret preference, deterministic customer tie break.
                regret = options[1][0] - options[0][0] if len(options) > 1 else float('inf')
                choices.append((regret, i, options[0]))
        if not choices:
            return candidate, remaining, insertions, 'no_feasible_insertion'
        _, i, (score, k, at) = max(choices, key=lambda row: (row[0], -row[1]))
        candidate[k].insert(at, i)
        remaining.remove(i)
        insertions.append({'customer': i, 'vehicle': k, 'position': at, 'score': score})
    return candidate, [], insertions, 'complete'


def run_arm(instance, initial_routes, contract, schedule, provider=None,
            time_limit_s=10., guide_weight=1., clock=time.perf_counter):
    if not math.isfinite(time_limit_s) or time_limit_s <= 0:
        raise ValueError('Invalid wall budget')
    nodes = len(instance['demands'])
    if len(schedule) != 8 or any(len(row) != len(set(row)) or not row or
                any(type(i) is not int or not 1 <= i < nodes for i in row) for row in schedule):
        raise ValueError('Exactly eight valid customer masks required')
    started = clock()
    deadline = started + time_limit_s
    initial = evaluate(instance, initial_routes, contract, deadline=deadline, clock=clock)
    if not initial['hard']:
        raise ValueError('Unqualified common initial incumbent; do not screen/resample')
    feature_start = clock()
    features = provider.features(instance, contract.capacity) if provider is not None else None
    # Force encoder completion before measuring its duration.
    if features is not None and hasattr(features, 'block_until_ready'):
        features.block_until_ready()
    feature_s = clock() - feature_start
    incumbent, report = deepcopy(initial_routes), initial
    attempts = []
    first_cost = initial['cost']
    for step, mask in enumerate(schedule, 1):
        step_start = clock()
        removed = set(mask)
        adjacency = partial_adjacency(incumbent, removed, nodes)
        row = {'step': step, 'removed': mask, 'mask_sha256': digest(mask),
               'incumbent_before': deepcopy(incumbent), 'cost_before': report['cost'],
               'adjacency_sha256': hashlib.sha256(adjacency.tobytes()).hexdigest(),
               'retained_edges': np.argwhere(np.triu(adjacency, 1) > 0).tolist(),
               'decode_calls': 0, 'candidate_complete': False, 'candidate_hard': False,
               'accepted': False, 'decode_s': 0., 'repair_s': 0., 'certification_s': 0.}
        if clock() >= deadline:
            row['status'] = 'not_started_deadline'
            row['candidate_routes'] = None
            row['remaining'] = mask
            row['insertions'] = []
        else:
            decode_start = clock()
            logits = provider.logits(features, adjacency) if provider is not None else None
            row['decode_s'] = clock() - decode_start
            row['decode_calls'] = int(provider is not None)
            if logits is not None:
                off = ~np.eye(nodes, dtype=bool)
                if logits.shape != (nodes, nodes) or not np.all(np.isfinite(logits[off])):
                    raise ValueError('Actual model logits invalid')
                row['logits_sha256'] = hashlib.sha256(np.ascontiguousarray(logits).tobytes()).hexdigest()
            repair_start = clock()
            candidate, remaining, insertions, status = reconstruct(
                instance, incumbent, contract, removed, adjacency, logits,
                guide_weight if provider is not None else 0., deadline, clock)
            row.update(candidate_routes=candidate, remaining=remaining,
                       insertions=insertions, status=status, repair_s=clock()-repair_start)
            if not remaining:
                row['candidate_complete'] = True
                retained = partial_adjacency(candidate, set(), nodes)
                if not np.all(retained[adjacency > 0] > 0):
                    raise RuntimeError('Unmasked original edge changed')
                row['retained_edges_preserved'] = True
                cert_start = clock()
                checked = evaluate(instance, candidate, contract, deadline=deadline, clock=clock)
                row.update(certification_s=clock()-cert_start, candidate_report=checked,
                           candidate_hard=bool(checked['hard']))
                if checked['hard'] and clock() < deadline and checked['cost'] < report['cost']:
                    incumbent, report = deepcopy(candidate), checked
                    row['accepted'] = True
                if clock() >= deadline:
                    row['status'] = 'certification_deadline'
        row.update(cost_after=report['cost'], incumbent_after=deepcopy(incumbent),
                   step_wall_s=clock()-step_start, elapsed_s=clock()-started)
        attempts.append(row)
        if step == 1:
            first_cost = report['cost']
    wall_s = clock()-started
    late = wall_s >= time_limit_s
    if late:
        incumbent, report = deepcopy(initial_routes), initial
    actual_attempts = sum(r['status'] != 'not_started_deadline' for r in attempts)
    return {'routes': incumbent, 'report': report, 'wall_s': wall_s, 'timeout': late,
            'feature_s': feature_s, 'attempts': attempts, 'planned_attempts': 8,
            'actual_attempts': actual_attempts,
            'complete_candidates': sum(r['candidate_complete'] for r in attempts),
            'hard_candidates': sum(r['candidate_hard'] for r in attempts),
            'decode_calls': sum(r['decode_calls'] for r in attempts),
            'fixed_schedule_qualified': actual_attempts == 8 and not late,
            'first_endpoint_cost': first_cost, 'final_endpoint_cost': report['cost'],
            'learning_activated': provider is not None and sum(r['decode_calls'] for r in attempts) > 0,
            'online_gain_established': False,
            'deadline_mode': 'cooperative_soft_checks_with_initial_fallback_not_preemptive',
            'snapshot_physics': 'D2_isolation_reference_not_physical_calibration'}


def generate(seed, instance_id, customers=50):
    if customers != 50:
        raise ValueError('Research instances require fifty customers')
    rng = np.random.default_rng(np.random.SeedSequence([seed, instance_id]))
    data = {'coords': np.r_[np.array([[5., 5.]]), rng.uniform(0, 10, (50, 2))].tolist(),
            'demands': np.r_[0, rng.integers(1, 6, 50)].tolist(),
            'temp_class': np.r_[0, rng.integers(0, 3, 50)].tolist(),
            'reveal_time': [0.] * 51, 'ready_time': [0.] * 51,
            'tw_start': [0.] * 51, 'tw_end': [8.] * 51,
            'service_time': [0.] + [.03] * 50}
    return data, [list(range(1 + 10*k, 11 + 10*k)) for k in range(5)]


def parameter_digest(provider):
    from flax import nnx
    import jax
    leaves = jax.tree_util.tree_leaves(nnx.state(provider.model, nnx.Param))
    sha = hashlib.sha256()
    for leaf in leaves:
        array = np.ascontiguousarray(leaf)
        sha.update(str(array.shape).encode())
        sha.update(str(array.dtype).encode())
        sha.update(array.tobytes())
    return {'sha256': sha.hexdigest(), 'parameter_leaves': len(leaves)}


def verify_protocol(protocol):
    expected = {'version': 'D-EQ8-v1', 'dataset_seed': 2026101002,
                'mask_seed': 2026101041, 'arm_order_seed': 2026101099,
                'bootstrap_seed': 2026101077, 'bootstrap_repetitions': 2000,
                'random_model_seed': 42, 'steps': 8, 'mask_fraction': .3,
                'guide_weight': 1., 'time_limit_s': 10.}
    if any(protocol.get(k) != v or type(protocol.get(k)) is bool or
           (type(v) is int and type(protocol.get(k)) is not int) for k, v in expected.items()):
        raise ValueError('New registered D-EQ8-v1 fields differ; no mismatch bypass')
    if protocol.get('instance_ids') != list(range(32)) or any(type(i) is not int for i in protocol['instance_ids']):
        raise ValueError('All thirty-two unique instances required')
    if protocol.get('stage') not in ('synthetic_engineering_qa', 'fresh_mechanism_pilot'):
        raise ValueError('Declare permitted stage')
    if protocol.get('scope') != 'fixed_schedule_snapshot_attribution_no_online_gain':
        raise ValueError('Scope cannot become dynamic online gain')
    if protocol.get('stage') == 'fresh_mechanism_pilot' and not protocol.get('synthetic_qa_artifact_sha256'):
        raise ValueError('Freeze actual synthetic runtime QA before research data')
    return expected


def validate_qa(qa, identity):
    if qa.get('stage') != 'synthetic_engineering_qa' or qa.get('actual_engineering_qa_pass') is not True:
        raise ValueError('Synthetic QA did not qualify actual eight-step activation')
    rows = qa.get('rows', [])
    if len(rows) != 1 or type(rows[0].get('instance_id')) is not int or rows[0]['instance_id'] != -1:
        raise ValueError('Synthetic QA must contain only the isolated engineering input')
    if set(rows[0].get('arms',{})) != {'P','R','H'}:
        raise ValueError('Synthetic QA arms incomplete')
    for arm, result in rows[0]['arms'].items():
        if (result.get('fixed_schedule_qualified') is not True or result.get('timeout') is not False
            or result.get('report',{}).get('hard') is not True or result.get('actual_attempts') != 8
            or result.get('complete_candidates') != 8 or result.get('hard_candidates') != 8
            or result.get('decode_calls') != (0 if arm == 'H' else 8)
            or len(result.get('attempts',[])) != 8):
            raise ValueError('Actual synthetic arm failed eight-step completion/activation')
        for step, attempt in enumerate(result['attempts'],1):
            if (attempt.get('step') != step or attempt.get('candidate_complete') is not True
                or attempt.get('candidate_hard') is not True
                or attempt.get('retained_edges_preserved') is not True
                or attempt.get('decode_calls') != (0 if arm == 'H' else 1)):
                raise ValueError('Synthetic attempt evidence incomplete')
    for key in ('source_manifest_sha256','checkpoint_sha256','contract_sha256','model_parameters'):
        if qa.get('identity',{}).get(key) != identity.get(key):
            raise ValueError('Synthetic QA actual identity differs: '+key)


def main(args):
    raw = Path(args.protocol).read_bytes()
    protocol = json.loads(raw)
    verify_protocol(protocol)
    if protocol['stage'] == 'fresh_mechanism_pilot':
        if not args.synthetic_qa_artifact:
            raise ValueError('Actual successful synthetic QA result required')
        qa_raw = Path(args.synthetic_qa_artifact).read_bytes()
        qa = json.loads(qa_raw)
        if hashlib.sha256(qa_raw).hexdigest() != protocol['synthetic_qa_artifact_sha256']:
            raise ValueError('Actual synthetic QA artifact SHA mismatch')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    contract = DeliveryContract()
    manifest = json.loads(Path(args.source_manifest).read_text(encoding='utf-8'))
    source_root = Path(args.source_root).resolve()
    for name, expected in manifest.items():
        target = (source_root / name).resolve()
        if not target.is_relative_to(source_root) or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError('Actual source manifest mismatch: '+name)
    checkpoint = Path(args.checkpoint)
    weight_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if weight_sha != '03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db':
        raise ValueError('Original CVRP checkpoint mismatch')
    identity = {'protocol_sha256': hashlib.sha256(raw).hexdigest(),
                'source_manifest_sha256': digest(manifest), 'checkpoint_sha256': weight_sha,
                'contract_sha256': contract.identity(), 'pid': os.getpid(),
                'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                'threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                'claim_scope': protocol['scope']}
    providers = {'P': MaskCORouteProvider(checkpoint, args.upstream),
                 'R': MaskCORouteProvider(checkpoint, args.upstream, random_seed=42), 'H': None}
    for provider in (providers['P'], providers['R']):
        provider.prewarm(51, contract.capacity)
    parameters = {arm: parameter_digest(providers[arm]) for arm in ('P', 'R')}
    if parameters['P']['sha256'] == parameters['R']['sha256']:
        raise ValueError('Actual pretrained and random parameters identical')
    identity['model_parameters'] = parameters
    if protocol['stage'] == 'fresh_mechanism_pilot':
        validate_qa(qa, identity)
    dump_new(out / 'identity_start.json', identity)
    dump_new(out / 'PROTOCOL.json', protocol)
    dump_new(out / 'SOURCE_MANIFEST.json', manifest)
    rows = []
    # QA has one fixed synthetic input with same shape, independent of scientific seeds.
    ids = [-1] if protocol['stage'] == 'synthetic_engineering_qa' else protocol['instance_ids']
    with (out / 'progress.jsonl').open('x', encoding='utf-8') as f:
        for instance_id in ids:
            if instance_id == -1:
                coords = np.stack([np.arange(51)/10., np.zeros(51)], axis=1)
                data = {'coords': coords.tolist(), 'demands': [0]+[1]*50,
                        'temp_class': [0]+[i % 3 for i in range(50)],
                        'reveal_time': [0.]*51, 'ready_time': [0.]*51,
                        'tw_start': [0.]*51, 'tw_end': [8.]*51,
                        'service_time': [0.]+[.03]*50}
                initial = [list(range(1+10*k,11+10*k)) for k in range(5)]
                schedule = make_schedule(51,8,9001,0)
            else:
                data, initial = generate(protocol['dataset_seed'], instance_id)
                schedule = make_schedule(51,8,protocol['mask_seed'],instance_id)
            checked = evaluate(data, initial, contract)
            if not checked['hard']:
                raise RuntimeError('Initial data hard failure retained; no replacement')
            order = list(np.random.default_rng(np.random.SeedSequence(
                [protocol['arm_order_seed'], max(0,instance_id)])).permutation(['P','R','H']))
            arms = {}
            for arm in order:
                arms[arm] = run_arm(data, initial, contract, schedule, providers[arm],
                                    protocol['time_limit_s'], protocol['guide_weight'])
                replay = evaluate(data, arms[arm]['routes'], contract)
                if not replay['hard'] or abs(replay['cost'] - arms[arm]['report']['cost']) > 1e-9:
                    raise RuntimeError('Submitted route replay mismatch')
            row = {'instance_id': int(instance_id), 'data': data, 'initial_routes': initial,
                   'data_initial_sha256': digest([data,initial]), 'initial_report': checked,
                   'schedule': schedule, 'schedule_sha256': digest(schedule),
                   'arm_order': order, 'arms': arms}
            f.write(json.dumps(row, allow_nan=False)+'\n')
            f.flush()
            os.fsync(f.fileno())
            rows.append(row)
            print(json.dumps({'instance_id':int(instance_id),'completed':len(rows)}), flush=True)
    for name, expected in manifest.items():
        if hashlib.sha256((source_root/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Source changed during evaluation')
    for arm in ('P','R'):
        if parameter_digest(providers[arm]) != parameters[arm]:
            raise RuntimeError('Actual read-only model changed')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != weight_sha:
        raise RuntimeError('Checkpoint changed')
    dump_new(out/'identity_end.json',identity)
    qa_pass = (protocol['stage'] == 'synthetic_engineering_qa' and len(rows) == 1 and
               all(a['fixed_schedule_qualified'] and a['complete_candidates'] == 8 and
                   a['hard_candidates'] == 8 and a['decode_calls'] == (0 if arm == 'H' else 8)
                   for arm,a in rows[0]['arms'].items()))
    dump_new(out/'result.json', {'identity':identity,'stage':protocol['stage'],'rows':rows,
              'fixed_schedule_qualified':all(a['fixed_schedule_qualified'] for r in rows for a in r['arms'].values()),
              'actual_engineering_qa_pass':qa_pass,
              'engineering_only':protocol['stage']=='synthetic_engineering_qa',
              'online_gain_established':False,'independent_numeric_adjudication_pending':True})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('protocol','checkpoint','upstream','out','source-manifest','source-root'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--synthetic-qa-artifact')
    main(p.parse_args())
