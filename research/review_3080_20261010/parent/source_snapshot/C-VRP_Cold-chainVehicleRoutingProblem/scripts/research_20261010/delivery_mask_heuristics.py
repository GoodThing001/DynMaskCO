"""D-MASKHEUR-v1: does the CHOICE of which customers to mask carry any information?

Origin. The confirmed delivery base (ANY50, merged node n7) removes a uniformly random
50 percent of customers each cycle of a deadline-driven reconstruction loop. Its mask
choice has never been probed: D-DESTROY-v1 tried a guide-driven destroy (the lowest
scoring edges from the pretrained decoder) and lost, but no NON-LEARNED destroy heuristic
has ever been compared against uniform random on this base. That comparison is the cheap
screen that must come before training any learned mask selector: if no hand-designed mask
choice beats uniform random, a learned one has no signal to find, and the project's own
screening rule says to measure that before spending anything on a model.

Arms, all on the identical confirmed base - deadline-driven reconstruction, 50 percent
mask, 10.0s wall budget, no neural decoding anywhere:
  random      uniform random mask (the confirmed base, unchanged)
  far         destroy the `size` customers farthest from the depot
  long_edge   destroy the endpoints of the `size`-proportional longest edges of the
              incumbent plan, depot legs included
  near        destroy the `size` customers nearest the depot (an anti-heuristic: it should
              be worse, which makes it a useful direction check)

Registered reading: any heuristic counts as carrying signal only if it beats `random` with
mean relative gain >= 1 percent and paired 95 percent CI lower > 0 on the dev cohort, in
which case that arm is carried into a fresh-seed confirmation cohort. If none does, the
mask-choice channel is closed and no learned mask selector should be trained.

Scope: non-learning throughout, pre-dispatch snapshot with simulated parent physics, no
online claim and no MaskCO-attributable gain.

Usage:
    python delivery_mask_heuristics.py --protocol P.json --out NEWDIR
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
for sub in ('delivery', 'coldchain', 'research_20261010'):
    sys.path.insert(0, str(SCRIPTS / sub))

from delivery_baselines import matrices
from delivery_contract import DeliveryContract
from delivery_evaluator import evaluate
from delivery_equal_schedule import digest, dump_new, generate, reconstruct
from delivery_maskco import partial_adjacency

PROTOCOL = {
    'version': 'D-MASKHEUR-v1',
    'kind': 'mask_choice_signal_screen_not_gain_claim',
    'base': 'confirmed ANY50 (deadline-driven reconstruction, 50 percent mask, 10s)',
    'arms': ['random', 'far', 'long_edge', 'near'],
    'dataset_seed_dev': 2026101002,
    'dataset_seed_confirmation': 2026101013,
    'dev_instance_ids': list(range(32)),
    'confirmation_instance_ids': list(range(1000, 1032)),
    'anytime_seed_dev': 2026101131,
    'anytime_seed_confirmation': 2026101174,
    'mask_fraction': 0.5,
    'time_limit_s': 10.0,
    'bootstrap_seed': 2026101231,
    'bootstrap_repetitions': 2000,
    'signal_threshold': 0.01,
    'selection_rule': ('carry an arm into confirmation only if it beats random on dev with '
                       'mean>=1 percent and paired 95 percent CI lower>0'),
    'claims_online_gain': False,
    'all_failures_retained': True,
    'physics_note': 'simulated parent-physics snapshot; not physical calibration',
}


def verify_protocol(p):
    expected = {'version': 'D-MASKHEUR-v1', 'dataset_seed_dev': 2026101002,
                'dataset_seed_confirmation': 2026101013, 'anytime_seed_dev': 2026101131,
                'anytime_seed_confirmation': 2026101174, 'mask_fraction': .5,
                'time_limit_s': 10., 'bootstrap_seed': 2026101231,
                'bootstrap_repetitions': 2000, 'signal_threshold': .01,
                'arms': ['random', 'far', 'long_edge', 'near']}
    for k, v in expected.items():
        got = p.get(k)
        if isinstance(v, bool):
            if got is not v:
                raise ValueError('D-MASKHEUR-v1 registered field changed: ' + k)
        elif got != v or isinstance(got, bool):
            raise ValueError('D-MASKHEUR-v1 registered field changed: ' + k)
    if p.get('dev_instance_ids') != list(range(32)):
        raise ValueError('Dev cohort ids are registered and fixed')
    if p.get('confirmation_instance_ids') != list(range(1000, 1032)):
        raise ValueError('Confirmation cohort ids are registered and fixed')
    if p.get('claims_online_gain') is not False:
        raise ValueError('Scope cannot become an online gain claim')
    return expected


def choose_mask(strategy, rng, best, dist, nodes, size):
    served = sorted({int(o) for route in best for o in route})
    if not served:
        return set()
    k = min(size, len(served))
    if strategy == 'random':
        return set(map(int, rng.choice(np.arange(1, nodes), size=size, replace=False)))
    if strategy == 'far':
        return set(sorted(served, key=lambda o: -float(dist[0, o]))[:k])
    if strategy == 'near':
        return set(sorted(served, key=lambda o: float(dist[0, o]))[:k])
    if strategy == 'long_edge':
        edges = []
        for route in best:
            chain = [0] + list(route) + [0]
            for a, b in zip(chain, chain[1:]):
                if a != b:
                    edges.append((float(dist[a, b]), int(a), int(b)))
        edges.sort(key=lambda row: (-row[0], row[1], row[2]))
        removed = set()
        need = max(1, int(math.ceil(size / 2.0)))
        for _w, a, b in edges:
            if len(removed) >= need:
                break
            for node in (a, b):
                # The depot is not a customer and must never enter a removal mask.
                if node != 0 and len(removed) < need:
                    removed.add(node)
        if len(removed) < size:
            pool = [o for o in served if o not in removed]
            extra = rng.choice(pool, size=min(size - len(removed), len(pool)), replace=False)
            removed.update(int(o) for o in extra)
        return removed
    raise ValueError('unknown strategy')


def anytime(instance, initial_routes, contract, time_limit_s, mask_fraction, seed,
            strategy, clock=time.perf_counter):
    started = clock()
    deadline = started + time_limit_s
    initial = evaluate(instance, initial_routes, contract)
    if not initial['hard']:
        raise ValueError('Unqualified common initial incumbent')
    dist = matrices(instance)
    nodes = len(dist)
    rng = np.random.default_rng(seed)
    size = max(1, int(math.ceil(mask_fraction * (nodes - 1))))
    best, report = deepcopy(initial_routes), initial
    cycles = complete = hard = accepted = 0
    while clock() < deadline:
        removed = choose_mask(strategy, rng, best, dist, nodes, size)
        if not removed:
            break
        adjacency = partial_adjacency(best, removed, nodes)
        candidate, remaining, _ins, status = reconstruct(
            instance, best, contract, removed, adjacency, None, 0., deadline, clock)
        cycles += 1
        if remaining:
            if status == 'deadline' or clock() >= deadline:
                break
            continue
        complete += 1
        if clock() >= deadline:
            break
        checked = evaluate(instance, candidate, contract, deadline=deadline, clock=clock)
        if checked['hard']:
            hard += 1
            if checked['cost'] < report['cost']:
                best, report = deepcopy([r for r in candidate if r]), checked
                accepted += 1
        if clock() >= deadline:
            break
    return {'routes': best, 'final_cost': report['cost'], 'wall_s': clock() - started,
            'cycles': cycles, 'complete_candidates': complete, 'hard_candidates': hard,
            'accepted': accepted, 'decode_calls': 0, 'strategy': strategy}


def run_cohort(cohort, instance_ids, dataset_seed, anytime_seed, protocol, contract, arms):
    rows = []
    for instance_id in instance_ids:
        data, initial = generate(dataset_seed, instance_id)
        checked = evaluate(data, initial, contract)
        if not checked['hard']:
            raise RuntimeError('Initial data hard failure retained: %d' % instance_id)
        out = {}
        for strategy in arms:
            res = anytime(data, initial, contract, protocol['time_limit_s'],
                          protocol['mask_fraction'], anytime_seed, strategy)
            replay = evaluate(data, res['routes'], contract)
            if not replay['hard'] or abs(replay['cost'] - res['final_cost']) > 1e-9:
                raise RuntimeError('Arm %s replay mismatch: %d' % (strategy, instance_id))
            out[strategy] = {k: v for k, v in res.items() if k != 'routes'}
        rows.append({'cohort': cohort, 'instance_id': int(instance_id),
                     'initial_cost': checked['cost'], 'arms': out})
        print(json.dumps({'cohort': cohort, 'instance_id': instance_id,
                          'completed': len(rows),
                          **{s: round(out[s]['final_cost'], 4) for s in arms}}), flush=True)
    return rows


def paired(control, other, seed, reps, name_control, name_other):
    shared = sorted(set(control) & set(other))
    diffs = np.array([(control[i] - other[i]) / control[i] for i in shared], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(reps, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return {'control': name_control, 'other': name_other, 'n_instances': len(shared),
            'mean_relative_gain': float(diffs.mean()),
            'descriptive_95_CI': [float(np.percentile(means, 2.5)),
                                  float(np.percentile(means, 97.5))],
            'instances_other_better': int(sum(d > 0 for d in diffs)),
            'instances_other_worse': int(sum(d < 0 for d in diffs)),
            'instances_tied': int(sum(d == 0 for d in diffs))}


def summarise(rows, cohort, arms, protocol):
    table = {a: {'n': len(rows),
                 'mean_final_cost': statistics.fmean(r['arms'][a]['final_cost'] for r in rows),
                 'mean_wall_s': statistics.fmean(r['arms'][a]['wall_s'] for r in rows),
                 'total_cycles': sum(r['arms'][a]['cycles'] for r in rows),
                 'total_accepted': sum(r['arms'][a]['accepted'] for r in rows),
                 'total_decode_calls': sum(r['arms'][a]['decode_calls'] for r in rows)}
             for a in arms}
    by = {a: {r['instance_id']: r['arms'][a]['final_cost'] for r in rows} for a in arms}
    gains = {a: paired(by['random'], by[a], protocol['bootstrap_seed'],
                       protocol['bootstrap_repetitions'], 'random', a)
             for a in arms if a != 'random'}
    return {'cohort': cohort, 'table': table, 'gains_vs_random': gains}


def main(args):
    started = time.perf_counter()
    protocol = json.loads(Path(args.protocol).read_text(encoding='utf-8'))
    verify_protocol(protocol)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    contract = DeliveryContract()
    dump_new(out / 'PROTOCOL.json', protocol)
    arms = protocol['arms']

    dev_ids = protocol['dev_instance_ids']
    if args.max_instances is not None:
        dev_ids = dev_ids[:args.max_instances]
    dev_rows = run_cohort('dev', dev_ids, protocol['dataset_seed_dev'],
                          protocol['anytime_seed_dev'], protocol, contract, arms)
    with (out / 'dev_progress.jsonl').open('x', encoding='utf-8') as f:
        for row in dev_rows:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    dev = summarise(dev_rows, 'dev', arms, protocol)
    thr = protocol['signal_threshold']
    carried = [a for a, g in dev['gains_vs_random'].items()
               if g['mean_relative_gain'] >= thr and g['descriptive_95_CI'][0] > 0]
    result = {'protocol': protocol, 'dev': dev, 'arms_with_signal': sorted(carried),
              'mask_choice_channel_open': bool(carried)}
    if carried:
        confirm_arms = ['random'] + sorted(carried)
        conf_rows = run_cohort('confirmation', protocol['confirmation_instance_ids'],
                               protocol['dataset_seed_confirmation'],
                               protocol['anytime_seed_confirmation'], protocol, contract,
                               confirm_arms)
        with (out / 'confirmation_progress.jsonl').open('x', encoding='utf-8') as f:
            for row in conf_rows:
                f.write(json.dumps(row, allow_nan=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        result['confirmation'] = summarise(conf_rows, 'confirmation', confirm_arms, protocol)
        result['confirmation_run'] = True
    else:
        result['confirmation'] = None
        result['confirmation_run'] = False
        result['outcome'] = 'no_non_learned_mask_heuristic_beats_uniform_random'
    result['source_hashes'] = {str(Path(__file__).name):
                               hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result['online_gain_established'] = False
    result['maskco_attributable_gain_established'] = False
    result['stable_gain_established'] = False
    result['wall_s'] = time.perf_counter() - started
    dump_new(out / 'result.json', result)
    print(json.dumps({'dev_means': {a: round(t['mean_final_cost'], 4)
                                    for a, t in dev['table'].items()},
                      'dev_gains': {a: [round(g['mean_relative_gain'], 4),
                                        [round(c, 4) for c in g['descriptive_95_CI']]]
                                    for a, g in dev['gains_vs_random'].items()},
                      'arms_with_signal': sorted(carried),
                      'outcome': result.get('outcome', 'carried_to_confirmation'),
                      'wall_s': round(result['wall_s'], 1)}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-instances', type=int, default=None)
    main(ap.parse_args())
