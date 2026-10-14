"""D-TRAINTASK-v1 data-export feasibility: can the per-cycle reconstruction samples that a
delivery-specific trained component would need actually be produced and archived?

Why this is the first step rather than the training itself. Every delivery panel in both
cycles archived scalars (mean cost, cycle count, accepted count) and, in the confirm panel,
explicitly deleted the routes. No per-cycle decision sample - the incumbent, the mask that
was removed, and the certified change in contract cost - has ever been written to disk, so
no training set exists and none can be reconstructed from the archive. This panel proves the
export path works and records what a sample looks like, without training anything.

Scope: generates a small number of exported samples on ONE instance of a fresh seed purely
to demonstrate the export, and writes them durably. It trains nothing, evaluates nothing
about learned performance, and establishes no improvement. The registered D-TRAINTASK-v1
protocol deliberately carries no seeds yet, so this export run is engineering data and is
labelled as such rather than being treated as the protocol's TRAIN split.
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
for sub in ('delivery', 'coldchain', 'research_20261010'):
    sys.path.insert(0, str(SCRIPTS / sub))

from delivery_baselines import matrices
from delivery_contract import DeliveryContract
from delivery_evaluator import evaluate
from delivery_equal_schedule import generate, reconstruct
from delivery_maskco import partial_adjacency


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def export_cycles(instance, initial_routes, contract, time_limit_s, mask_fraction, seed,
                  clock=time.perf_counter):
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
    samples = []
    while clock() < deadline:
        removed = sorted(int(v) for v in rng.choice(np.arange(1, nodes), size=size,
                                                    replace=False))
        adjacency = partial_adjacency(best, removed, nodes)
        candidate, remaining, _ins, status = reconstruct(
            instance, best, contract, removed, adjacency, None, 0., deadline, clock)
        if remaining:
            if status == 'deadline' or clock() >= deadline:
                break
            samples.append({'outcome': 'incomplete', 'mask': removed})
            continue
        if clock() >= deadline:
            break
        checked = evaluate(instance, candidate, contract, deadline=deadline, clock=clock)
        accepted = bool(checked['hard'] and checked['cost'] < report['cost'])
        samples.append({
            'outcome': 'certified' if checked['hard'] else 'infeasible',
            'mask': removed,
            'mask_size': len(removed),
            'cost_before': float(report['cost']),
            'cost_after_candidate': (float(checked['cost']) if checked['hard'] else None),
            'delta_cost': (float(report['cost']) - float(checked['cost'])
                           if checked['hard'] else None),
            'accepted': accepted,
            'incumbent_routes_before': deepcopy(best),
            'cost_after_accepted': float(report['cost'])})
        if accepted:
            best, report = deepcopy([r for r in candidate if r]), checked
        if clock() >= deadline:
            break
    return {'samples': samples, 'final_cost': report['cost'],
            'wall_s': clock() - started}


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    contract = DeliveryContract()
    data, initial = generate(args.seed, args.instance)
    checked = evaluate(data, initial, contract)
    if not checked['hard']:
        raise RuntimeError('initial instance not hard feasible; retained, not replaced')
    result = export_cycles(data, initial, contract, args.time_limit, 0.5, args.mask_seed)
    rows = result['samples']
    certified = [r for r in rows if r['outcome'] == 'certified']
    with (out / 'samples.jsonl').open('x', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    summary = {
        'kind': 'traintask_data_export_feasibility_engineering_only',
        'isa_protocol_train_split': False,
        'seed': args.seed, 'instance': args.instance, 'mask_seed': args.mask_seed,
        'n_samples': len(rows), 'n_certified': len(certified),
        'n_accepted': sum(1 for r in rows if r['accepted']),
        'n_incomplete': sum(1 for r in rows if r['outcome'] == 'incomplete'),
        'n_infeasible': sum(1 for r in rows if r['outcome'] == 'infeasible'),
        'final_cost': result['final_cost'], 'wall_s': result['wall_s'],
        'sample_sha256': sha(rows[:50]),
        'what_a_sample_carries': ['the mask that was removed', 'the incumbent routes before',
                                  'the certified contract cost before and after',
                                  'whether the candidate was accepted'],
        'what_this_establishes': ('a per-cycle decision sample CAN be produced and archived '
                                  'durably, so the registered D-TRAINTASK-v1 protocol has a '
                                  'viable TRAIN-export path'),
        'what_this_does_not_establish': [
            'no model is trained and no learned performance is measured',
            'no improvement of any kind is claimed',
            'this engineering run is NOT the protocol TRAIN split and must not be reused as one'],
        'stable_gain_established': False,
        'maskco_attributable': False}
    (out / 'result.json').write_text(json.dumps(summary, indent=2, allow_nan=False),
                                     encoding='utf-8')
    print(json.dumps({k: summary[k] for k in
                      ('n_samples', 'n_certified', 'n_accepted', 'n_incomplete',
                       'n_infeasible', 'final_cost', 'wall_s')}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=202610101701)
    ap.add_argument('--mask-seed', type=int, default=202610101702)
    ap.add_argument('--instance', type=int, default=17000)
    ap.add_argument('--time-limit', type=float, default=5.0)
    main(ap.parse_args())
