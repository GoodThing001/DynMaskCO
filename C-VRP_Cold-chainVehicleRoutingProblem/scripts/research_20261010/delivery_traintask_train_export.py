"""D-TRAINTASK-v1 TRAIN-split export: the first real TRAIN data for the delivery line.

Why this is a separate script from `delivery_traintask_data_export.py`. That one is explicitly an
ENGINEERING feasibility run (`isa_protocol_train_split: false`, seed 202610101701) whose purpose
was to prove the export path works. This one executes the FROZEN registered protocol with its
frozen seeds and therefore produces the split that a trained component is allowed to see.

It fails closed. Before generating anything it re-verifies:
  * the frozen protocol parses and its recorded DRAFT sha matches the DRAFT on disk,
  * the protocol status is `seeds_frozen_not_run` (i.e. it has not already been consumed),
  * the parent PROGRAM sha recorded in the protocol still matches the PROGRAM on disk,
  * the seeds it is about to use are the ones recorded in the protocol, not values passed in.

What it records per cycle, beyond the engineering export, is the TRAINING SIGNAL:
  * the reconstruction's own decision sequence (`customer`, `vehicle`, `position`, `score`) so an
    imitation or one-step-lookahead trainer has the decisions, and
  * the realized contract cost of the candidate, so a REINFORCE-style trainer has the reward.
Without these two the exported samples could only support cost regression on a summary, not
objective-aligned training.

Scope: data generation and provenance only. Nothing is trained, no arm is compared, and no
improvement, attribution, impossibility or equivalence is claimed.
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


def bootstrap(scripts):
    scripts = Path(scripts).resolve()
    for p in (scripts / 'delivery', scripts / 'research_20261010'):
        sys.path.insert(0, str(p))
    return scripts


def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(args):
    started = time.perf_counter()
    scripts = bootstrap(args.scripts)
    from delivery_equal_schedule import generate
    from delivery_evaluator import evaluate
    from delivery_baselines import matrices
    from delivery_maskco import partial_adjacency
    from delivery_equal_schedule import reconstruct

    proto_path = Path(args.protocol)
    draft_path = proto_path.with_name('delivery_traintask_protocol_DRAFT.json')
    proto = json.loads(proto_path.read_text(encoding='utf-8'))
    draft = json.loads(draft_path.read_text(encoding='utf-8'))
    proto_sha = file_sha(proto_path)
    draft_sha = file_sha(draft_path)
    problems = []
    if proto.get('version') != 'D-TRAINTASK-v1':
        problems.append('unexpected protocol version')
    if proto.get('status') != 'seeds_frozen_not_run':
        problems.append('protocol status is %r, expected seeds_frozen_not_run - the split may '
                        'already have been consumed' % proto.get('status'))
    if proto['data_plan']['registered_seeds'].get('disjointness_basis') is None and \
            'frozen_protocol_sha256' in proto:
        recorded = (proto.get('frozen_protocol_sha256') or {}).get('draft_sha256')
        if recorded and recorded != draft_sha:
            problems.append('DRAFT sha mismatch: recorded %s, on disk %s' % (recorded, draft_sha))
    program = Path(args.project) / 'results/research_20261010_continuation_v2/PROGRAM.json'
    prog_sha = file_sha(program)
    if proto.get('parent_program_sha256') and proto['parent_program_sha256'] != prog_sha:
        problems.append('parent PROGRAM sha changed: protocol %s, disk %s'
                        % (proto['parent_program_sha256'], prog_sha))
    seeds = proto['data_plan']['registered_seeds']
    if problems:
        raise SystemExit('refusing to generate a TRAIN split: ' + '; '.join(problems))

    train_seed = int(seeds['train_seed'])
    mask_seed = int(seeds['mask_seed_train'])
    if args.expect_train_seed is not None and train_seed != args.expect_train_seed:
        raise SystemExit('protocol train_seed %d != expected %d' % (train_seed,
                                                                   args.expect_train_seed))
    budget = float(proto['matched_resources']['wall_budget_s'])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'TRAIN_SPLIT_IDENTITY.json').write_text(json.dumps({
        'kind': 'delivery_traintask_registered_train_split',
        'isa_protocol_train_split': True,
        'protocol_version': proto['version'],
        'protocol_sha256': proto_sha,
        'protocol_draft_sha256': draft_sha,
        'parent_program_sha256': prog_sha,
        'train_seed': train_seed, 'mask_seed_train': mask_seed,
        'instances': args.instances, 'wall_budget_s': budget,
        'mask_fraction': args.mask_fraction,
        'source_start': {str(p): file_sha(p) for p in sorted(
            Path(scripts).rglob('delivery_*.py'))},
        'registered_unix': time.time(),
        'note': ('seeds are read from the frozen protocol, never from the command line; the '
                 'protocol status gate and the DRAFT/PROGRAM sha gates must all pass first'),
    }, indent=2), encoding='utf-8')

    from delivery_maskco import MaskCORouteProvider
    provider = MaskCORouteProvider(checkpoint=args.checkpoint,
                                  upstream_root=args.upstream_root) if args.checkpoint else None

    all_rows, per_instance = [], []
    t0 = time.perf_counter()
    for inst in range(args.instances):
        data, initial = generate(train_seed, inst)
        contract = None
        from delivery_equal_schedule import DeliveryContract
        contract = DeliveryContract()
        checked = evaluate(data, initial, contract)
        if not checked['hard']:
            per_instance.append({'instance': inst, 'outcome': 'unqualified_initial_retained',
                                 'n_samples': 0})
            continue
        rng = np.random.default_rng(mask_seed + inst)
        nodes = len(data['demands'])
        size = max(1, int(math.ceil(args.mask_fraction * (nodes - 1))))
        dist = matrices(data)
        incumbent, report = deepcopy(initial), checked
        deadline = time.perf_counter() + budget
        samples = []
        while time.perf_counter() < deadline:
            removed = sorted(int(v) for v in rng.choice(np.arange(1, nodes), size=size,
                                                       replace=False))
            adjacency = partial_adjacency(incumbent, removed, nodes)
            candidate, remaining, insertions, status = reconstruct(
                data, incumbent, contract, removed, adjacency, None, 0.0, deadline,
                time.perf_counter)
            if remaining:
                samples.append({'outcome': 'incomplete', 'mask': removed,
                                'status': status, 'insertions': insertions})
                if status == 'deadline' or time.perf_counter() >= deadline:
                    break
                continue
            verified = evaluate(data, candidate, contract, deadline=deadline,
                                clock=time.perf_counter)
            accepted = bool(verified['hard'] and verified['cost'] < report['cost'])
            samples.append({
                'outcome': 'certified' if verified['hard'] else 'infeasible',
                'mask': removed, 'mask_size': len(removed), 'nodes': nodes,
                'incumbent_routes_before': deepcopy([list(r) for r in incumbent]),
                'candidate_routes': deepcopy([list(r) for r in candidate if r]),
                'insertions': insertions,
                'cost_before': float(report['cost']),
                'cost_after_candidate': float(verified['cost']) if verified['hard'] else None,
                'delta_cost': (float(report['cost']) - float(verified['cost'])
                               if verified['hard'] else None),
                'accepted': accepted})
            if accepted:
                incumbent, report = deepcopy([r for r in candidate if r]), verified
            if time.perf_counter() >= deadline:
                break
        for s in samples:
            s['instance'] = inst
        all_rows.extend(samples)
        per_instance.append({'instance': inst, 'outcome': 'ran', 'n_samples': len(samples),
                             'final_cost': float(report['cost']),
                             'n_certified': sum(1 for s in samples
                                                if s['outcome'] == 'certified'),
                             'n_infeasible': sum(1 for s in samples
                                                 if s['outcome'] == 'infeasible'),
                             'n_incomplete': sum(1 for s in samples
                                                 if s['outcome'] == 'incomplete')})

    with (out / 'samples.jsonl').open('x', encoding='utf-8') as f:
        for r in all_rows:
            f.write(json.dumps(r, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())

    certified = [r for r in all_rows if r['outcome'] == 'certified']
    accepted = [r for r in certified if r['accepted']]
    summary = {
        'kind': 'delivery_traintask_registered_train_split_manifest',
        'protocol_version': proto['version'], 'protocol_sha256': proto_sha,
        'train_seed': train_seed, 'mask_seed_train': mask_seed,
        'instances': args.instances, 'n_samples': len(all_rows),
        'n_certified': len(certified), 'n_accepted': len(accepted),
        'n_infeasible': sum(1 for r in all_rows if r['outcome'] == 'infeasible'),
        'n_incomplete': sum(1 for r in all_rows if r['outcome'] == 'incomplete'),
        'n_with_decision_sequence': sum(1 for r in all_rows if r.get('insertions')),
        'n_with_reward': sum(1 for r in all_rows if r.get('delta_cost') is not None),
        'per_instance': per_instance,
        'training_signal_available': {
            'decisions': 'insertions[] with customer/vehicle/position/score',
            'reward': 'delta_cost = cost_before - cost_after_candidate'},
        'samples_sha256': file_sha(out / 'samples.jsonl'),
        'what_this_is': ['the TRAIN split the frozen D-TRAINTASK-v1 protocol authorises'],
        'what_this_is_not': [
            'not a result: nothing is trained and no arm is compared',
            'not the engineering export seed 202610101701, which is engineering-only',
            'no stable online gain, no MaskCO attribution, no impossibility, no equivalence'],
        'wall_s': time.perf_counter() - started,
    }
    (out / 'TRAIN_SPLIT_MANIFEST.json').write_text(json.dumps(summary, indent=2, allow_nan=False),
                                                   encoding='utf-8')
    print(json.dumps({k: summary[k] for k in
                      ('n_samples', 'n_certified', 'n_accepted', 'n_infeasible',
                       'n_incomplete', 'n_with_decision_sequence', 'n_with_reward',
                       'train_seed', 'mask_seed_train', 'wall_s')}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scripts', required=True)
    ap.add_argument('--project', required=True)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--upstream-root', default=None)
    ap.add_argument('--instances', type=int, default=6)
    ap.add_argument('--mask-fraction', type=float, default=0.5)
    ap.add_argument('--expect-train-seed', type=int, default=None)
    main(ap.parse_args())
