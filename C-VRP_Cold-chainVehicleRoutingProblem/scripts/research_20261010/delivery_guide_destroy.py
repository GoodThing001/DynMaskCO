"""D-DESTROY-v1: does a pretrained guide add value if it is ONLY a destroy operator?

Motivation, and why this is not a repeat of a falsified experiment. Three independent
delivery results closed the "blend a guide into the insertion score" family: D-EQ8-v1
(pretrained guide worse than random and than no guide), D-GUIDE-v1 (harmful in both the
regret and the position role, position dominant) and D-DOSE-v1 (harm monotone in guide
weight, pretrained direction worse than random direction). The direction-level insight
recorded at D-GUIDE-v1 states the one untested alternative explicitly: a guide must be
either excluded from the decision score entirely -- used only to generate candidates --
or expressed in the same units as the objective. D-DESTROY-v1 tests the first branch.

Intervention. The anytime masked reconstructor that passed the held-out gate
(D-ANY50-CONFIRM-v1: 50 percent mask, deadline-driven, non-learning) chooses the 25
customers to remove uniformly at random each cycle. This panel changes ONLY that choice:

  random      uniform random destroy (the current M_best mechanism)
  guided      the pretrained CVRP decoder scores every edge of the incumbent; the
              lowest-scoring edges are destroyed first, their endpoints removed until
              the mask size is reached; ties and shortfalls fall back to the seeded RNG
  guided_rand identical to 'guided' but with a random same-architecture model

No guide ever enters the insertion score, the regret ranking or the acceptance test:
those remain the identical distance-based regret-2 insertion and the identical certified
contract-cost improvement rule used by the confirmed base.

Honesty about the budget. The guided arms pay one decoder call per cycle, so at an equal
10.0s wall budget they complete far fewer cycles than the random arm. This is a real
trade-off and is reported as such: per-arm consumed wall time, cycles and decode calls are
all recorded, and the panel does NOT claim equal consumed compute between arms -- that
was exactly the error D-CTRL-v1 made and D-ANYTIME-v1 corrected.

Registered rule, fixed before data:
  * 'guided' must beat 'random' with mean relative gain >= 1% and paired 95% CI lower > 0
  * only then is the confirmation cohort run; otherwise the panel closes negative and the
    confirmation cohort is not spent
  * 'guided' vs 'guided_rand' attributes any gain to the pretrained weights rather than to
    the destroy mechanism itself

Usage:
    python delivery_guide_destroy.py --protocol P.json --checkpoint CKPT --upstream ROOT \
        --recorded-identity ID.json --out NEWDIR
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
from delivery_equal_schedule import dump_new, generate, parameter_digest, reconstruct
from delivery_maskco import MaskCORouteProvider, partial_adjacency

PROTOCOL = {
    'version': 'D-DESTROY-v1',
    'kind': 'candidate_generation_guide_test_not_gain_claim',
    'parent_panels': ['D-EQ8-v1', 'D-GUIDE-v1', 'D-DOSE-v1', 'D-ANYTIME-v1',
                      'D-ANY50-CONFIRM-v1'],
    'guide_role': 'destroy_operator_only_never_in_decision_score',
    'base_mechanism': 'anytime masked reconstruction, 50 percent mask, deadline-driven',
    'dataset_seed_dev': 2026101161,
    'dataset_seed_confirmation': 2026101171,
    'dev_instance_ids': list(range(3000, 3032)),
    'confirmation_instance_ids': list(range(4000, 4032)),
    'anytime_seed_dev': 2026101163,
    'anytime_seed_confirmation': 2026101173,
    'mask_fraction': 0.5,
    'time_limit_s': 10.0,
    'random_model_seed': 42,
    'bootstrap_seed': 2026101181,
    'bootstrap_repetitions': 2000,
    'guide_threshold': 0.01,
    'selection_rule': 'run confirmation only if guided beats random on dev with mean>=1% and CI lower>0',
    'claims_equal_consumed_compute': False,
    'claims_online_gain': False,
    'all_failures_retained': True,
    'physics_note': 'simulated parent-physics snapshot; not physical calibration',
}


def verify_protocol(p):
    expected = {'version': 'D-DESTROY-v1', 'dataset_seed_dev': 2026101161,
                'dataset_seed_confirmation': 2026101171, 'anytime_seed_dev': 2026101163,
                'anytime_seed_confirmation': 2026101173, 'mask_fraction': .5,
                'time_limit_s': 10., 'random_model_seed': 42,
                'bootstrap_seed': 2026101181, 'bootstrap_repetitions': 2000,
                'guide_threshold': .01, 'claims_equal_consumed_compute': False}
    for k, v in expected.items():
        got = p.get(k)
        if isinstance(v, bool):
            if got is not v:
                raise ValueError('D-DESTROY-v1 registered field changed: ' + k)
        elif got != v or isinstance(got, bool):
            raise ValueError('D-DESTROY-v1 registered field changed: ' + k)
    if p.get('dev_instance_ids') != list(range(3000, 3032)):
        raise ValueError('Dev cohort ids are registered and fixed')
    if p.get('confirmation_instance_ids') != list(range(4000, 4032)):
        raise ValueError('Confirmation cohort ids are registered and fixed')
    if p.get('claims_online_gain') is not False:
        raise ValueError('Scope cannot become an online gain claim')
    if p.get('guide_role') != 'destroy_operator_only_never_in_decision_score':
        raise ValueError('The guide role is the registered intervention and cannot change')
    return expected


def random_destroy(rng, nodes, size):
    return set(map(int, rng.choice(np.arange(1, nodes), size=size, replace=False)))


def guided_destroy(logits, incumbent, nodes, size, rng):
    """Destroy the edges the model scores lowest; endpoints only, never depot."""
    edges = []
    for route in incumbent:
        chain = [0] + list(route) + [0]
        for a, b in zip(chain, chain[1:]):
            if a != b:
                edges.append((float(logits[a, b]), int(a), int(b)))
    edges.sort(key=lambda row: (row[0], row[1], row[2]))
    removed = set()
    for _score, a, b in edges:
        if len(removed) >= size:
            break
        for node in (a, b):
            if node != 0 and len(removed) < size:
                removed.add(node)
    if len(removed) < size:
        pool = [c for c in range(1, nodes) if c not in removed]
        extra = rng.choice(pool, size=size - len(removed), replace=False)
        removed.update(int(c) for c in extra)
    return removed


def anytime(instance, initial_routes, contract, time_limit_s, mask_fraction, seed,
            destroy, provider=None, features=None, clock=time.perf_counter):
    """Deadline-driven masked reconstruction; only the destroy rule differs by arm."""
    started = clock()
    deadline = started + time_limit_s
    initial = evaluate(instance, initial_routes, contract)
    if not initial['hard']:
        raise ValueError('Unqualified common initial incumbent')
    nodes = len(instance['demands'])
    rng = np.random.default_rng(seed)
    size = max(1, int(math.ceil(mask_fraction * (nodes - 1))))
    best, report = deepcopy(initial_routes), initial
    cycles = complete = hard = accepted = decodes = 0
    while clock() < deadline:
        if destroy == 'random':
            removed = random_destroy(rng, nodes, size)
        else:
            full = partial_adjacency(best, set(), nodes)
            logits = provider.logits(features, full)
            decodes += 1
            if logits.shape != (nodes, nodes) or not np.all(np.isfinite(
                    logits[~np.eye(nodes, dtype=bool)])):
                raise ValueError('Actual model logits invalid')
            removed = guided_destroy(logits, best, nodes, size, rng)
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
            'accepted': accepted, 'decode_calls': decodes,
            'mask_size': size, 'destroy': destroy}


def run_cohort(cohort, instance_ids, dataset_seed, anytime_seed, protocol, contract,
               providers):
    rows = []
    for instance_id in instance_ids:
        data, initial = generate(dataset_seed, instance_id)
        checked = evaluate(data, initial, contract)
        if not checked['hard']:
            raise RuntimeError('Initial data hard failure retained: %d' % instance_id)
        arms = {}
        specs = [('random', None, None)]
        for name, arm in (('guided', 'P'), ('guided_rand', 'R')):
            specs.append((name, arm, providers[arm]))
        for name, arm, provider in specs:
            kwargs = {}
            if provider is not None:
                feats = provider.features(data, contract.capacity)
                if hasattr(feats, 'block_until_ready'):
                    feats.block_until_ready()
                kwargs = {'provider': provider, 'features': feats}
            out = anytime(data, initial, contract, protocol['time_limit_s'],
                          protocol['mask_fraction'], anytime_seed, name, **kwargs)
            replay = evaluate(data, out['routes'], contract)
            if not replay['hard'] or abs(replay['cost'] - out['final_cost']) > 1e-9:
                raise RuntimeError('Arm %s replay mismatch: %d' % (name, instance_id))
            arms[name] = {k: v for k, v in out.items() if k != 'routes'}
        rows.append({'cohort': cohort, 'instance_id': int(instance_id),
                     'initial_cost': checked['cost'], 'arms': arms})
        print(json.dumps({'cohort': cohort, 'instance_id': instance_id,
                          'completed': len(rows),
                          'random': round(arms['random']['final_cost'], 4),
                          'guided': round(arms['guided']['final_cost'], 4),
                          'guided_rand': round(arms['guided_rand']['final_cost'], 4)}),
              flush=True)
    return rows


def paired(control, other, seed, reps, name_control, name_other):
    shared = sorted(set(control) & set(other))
    diffs = np.array([(control[i] - other[i]) / control[i] for i in shared], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(reps, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return {'n_instances': len(shared), 'control': name_control, 'other': name_other,
            'mean_relative_gain': float(diffs.mean()),
            'descriptive_95_CI': [float(np.percentile(means, 2.5)),
                                  float(np.percentile(means, 97.5))],
            'instances_other_better': int(sum(d > 0 for d in diffs)),
            'instances_other_worse': int(sum(d < 0 for d in diffs)),
            'instances_tied': int(sum(d == 0 for d in diffs))}


def summarise(rows, cohort, protocol):
    table = {}
    for arm in ('random', 'guided', 'guided_rand'):
        table[arm] = {
            'n': len(rows),
            'mean_final_cost': statistics.fmean(r['arms'][arm]['final_cost'] for r in rows),
            'mean_wall_s': statistics.fmean(r['arms'][arm]['wall_s'] for r in rows),
            'total_cycles': sum(r['arms'][arm]['cycles'] for r in rows),
            'total_accepted': sum(r['arms'][arm]['accepted'] for r in rows),
            'total_decode_calls': sum(r['arms'][arm]['decode_calls'] for r in rows),
            'n_decode_calls_zero': sum(r['arms'][arm]['decode_calls'] == 0 for r in rows)}
    by = {a: {r['instance_id']: r['arms'][a]['final_cost'] for r in rows}
          for a in ('random', 'guided', 'guided_rand')}
    bs, reps = protocol['bootstrap_seed'], protocol['bootstrap_repetitions']
    gains = {'guided_over_random': paired(by['random'], by['guided'], bs, reps, 'random', 'guided'),
             'guided_rand_over_random': paired(by['random'], by['guided_rand'], bs, reps,
                                               'random', 'guided_rand'),
             'guided_over_guided_rand': paired(by['guided_rand'], by['guided'], bs, reps,
                                               'guided_rand', 'guided')}
    thr = protocol['guide_threshold']
    def verdict(g):
        lo, hi = g['descriptive_95_CI']
        if g['mean_relative_gain'] >= thr and lo > 0:
            return 'other_dominates'
        if g['mean_relative_gain'] <= -thr and hi < 0:
            return 'control_dominates'
        return 'indistinguishable'
    return {'cohort': cohort, 'table': table, 'gains': gains,
            'verdicts': {k: verdict(v) for k, v in gains.items()}}


def main(args):
    started = time.perf_counter()
    protocol = json.loads(Path(args.protocol).read_text(encoding='utf-8'))
    verify_protocol(protocol)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    contract = DeliveryContract()
    checkpoint = Path(args.checkpoint)
    recorded = json.loads(Path(args.recorded_identity).read_text(encoding='utf-8'))
    weight_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if weight_sha != recorded['checkpoint_sha256']:
        raise ValueError('Checkpoint does not match the recorded identity')
    providers = {'P': MaskCORouteProvider(str(checkpoint), args.upstream),
                 'R': MaskCORouteProvider(str(checkpoint), args.upstream,
                                          random_seed=protocol['random_model_seed'])}
    local = {a: parameter_digest(providers[a]) for a in ('P', 'R')}
    for a in ('P', 'R'):
        if local[a] != recorded['model_parameters'][a]:
            raise ValueError('Local model parameters differ from the recorded identity: ' + a)
        providers[a].prewarm(51, contract.capacity)
    dump_new(out / 'PROTOCOL.json', protocol)
    dump_new(out / 'identity_start.json', {
        'checkpoint_sha256': weight_sha, 'local_model_parameters': local,
        'recorded_model_parameters': recorded['model_parameters'],
        'guide_role': protocol['guide_role'],
        'threads': {k: os.environ.get(k) for k in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}})
    result = {'protocol': protocol,
              'smoke_test_truncated': args.max_instances is not None}
    dev_ids = protocol['dev_instance_ids']
    if args.max_instances is not None:
        dev_ids = dev_ids[:args.max_instances]
    rows = run_cohort('dev', dev_ids, protocol['dataset_seed_dev'],
                      protocol['anytime_seed_dev'], protocol, contract, providers)
    with (out / 'dev_progress.jsonl').open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    result['dev'] = summarise(rows, 'dev', protocol)
    g = result['dev']['gains']['guided_over_random']
    selected = (g['mean_relative_gain'] >= protocol['guide_threshold']
                and g['descriptive_95_CI'][0] > 0)
    result['dev_selected_guided'] = bool(selected)
    if selected:
        rows_c = run_cohort('confirmation', protocol['confirmation_instance_ids'],
                            protocol['dataset_seed_confirmation'],
                            protocol['anytime_seed_confirmation'], protocol, contract,
                            providers)
        with (out / 'confirmation_progress.jsonl').open('x', encoding='utf-8') as f:
            for row in rows_c:
                f.write(json.dumps(row, allow_nan=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        result['confirmation'] = summarise(rows_c, 'confirmation', protocol)
        result['confirmation_run'] = True
    else:
        result['confirmation'] = None
        result['confirmation_run'] = False
        result['outcome'] = 'development_stage_found_no_guided_destroy_gain'
    result['online_gain_established'] = False
    result['maskco_attributable_gain_established'] = False
    result['stable_gain_established'] = bool(selected) and result.get('confirmation') is not None
    result['wall_s'] = time.perf_counter() - started
    dump_new(out / 'result.json', result)
    print(json.dumps({
        'dev_means': {a: round(t['mean_final_cost'], 4)
                      for a, t in result['dev']['table'].items()},
        'dev_cycles': {a: result['dev']['table'][a]['total_cycles']
                       for a in result['dev']['table']},
        'dev_decode_calls': {a: result['dev']['table'][a]['total_decode_calls']
                             for a in result['dev']['table']},
        'dev_verdicts': result['dev']['verdicts'],
        'dev_selected_guided': bool(selected),
        'confirmation_run': result['confirmation_run'],
        'wall_s': round(result['wall_s'], 1)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--recorded-identity', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-instances', type=int, default=None)
    main(parser.parse_args())
