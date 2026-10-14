"""D-MASKFRAC-v1: is 50 percent actually the best mask fraction, or just the best tested?

Origin. The confirmed delivery base uses a 50 percent mask. It was chosen because
D-ANYTIME-v1 compared exactly two fractions, 0.3 and 0.5, and 0.5 won. No curve was ever
measured, so the confirmed value is the better of two points rather than an optimum, and
the shape of the response is unknown. This matters because the mask fraction is the single
structural parameter of the confirmed mechanism: if the response is flat then the choice
is arbitrary and nothing is lost, and if there is a better value then the base itself can
be improved without any learned component.

Arms: the same deadline-driven reconstruction loop under the same 10.0s wall budget and
the same initial solution, with uniform random masks at fractions 0.2, 0.35, 0.4, 0.5,
0.6 and 0.7. Every arm does zero neural decoding. The 0.5 arm is the confirmed base and
this run's own baseline, reported alongside as required since D-MASKHEUR-v1 established
that anytime values do not reproduce across runs.

Registered reading: a fraction counts as an improvement only if it beats 0.5 with mean
relative gain >= 1 percent and paired 95 percent CI lower > 0 on dev. If any does, that
fraction goes to a fresh-seed confirmation cohort; if none does, the fraction channel is
closed and 50 percent stands as measured rather than assumed.

Scope: non-learning, pre-dispatch snapshot, simulated parent physics. Improving the base
is not a MaskCO-attributable gain.

Usage:
    python delivery_mask_fraction.py --protocol P.json --out NEWDIR [--max-instances N]
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
for sub in ('delivery', 'coldchain', 'research_20261010'):
    sys.path.insert(0, str(SCRIPTS / sub))

from delivery_anytime import anytime_reconstruct
from delivery_contract import DeliveryContract
from delivery_evaluator import evaluate
from delivery_equal_schedule import dump_new, generate

PROTOCOL = {
    'version': 'D-MASKFRAC-v1',
    'kind': 'mask_fraction_response_curve_not_gain_claim',
    'parent': 'D-ANYTIME-v1',
    'base_fraction': 0.5,
    'fractions': [0.2, 0.35, 0.4, 0.5, 0.6, 0.7],
    'dataset_seed_dev': 2026101002,
    'dataset_seed_confirmation': 2026101013,
    'dev_instance_ids': list(range(32)),
    'confirmation_instance_ids': list(range(1000, 1032)),
    'anytime_seed_dev': 2026101131,
    'anytime_seed_confirmation': 2026101176,
    'time_limit_s': 10.0,
    'bootstrap_seed': 2026101261,
    'bootstrap_repetitions': 2000,
    'signal_threshold': 0.01,
    'selection_rule': ('carry a fraction into confirmation only if it beats 0.5 on dev with '
                       'mean>=1 percent and paired 95 percent CI lower>0'),
    'claims_online_gain': False,
    'all_failures_retained': True,
    'physics_note': 'simulated parent-physics snapshot; not physical calibration',
}


def verify_protocol(p):
    expected = {'version': 'D-MASKFRAC-v1', 'base_fraction': .5,
                'fractions': [0.2, 0.35, 0.4, 0.5, 0.6, 0.7],
                'dataset_seed_dev': 2026101002, 'dataset_seed_confirmation': 2026101013,
                'anytime_seed_dev': 2026101131, 'anytime_seed_confirmation': 2026101176,
                'time_limit_s': 10., 'bootstrap_seed': 2026101261,
                'bootstrap_repetitions': 2000, 'signal_threshold': .01}
    for k, v in expected.items():
        got = p.get(k)
        if isinstance(v, (bool, list)):
            if got != v:
                raise ValueError('D-MASKFRAC-v1 registered field changed: ' + k)
        elif got != v or isinstance(got, bool):
            raise ValueError('D-MASKFRAC-v1 registered field changed: ' + k)
    if p['base_fraction'] not in p['fractions']:
        raise ValueError('The confirmed base fraction must be one of the arms')
    if p.get('dev_instance_ids') != list(range(32)):
        raise ValueError('Dev cohort ids are registered and fixed')
    if p.get('confirmation_instance_ids') != list(range(1000, 1032)):
        raise ValueError('Confirmation cohort ids are registered and fixed')
    if p.get('claims_online_gain') is not False:
        raise ValueError('Scope cannot become an online gain claim')
    return expected


def run_cohort(ids, dataset_seed, anytime_seed, protocol, contract, fractions):
    rows = []
    for instance_id in ids:
        data, initial = generate(dataset_seed, instance_id)
        checked = evaluate(data, initial, contract)
        if not checked['hard']:
            raise RuntimeError('Initial data hard failure retained: %d' % instance_id)
        out = {}
        for frac in fractions:
            res = anytime_reconstruct(data, initial, contract, protocol['time_limit_s'],
                                      float(frac), anytime_seed)
            replay = evaluate(data, res['routes'], contract)
            if not replay['hard'] or abs(replay['cost'] - res['final_endpoint_cost']) > 1e-9:
                raise RuntimeError('Fraction %s replay mismatch: %d' % (frac, instance_id))
            out[str(frac)] = {'final_cost': res['final_endpoint_cost'],
                              'wall_s': res['wall_s'], 'cycles': res['cycles'],
                              'accepted': res['accepted'], 'decode_calls': 0}
        rows.append({'instance_id': int(instance_id), 'arms': out})
        print(json.dumps({'instance_id': instance_id, 'completed': len(rows),
                          **{f: round(out[f]['final_cost'], 4) for f in out}}), flush=True)
    return rows


def paired(control, other, seed, reps):
    shared = sorted(set(control) & set(other))
    diffs = np.array([(control[i] - other[i]) / control[i] for i in shared], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(reps, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return {'n_instances': len(shared), 'mean_relative_gain': float(diffs.mean()),
            'descriptive_95_CI': [float(np.percentile(means, 2.5)),
                                  float(np.percentile(means, 97.5))],
            'instances_other_better': int(sum(d > 0 for d in diffs)),
            'instances_other_worse': int(sum(d < 0 for d in diffs))}


def summarise(rows, fractions, protocol):
    table = {}
    for f in fractions:
        table[f] = {'n': len(rows),
                    'mean_final_cost': statistics.fmean(r['arms'][f]['final_cost']
                                                        for r in rows),
                    'mean_wall_s': statistics.fmean(r['arms'][f]['wall_s'] for r in rows),
                    'total_cycles': sum(r['arms'][f]['cycles'] for r in rows),
                    'total_accepted': sum(r['arms'][f]['accepted'] for r in rows),
                    'total_decode_calls': sum(r['arms'][f]['decode_calls'] for r in rows)}
    base = str(protocol['base_fraction'])
    by = {f: {r['instance_id']: r['arms'][f]['final_cost'] for r in rows} for f in fractions}
    gains = {f: paired(by[base], by[f], protocol['bootstrap_seed'],
                       protocol['bootstrap_repetitions'])
             for f in fractions if f != base}
    return {'table': table, 'gains_vs_base': gains}


def main(args):
    started = time.perf_counter()
    protocol = json.loads(Path(args.protocol).read_text(encoding='utf-8'))
    verify_protocol(protocol)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    dump_new(out / 'PROTOCOL.json', protocol)
    contract = DeliveryContract()
    fractions = [str(f) for f in protocol['fractions']]

    ids = protocol['dev_instance_ids']
    if args.max_instances is not None:
        ids = ids[:args.max_instances]
    dev_rows = run_cohort(ids, protocol['dataset_seed_dev'], protocol['anytime_seed_dev'],
                          protocol, contract, fractions)
    with (out / 'dev_progress.jsonl').open('x', encoding='utf-8') as f:
        for row in dev_rows:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    dev = summarise(dev_rows, fractions, protocol)
    thr = protocol['signal_threshold']
    carried = [f for f, g in dev['gains_vs_base'].items()
               if g['mean_relative_gain'] >= thr and g['descriptive_95_CI'][0] > 0]
    result = {'protocol': protocol, 'dev': dev, 'fractions_with_signal': sorted(carried),
              'fraction_channel_open': bool(carried)}
    if carried:
        confirm = [str(protocol['base_fraction'])] + sorted(carried)
        conf_rows = run_cohort(protocol['confirmation_instance_ids'],
                               protocol['dataset_seed_confirmation'],
                               protocol['anytime_seed_confirmation'], protocol, contract,
                               confirm)
        with (out / 'confirmation_progress.jsonl').open('x', encoding='utf-8') as f:
            for row in conf_rows:
                f.write(json.dumps(row, allow_nan=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        result['confirmation'] = summarise(conf_rows, confirm, protocol)
        result['confirmation_run'] = True
    else:
        result['confirmation'] = None
        result['confirmation_run'] = False
        result['outcome'] = 'no_fraction_beats_the_confirmed_half'
    result['online_gain_established'] = False
    result['maskco_attributable_gain_established'] = False
    result['stable_gain_established'] = False
    result['wall_s'] = time.perf_counter() - started
    dump_new(out / 'result.json', result)
    curves = {f: dev['table'][f]['mean_final_cost'] for f in fractions}
    print(json.dumps({'curve': {k: round(v, 4) for k, v in curves.items()},
                      'gains_vs_half': {f: [round(g['mean_relative_gain'], 4),
                                            [round(c, 4) for c in g['descriptive_95_CI']]]
                                        for f, g in dev['gains_vs_base'].items()},
                      'fractions_with_signal': sorted(carried),
                      'outcome': result.get('outcome', 'carried_to_confirmation'),
                      'wall_s': round(result['wall_s'], 1)}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-instances', type=int, default=None)
    main(ap.parse_args())
