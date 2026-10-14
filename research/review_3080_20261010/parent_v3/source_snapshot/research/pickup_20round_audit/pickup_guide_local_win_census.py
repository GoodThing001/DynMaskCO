"""Stage A gate component: the guide term's LOCAL-WIN census, read off archived pilot results.

WHY: handover rule 6 - "a learned term that never wins locally cannot win globally; before tuning a
learned term, count how often it beats the base rule at an IDENTICAL state". The delivery line ran
exactly this census and it was decisive (0 of 400 decisions ever picked a strictly cheaper
insertion). The pickup pilots already contain the paired data needed to run the same census, so
this needs NO new computation, NO new data and no new seeds - it only re-reads archived rows.

WHAT IT COUNTS, per pilot artifact:
  * maskco vs recon_only, paired on (inst, clock, mask_seed): the guide term is present in maskco
    and absent in recon_only, so a strictly-better maskco row means the guide won at an identical
    state; strictly-worse means it lost; equal means it changed nothing or tied.
  * recon_only vs discard, same pairing: a NULL CONTROL. The arm's own qualification asserted these
    two produce byte-identical permutations, so every pair here must tie; any non-tie is a defect.
  * every arm against DOING NOTHING (handover rule 4): a positive raw_delta_utility means installing
    the reconstruction improved the whole-day A-v1 utility over the baseline trajectory.

SCOPE: read-only diagnostic over already-archived exploratory pilots. Descriptive only, no alpha.
It cannot establish or refute a stable online gain, MaskCO attribution, impossibility or
equivalence, and a mean above zero is far below the >= 20 utility/day main gate.
"""
import argparse
import json
import os
from pathlib import Path

EPS = 1e-10
DEFAULT_DIRS = [
    'pickup_arm_pilot_v1',
    'pickup_arm_pilot_n45_v1',
    'pickup_arm_pilot_mf50_v1',
    'pickup_arm_pilot_mf70_v1',
    'pickup_arm_pilot_mf80_v1',
    'pickup_arm_pilot_mf90_v1',
    'pickup_arm_pilot_mf100_v1',
]


def load_rows(path):
    with open(path, encoding='utf-8') as fh:
        payload = json.load(fh)
    return payload.get('rows') or [], payload


def key_of(row):
    return (row.get('inst'), row.get('clock'), row.get('mask_seed'))


def pair_census(rows, mode_a, mode_b):
    a, b = {}, {}
    for r in rows:
        if r.get('raw_delta_utility') is None:
            continue
        if r.get('mode') == mode_a:
            a[key_of(r)] = r
        elif r.get('mode') == mode_b:
            b[key_of(r)] = r
    keys = sorted(set(a) & set(b), key=lambda k: (str(k[0]), float(k[1]), int(k[2])))
    better = worse = tied = 0
    deltas = []
    for k in keys:
        d = float(a[k]['raw_delta_utility']) - float(b[k]['raw_delta_utility'])
        deltas.append(d)
        if d > EPS:
            better += 1
        elif d < -EPS:
            worse += 1
        else:
            tied += 1
    return {'mode_a': mode_a, 'mode_b': mode_b, 'n_pairs': len(keys),
            'a_better': better, 'a_worse': worse, 'tied': tied,
            'mean_delta_a_minus_b': (sum(deltas) / len(deltas)) if deltas else None,
            'permutation_identical_pairs': sum(
                1 for k in keys if list(a[k].get('permutation') or []) ==
                list(b[k].get('permutation') or []))}


def do_nothing_census(rows):
    out = {}
    for mode in sorted({r.get('mode') for r in rows if r.get('mode')}):
        vals = [float(r['raw_delta_utility']) for r in rows
                if r.get('mode') == mode and r.get('raw_delta_utility') is not None]
        if not vals:
            continue
        out[mode] = {'n': len(vals),
                     'n_better_than_doing_nothing': sum(1 for v in vals if v > EPS),
                     'n_equal_to_doing_nothing': sum(1 for v in vals if abs(v) <= EPS),
                     'n_worse_than_doing_nothing': sum(1 for v in vals if v < -EPS),
                     'mean_raw_delta_utility': sum(vals) / len(vals)}
    return out


def main(args):
    base = Path(args.base)
    dirs = args.dirs or DEFAULT_DIRS
    report = {'kind': 'pickup_guide_local_win_census_readonly',
              'what_this_is': 'read-only re-reading of archived exploratory pilot rows',
              'eps': EPS, 'artifacts': {}}
    for name in dirs:
        p = base / name / 'result.json'
        if not p.exists():
            report['artifacts'][name] = {'status': 'MISSING'}
            continue
        rows, payload = load_rows(p)
        entry = {
            'status': 'ok',
            'protocol_version': payload.get('protocol_version'),
            'mask_fraction': payload.get('mask_fraction'),
            'n_rows': len(rows),
            'guide_local_win_census': pair_census(rows, 'maskco', 'recon_only'),
            'null_control_recon_only_vs_discard': pair_census(rows, 'recon_only', 'discard'),
            'versus_doing_nothing': do_nothing_census(rows),
        }
        report['artifacts'][name] = entry
        g = entry['guide_local_win_census']
        n = entry['null_control_recon_only_vs_discard']
        print('%-26s n=%-4d guide: better=%-3d worse=%-3d tied=%-3d | null-control non-ties=%d'
              % (name, g['n_pairs'], g['a_better'], g['a_worse'], g['tied'],
                 n['a_better'] + n['a_worse']))

    ok = [v for v in report['artifacts'].values() if v.get('status') == 'ok']
    total_better = sum(v['guide_local_win_census']['a_better'] for v in ok)
    total_pairs = sum(v['guide_local_win_census']['n_pairs'] for v in ok)
    null_bad = sum(v['null_control_recon_only_vs_discard']['a_better']
                   + v['null_control_recon_only_vs_discard']['a_worse'] for v in ok)
    report['aggregate'] = {
        'total_paired_observations': total_pairs,
        'guide_strictly_better': total_better,
        'guide_strictly_worse': sum(v['guide_local_win_census']['a_worse'] for v in ok),
        'guide_tied': sum(v['guide_local_win_census']['tied'] for v in ok),
        'null_control_non_ties': null_bad,
        'stage_a_stop_rule_triggered': bool(total_better == 0),
        'stage_a_stop_rule_text': ('zero guide local wins across all archived pilots => the guide '
                                   'term is a perturbation by construction; STOP and do not run the '
                                   'panel'),
    }
    report['explicit_non_claims'] = {
        'stable_online_gain': False, 'maskco_attribution': False,
        'pickup_impossibility': False, 'equivalence': False,
        'note': 'these pilots ran n=3 and n=4-5 subproblems only and no online iterative policy; '
                'a zero local-win count closes THIS guide at THESE configurations, it does not '
                'close learned reconstruction in general'}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'result.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print()
    print(json.dumps(report['aggregate'], indent=2, default=str))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--out', required=True)
    ap.add_argument('--dirs', nargs='*')
    main(ap.parse_args())
