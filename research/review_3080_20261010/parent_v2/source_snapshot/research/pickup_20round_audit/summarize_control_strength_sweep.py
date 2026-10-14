"""Summarize the pickup masked-reconstruction control-strength sweep.

Read-only over the pilot result.json files. Writes only into the sweep output root.

Purpose. The distance-only `recon_only` control of the pickup arm was nearly INERT at the
registered mask_fraction 0.5 (it reproduced the baseline template in 9/12 rows at |S|=3 and
23/28 at |S|=4..5), so the contrast `maskco - recon_only` could not be interpreted as evidence
about the guide. This script aggregates a mask-fraction sweep so control strength can be read
off directly per fraction.

Metrics per fraction (all descriptive; no alpha, no confirmatory claim):
  * identity rate           - installed sequence == baseline template, per mode.
  * effective perturbation  - installed sequence != template AND |delta utility| > 1e-10.
                              A control can be non-identical yet behaviourally blind, because the
                              env commits only the suffix's FIRST element at the decision point
                              and later replans may rebuild the vehicle's plan; both are reported.
  * lead-order change rate  - first template slot holds a different order. This is the cheap
                              sufficient perturbation (the first suffix element is committed
                              immediately), so it is the most direct read on behavioural reach.
  * per-mode mean raw delta utility, hard feasibility, guard firings, decoder calls, failures.
  * paired contrast maskco - recon_only with a descriptive percentile bootstrap CI.

Honesty. These pilots have no `cond_hist` arm and few engineered states, so they CANNOT address
the pickup program MAIN gate (>= 20 utility/day versus cond_hist with a paired CI lower bound
above 0, plus a positive mean versus the same-initial-value reconstruction-only control).
A CI containing zero means THE DIFFERENCE IS NOT ESTABLISHED; it is not evidence of no difference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def state_key(inst, orders):
    return (int(inst), tuple(sorted(int(o) for o in orders)))


def per_mode(rows, mode):
    sub = [r for r in rows if r['mode'] == mode]
    n = len(sub)
    ident = sum(1 for r in sub if r['is_identity_to_template'])
    lead_change = sum(1 for r in sub
                      if r.get('baseline_first') is not None
                      and int(r['permutation'][0]) != int(r['baseline_first']))
    lead_known = sum(1 for r in sub if r.get('baseline_first') is not None)
    effective = sum(1 for r in sub
                    if (not r['is_identity_to_template'])
                    and r['raw_delta_utility'] is not None
                    and abs(float(r['raw_delta_utility'])) > 1e-10)
    vals = [float(r['raw_delta_utility']) for r in sub if r['raw_delta_utility'] is not None]
    state_ids = sorted({(r['inst'], r['event_id'] if 'event_id' in r else r['clock'],
                         r['vehicle_id'] if 'vehicle_id' in r else -1) for r in sub})
    return {
        'n': n,
        'n_identity_to_template': int(ident),
        'identity_rate': (ident / n) if n else None,
        'n_different_from_template': int(n - ident),
        'n_different_but_zero_delta': int(sum(
            1 for r in sub if (not r['is_identity_to_template'])
            and r['raw_delta_utility'] is not None
            and abs(float(r['raw_delta_utility'])) <= 1e-10)),
        'n_identity_and_zero_delta': int(sum(
            1 for r in sub if r['is_identity_to_template']
            and r['raw_delta_utility'] is not None
            and abs(float(r['raw_delta_utility'])) <= 1e-10)),
        'n_different_and_delta_nonzero': int(sum(
            1 for r in sub if (not r['is_identity_to_template'])
            and r['raw_delta_utility'] is not None
            and abs(float(r['raw_delta_utility'])) > 1e-10)),
        'n_effective_perturbation': int(effective),
        'effective_perturbation_rate': (effective / n) if n else None,
        'n_lead_order_changed': int(lead_change),
        'n_lead_order_known': int(lead_known),
        'lead_order_change_rate': (lead_change / lead_known) if lead_known else None,
        'n_utility_unchanged': int(sum(1 for r in sub if r['raw_delta_utility'] is not None
                                       and abs(float(r['raw_delta_utility'])) <= 1e-10)),
        'n_failures': int(sum(1 for r in sub if r['raw_delta_utility'] is None)),
        'n_hard_feasible': int(sum(1 for r in sub if r['hard_feasible'])),
        'n_mask_extended': int(sum(1 for r in sub if r['mask_extended'])),
        'n_rows_with_mask_extended': int(sum(1 for r in sub if r['mask_extended'])),
        'n_raw_positive': int(sum(1 for v in vals if v > 1e-10)),
        'n_raw_zero': int(sum(1 for v in vals if abs(v) <= 1e-10)),
        'n_raw_negative': int(sum(1 for v in vals if v < -1e-10)),
        'mean_raw_delta_utility': float(np.mean(vals)) if vals else None,
        'min_raw_delta_utility': float(np.min(vals)) if vals else None,
        'max_raw_delta_utility': float(np.max(vals)) if vals else None,
        'decoder_calls_total': int(sum(r['decoder_calls'] for r in sub)),
    }


def n_states_with_guard(rows, mode):
    sub = [r for r in rows if r['mode'] == mode]
    keys = sorted({state_key(r['inst'], r['permutation']) for r in sub})
    fired = sorted({state_key(r['inst'], r['permutation']) for r in sub if r['mask_extended']})
    return {'n_states': len(keys), 'n_states_where_guard_fired': len(fired),
            'n_rows_with_guard_fired': int(sum(1 for r in sub if r['mask_extended'])),
            'guard_fired_state_keys': [list(k) for k in fired]}


def paired(rows, a_mode, b_mode, bootstrap_seed, reps):
    a = {(r['inst'], r['mask_seed']): r for r in rows if r['mode'] == a_mode}
    b = {(r['inst'], r['mask_seed']): r for r in rows if r['mode'] == b_mode}
    d = [a[k]['raw_delta_utility'] - b[k]['raw_delta_utility']
         for k in sorted(set(a) & set(b))
         if a[k]['raw_delta_utility'] is not None and b[k]['raw_delta_utility'] is not None]
    if not d:
        return None
    d = np.asarray(d, dtype=np.float64)
    ci = None
    if len(d) >= 2:
        rng = np.random.default_rng(bootstrap_seed)
        draws = d[rng.integers(0, len(d), size=(reps, len(d)))].mean(axis=1)
        ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    return {'n_pairs': int(len(d)), 'mean': float(d.mean()), 'ci95': ci,
            'n_positive': int((d > 1e-10).sum()), 'n_zero': int((abs(d) <= 1e-10).sum()),
            'n_negative': int((d < -1e-10).sum()),
            'ci_includes_zero': (bool(ci[0] <= 0.0 <= ci[1]) if ci else None)}


def per_state_rows(rows):
    """Per-state (inst, clock, vehicle) per-mode mean delta, so sign reversals are visible."""
    keys = []
    for r in rows:
        k = (r['inst'], round(float(r['clock']), 6), r.get('vehicle_id', -1))
        if k not in keys:
            keys.append(k)
    out = []
    for k in sorted(keys):
        rec = {'inst': k[0], 'clock': k[1], 'vehicle_id': k[2],
               'n_orders': len(next(r for r in rows if r['inst'] == k[0]
                                    and round(float(r['clock']), 6) == k[1])['permutation'])}
        for m in ('maskco', 'recon_only'):
            sub = [r for r in rows if r['mode'] == m and r['inst'] == k[0]
                   and round(float(r['clock']), 6) == k[1]]
            vals = [float(r['raw_delta_utility']) for r in sub
                    if r['raw_delta_utility'] is not None]
            rec[m] = {'mean': float(np.mean(vals)) if vals else None,
                      'n_identity': int(sum(1 for r in sub if r['is_identity_to_template'])),
                      'n': len(sub),
                      'deltas': vals}
        ma = [r for r in rows if r['mode'] == 'maskco' and r['inst'] == k[0]
              and round(float(r['clock']), 6) == k[1]]
        rb = [r for r in rows if r['mode'] == 'recon_only' and r['inst'] == k[0]
              and round(float(r['clock']), 6) == k[1]]
        bys = {}
        for r in rb:
            bys[r['mask_seed']] = r
        dd = []
        for r in ma:
            o = bys.get(r['mask_seed'])
            if o is not None and r['raw_delta_utility'] is not None \
                    and o['raw_delta_utility'] is not None:
                dd.append(float(r['raw_delta_utility']) - float(o['raw_delta_utility']))
        rec['maskco_minus_recon_only'] = {
            'mean': float(np.mean(dd)) if dd else None,
            'n_positive': int(sum(1 for v in dd if v > 1e-10)),
            'n_zero': int(sum(1 for v in dd if abs(v) <= 1e-10)),
            'n_negative': int(sum(1 for v in dd if v < -1e-10)),
            'deltas': dd}
        out.append(rec)
    return out


def main(args):
    specs = []
    for s in args.spec:
        label, frac, path = s.split('=', 2)
        specs.append({'label': label, 'mask_fraction': float(frac), 'path': str(Path(path))})

    entries = []
    for sp in specs:
        data = json.loads(Path(sp['path']).read_text(encoding='utf-8'))
        rows = data['rows']
        # map order-set -> the template's leading order, so lead-order change can be measured
        first_by_key = {}
        for st in data['states']:
            if st['slots']:
                first_by_key[state_key(st['inst'], st['orders'])] = int(st['template'][st['slots'][0]])
        for r in rows:
            r['baseline_first'] = first_by_key.get(state_key(r['inst'], r['permutation']))
        modes = {m: per_mode(rows, m) for m in ('maskco', 'recon_only', 'discard')}
        guards = {m: n_states_with_guard(rows, m) for m in ('maskco', 'recon_only', 'discard')}
        contrast = paired(rows, 'maskco', 'recon_only', args.bootstrap_seed,
                          args.bootstrap_repetitions)
        contrast_rd = paired(rows, 'recon_only', 'discard', args.bootstrap_seed,
                             args.bootstrap_repetitions)
        ident = modes['recon_only']['identity_rate']
        entries.append({
            'label': sp['label'],
            'mask_fraction': sp['mask_fraction'],
            'result_json': sp['path'],
            'result_json_sha256': hashlib.sha256(Path(sp['path']).read_bytes()).hexdigest(),
            'n_states': data['n_states'], 'n_rows': data['n_rows'],
            'state_order_set_sizes': sorted({len(st['orders']) for st in data['states']}),
            'per_mode': modes,
            'guard_activations': guards,
            'per_state': per_state_rows(rows),
            'paired_contrast_maskco_minus_recon_only': contrast,
            'paired_contrast_recon_only_minus_discard': contrast_rd,
            'control_inert_by_identity_rate_gt_0p5': (ident is not None and ident > 0.5),
        })

    payload = {
        'kind': 'pickup_control_strength_mask_fraction_sweep_summary',
        'bootstrap_seed': int(args.bootstrap_seed),
        'bootstrap_repetitions': int(args.bootstrap_repetitions),
        'definition_of_inert': ('a control is called INERT when its identity rate (installed '
                                'sequence == baseline template) is ABOVE about 0.5, i.e. it '
                                'reproduces the baseline ordering in the majority of rows and so '
                                'cannot exhibit a difference either way'),
        'distance_only_identity_metric': ('per_mode.recon_only.identity_rate; `discard` is reported '
                                          'alongside because it must be behaviourally identical to '
                                          'recon_only (it pays the same decoder call)'),
        'interval_status': ('every interval here is DESCRIPTIVE ONLY, carries no alpha, and is not '
                            'confirmatory'),
        'ci_includes_zero_reading': ('a CI including zero means THE DIFFERENCE IS NOT ESTABLISHED; '
                                     'it is not evidence of no difference'),
        'program_main_gate_restated': ('the pickup MAIN gate requires >= 20 utility/day versus '
                                       'cond_hist with a paired CI lower bound > 0, plus a '
                                       'positive mean versus the same-initial-value '
                                       'reconstruction-only control. This sweep has NO cond_hist '
                                       'arm and few engineered states, so it CANNOT address the '
                                       'main gate.'),
        'entries': entries,
        'stable_online_gain_established': False, 'maskco_attributable': False,
        'pickup_impossibility_proven': False, 'equivalence_established': False,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'control_strength_sweep_summary.json').write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str), encoding='utf-8')
    compact = []
    for e in entries:
        compact.append({
            'label': e['label'], 'mask_fraction': e['mask_fraction'],
            'n_states': e['n_states'], 'n_rows': e['n_rows'],
            'recon_only_identity_rate': e['per_mode']['recon_only']['identity_rate'],
            'discard_identity_rate': e['per_mode']['discard']['identity_rate'],
            'maskco_identity_rate': e['per_mode']['maskco']['identity_rate'],
            'recon_only_lead_change_rate': e['per_mode']['recon_only']['lead_order_change_rate'],
            'recon_only_effective_perturbation_rate':
                e['per_mode']['recon_only']['effective_perturbation_rate'],
            'contrast': e['paired_contrast_maskco_minus_recon_only'],
            'recon_only_mean': e['per_mode']['recon_only']['mean_raw_delta_utility'],
            'maskco_mean': e['per_mode']['maskco']['mean_raw_delta_utility'],
            'inert': e['control_inert_by_identity_rate_gt_0p5'],
            'guard_states_recon_only': e['guard_activations']['recon_only'][
                'n_states_where_guard_fired'],
        })
    print(json.dumps(compact, indent=2, default=str))
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--spec', action='append', required=True,
                    help='label=mask_fraction=path/to/result.json (repeatable)')
    ap.add_argument('--bootstrap-seed', type=int, default=2026120212)
    ap.add_argument('--bootstrap-repetitions', type=int, default=2000)
    sys.exit(main(ap.parse_args()))
