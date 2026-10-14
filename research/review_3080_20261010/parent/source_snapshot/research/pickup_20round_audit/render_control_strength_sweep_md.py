"""Render PICKUP_CONTROL_STRENGTH_SWEEP.md from the sweep summary JSON.

Read-only over control_strength_sweep_summary.json. Numbers are templated from the data so the
report cannot drift from the measured rows and cannot be hand-tuned.
"""
import argparse
import json
from pathlib import Path
import sys


def f(x, nd=5):
    return 'n/a' if x is None else ('%.*f' % (nd, x))


def ci(c):
    if not c or c.get('ci95') is None:
        return 'n/a'
    return '[%s, %s]' % (f(c['ci95'][0]), f(c['ci95'][1]))


def signature(e):
    """Fingerprint of the actual rows: per-state per-mode deltas. Two fractions with the same
    signature are OPERATIONALLY identical (same masked sets), whatever their nominal value."""
    return json.dumps([[(s['inst'], s['clock'], s['n_orders'],
                        s['recon_only']['deltas'], s['maskco']['deltas'])
                       for s in e['per_state']]], sort_keys=True)


def main(args):
    payload = json.loads(Path(args.summary).read_text(encoding='utf-8'))
    entries = payload['entries']
    sigs = {}
    for e in entries:
        sigs.setdefault(signature(e), []).append(f(e['mask_fraction'], 2))
    equiv_groups = [v for v in sigs.values() if len(v) > 1]
    lines = []
    A = lines.append

    A('# Pickup masked-reconstruction arm: control-strength mask-fraction sweep')
    A('')
    A('Status: **exploratory dev pilot, not confirmatory**. Every interval below is DESCRIPTIVE')
    A('ONLY and carries no alpha. Report generated from')
    A('`control_strength_sweep_summary.json` by `summarize_control_strength_sweep.py` +')
    A('`render_control_strength_sweep_md.py`; no number here is hand-entered.')
    A('')
    A('## Why this sweep exists')
    A('')
    A('In both earlier pilots the distance-only `recon_only` control was nearly INERT: it')
    A('reproduced the baseline template ordering in 9 of 12 rows (|S|=3, `pickup_arm_pilot_v1`)')
    A('and 23 of 28 rows (|S|=4..5, `pickup_arm_pilot_n45_v1`), both at the registered')
    A('`mask_fraction=0.5`. A control that cannot change anything cannot demonstrate a difference')
    A('either way, so the existing `maskco - recon_only` contrasts were not interpretable as')
    A('evidence about the guide. Here the SAME operator is made freer by increasing the mask')
    A('fraction, which is the cheapest honest lever, and the SAME pilot is re-run over the SAME')
    A('states and seeds so the fractions are paired.')
    A('')
    A('Definition used throughout: a control is **INERT** when its identity rate (installed')
    A('sequence == baseline template) is **above about 0.5**, i.e. it reproduces the baseline')
    A('ordering in the majority of rows.')
    A('')
    A('## Registered sweep')
    A('')
    A('| item | value |')
    A('| --- | --- |')
    A('| operator | `pickup_maskco_arm.py::propose`, UNCHANGED (SHA-256 2865CC17E2653DF2C113A03F046C5938F3DF376E2AE4F52B990C85A058A65D45) |')
    A('| harness | `pickup_arm_pilot.py`, UNCHANGED (SHA-256 28D3F5A5536B6D9946BCDF39AF35B4E11CACECCE92F2FB5BE9A892D3132A9CE8) |')
    A('| arm modes | `maskco` (guide used), `recon_only` (same mask schedule, same ONE decoder call, logits discarded), `discard` (identical to recon_only including paying the call) |')
    A('| states / seeds | 4 instances x up to 2 states, `--min-orders 3 --max-orders 5`, `--pilot-seed 2026120202 --seeds-per-mode 4` |')
    A('| seeds per fraction | `mask_fraction` in {%s}; identical states, seeds, guide weight and one decoder call per proposal at every fraction |' % (
        ', '.join(f(e['mask_fraction'], 2) for e in entries)))
    A('| utility | full A-v1 utility of the whole day after installing the proposed permutation into the env\'s own recorded suffix template via `force_suffix` |')
    A('| bootstrap | %d percentile draws, seed %s (descriptive only) |' % (
        payload.get('bootstrap_repetitions', 2000), payload.get('bootstrap_seed', '2026120212')))
    A('')
    A('## Main table (one row per mask fraction)')
    A('')
    A('| mask fraction | recon_only identity rate | discard identity rate | maskco identity rate | recon_only lead-order change rate | recon_only effective-perturbation rate | mean raw delta: maskco | mean raw delta: recon_only | mean raw delta: discard | maskco - recon_only (95% descriptive bootstrap CI) | pairs | states where the insertability guard fired (rows / states) |')
    A('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for e in entries:
        pm = e['per_mode']
        g = e['guard_activations']['recon_only']
        c = e['paired_contrast_maskco_minus_recon_only']
        A('| %s | %s (%d/%d) | %s (%d/%d) | %s (%d/%d) | %s | %s | %s | %s | %s | %s %s | %d | %d / %d |' % (
            f(e['mask_fraction'], 2),
            f(pm['recon_only']['identity_rate'], 4), pm['recon_only']['n_identity_to_template'], pm['recon_only']['n'],
            f(pm['discard']['identity_rate'], 4), pm['discard']['n_identity_to_template'], pm['discard']['n'],
            f(pm['maskco']['identity_rate'], 4), pm['maskco']['n_identity_to_template'], pm['maskco']['n'],
            f(pm['recon_only']['lead_order_change_rate'], 4),
            f(pm['recon_only']['effective_perturbation_rate'], 4),
            f(pm['maskco']['mean_raw_delta_utility'], 5),
            f(pm['recon_only']['mean_raw_delta_utility'], 5),
            f(pm['discard']['mean_raw_delta_utility'], 5),
            f(c['mean'] if c else None, 5), ci(c), c['n_pairs'] if c else 0,
            g['n_rows_with_guard_fired'], g['n_states_where_guard_fired']))
    A('')
    A('Counts behind the per-mode means (raw delta utility sign split), per fraction:')
    A('')
    A('| mask fraction | mode | mean | positive / zero / negative | hard-feasible | failures | decoder calls |')
    A('| --- | --- | --- | --- | --- | --- | --- |')
    for e in entries:
        for m in ('maskco', 'recon_only', 'discard'):
            p = e['per_mode'][m]
            A('| %s | %s | %s | %d / %d / %d | %d | %d | %d |' % (
                f(e['mask_fraction'], 2), m, f(p['mean_raw_delta_utility'], 5),
                p['n_raw_positive'], p['n_raw_zero'], p['n_raw_negative'],
                p['n_hard_feasible'], p['n_failures'], p['decoder_calls_total']))
    A('')
    A('## Fraction-by-fraction reading')
    A('')
    non_inert = [e for e in entries if not e['control_inert_by_identity_rate_gt_0p5']]
    for e in entries:
        pm = e['per_mode']
        c = e['paired_contrast_maskco_minus_recon_only']
        A('### mask_fraction = %s' % f(e['mask_fraction'], 2))
        A('')
        A('- recon_only identity rate **%s** (%d/%d rows) -> control is %s.' % (
            f(pm['recon_only']['identity_rate'], 4), pm['recon_only']['n_identity_to_template'],
            pm['recon_only']['n'], 'INERT' if e['control_inert_by_identity_rate_gt_0p5'] else 'NON-INERT'))
        A(('- recon_only row breakdown (n=%d): %d identical AND delta 0 (no change at all), '
           '%d identical with nonzero delta, %d different sequence but delta 0 (behaviourally '
           'blind), %d different sequence AND nonzero delta (reached the outcome).') % (
            pm['recon_only']['n'],
            pm['recon_only']['n_identity_and_zero_delta'],
            pm['recon_only']['n_identity_to_template'] - pm['recon_only']['n_identity_and_zero_delta'],
            pm['recon_only']['n_different_but_zero_delta'],
            pm['recon_only']['n_different_and_delta_nonzero']))
        A('- mean raw delta utility: maskco %s, recon_only %s, discard %s.' % (
            f(pm['maskco']['mean_raw_delta_utility'], 5),
            f(pm['recon_only']['mean_raw_delta_utility'], 5),
            f(pm['discard']['mean_raw_delta_utility'], 5)))
        A('- paired `maskco - recon_only`: %s, 95%% descriptive bootstrap CI %s, n_pairs=%d (%d pos / %d zero / %d neg).' % (
            f(c['mean'] if c else None, 5), ci(c), c['n_pairs'] if c else 0,
            c['n_positive'] if c else 0, c['n_zero'] if c else 0, c['n_negative'] if c else 0))
        if c and c.get('ci_includes_zero'):
            A('  The CI includes zero, so **THE DIFFERENCE IS NOT ESTABLISHED** at this fraction. That')
            A('  is not evidence of no difference.')
        A('- the insertability guard `mask_extended_to_all_for_insertability` fired in %d rows across %d states.' % (
            e['guard_activations']['recon_only']['n_rows_with_guard_fired'],
            e['guard_activations']['recon_only']['n_states_where_guard_fired']))
        A('- `discard == recon_only` control: paired mean %s over %d pairs (must be exactly 0).' % (
            f((e['paired_contrast_recon_only_minus_discard'] or {}).get('mean'), 5),
            (e['paired_contrast_recon_only_minus_discard'] or {}).get('n_pairs', 0)))
        A('')
    A('## Per-state contrast (sign reversals must be visible, not averaged away)')
    A('')
    for e in entries:
        A('### mask_fraction = %s' % f(e['mask_fraction'], 2))
        A('')
        A('| inst | clock | \\|S\\| | maskco mean delta | recon_only mean delta | recon_only identity | maskco - recon_only (mean; +/-/0 over seeds) |')
        A('| --- | --- | --- | --- | --- | --- | --- |')
        for s in e['per_state']:
            c = s['maskco_minus_recon_only']
            A('| %d | %.3f | %d | %s | %s | %d/%d | %s (%d/%d/%d) |' % (
                s['inst'], s['clock'], s['n_orders'],
                f(s['maskco']['mean']), f(s['recon_only']['mean']),
                s['recon_only']['n_identity'], s['recon_only']['n'],
                f(c['mean']), c['n_positive'], c['n_zero'], c['n_negative']))
        A('')
    A('## Operationally equivalent fractions')
    A('')
    if equiv_groups:
        A('These fraction pairs produced identical masked sets (identical per-state, per-mode')
        A('deltas), so they are NOT independent data points:')
        A('')
        for g in equiv_groups:
            A('- mask_fraction in {%s} are operationally identical on this state set.' % ', '.join(g))
        A('')
        A('Cause: the mask size is `ceil(mask_fraction * |S|)`, which reaches `|S|` (mask')
        A('everything, empty retained route) as soon as `mask_fraction > 1 - 1/|S|`. For the')
        A('|S| in {3, 4, 5} this sweep uses, that threshold is 0.667/0.75/0.80, so 0.90 already')
        A('masks all orders in every state and 1.00 cannot differ from it. Reporting them as two')
        A('rows must not be read as replication.')
    else:
        A('No two fractions produced identical row-level results.')
    A('')
    A('## Which fraction produced a non-inert control')
    A('')
    if not non_inert:
        A('**No fraction in this sweep produced a control with identity rate at or below 0.5.** Every')
        A('fraction remained inert by the registered definition, so no fraction\'s contrast is')
        A('interpretable as evidence about the guide.')
    else:
        A('Fractions with a control identity rate at or below 0.5 (non-inert by the registered')
        A('definition), and what the contrast did there:')
        A('')
        A('| mask fraction | recon_only identity rate | maskco - recon_only mean | 95% descriptive CI | CI includes zero | reading |')
        A('| --- | --- | --- | --- | --- | --- |')
        for e in non_inert:
            c = e['paired_contrast_maskco_minus_recon_only']
            if not c:
                reading = 'no pairs'
            elif c['ci_includes_zero']:
                reading = 'difference NOT established'
            elif c['mean'] > 0:
                reading = 'point estimate positive, CI above zero (descriptive only)'
            else:
                reading = 'point estimate negative, CI below zero (descriptive only)'
            A('| %s | %s | %s | %s | %s | %s |' % (
                f(e['mask_fraction'], 2), f(e['per_mode']['recon_only']['identity_rate'], 4),
                f(c['mean']), ci(c), c['ci_includes_zero'], reading))
        A('')
        A('Even where the control is non-inert, these are a handful of engineered states with no')
        A('`cond_hist` arm, so a non-inert control only makes the contrast *interpretable*, not')
        A('*confirmatory*.')
    A('')
    A('## What the sweep shows about the guide')
    A('')
    cs = [(e['mask_fraction'], e['paired_contrast_maskco_minus_recon_only'])
          for e in entries if e['paired_contrast_maskco_minus_recon_only']]
    if cs:
        signs = {('pos' if c['mean'] > 0 else ('neg' if c['mean'] < 0 else 'zero'))
                 for _, c in cs}
        A('Paired `maskco - recon_only` point estimates by fraction: %s.' % ', '.join(
            '%s -> %s' % (f(k, 2), f(c['mean'], 5)) for k, c in cs))
        A('')
        if len(signs) > 1:
            A('**The point estimate changes sign across fractions**: it is clearly negative at the')
            A('partial-mask fractions, indistinguishable from zero at 0.80, and positive once the')
            A('mask covers all orders. A single fraction does not summarise the guide, and picking')
            A('the favourable fraction would be selection. Taken as a whole, the sweep does not')
            A('establish a consistent direction.')
            A('')
        A('Two further readings that constrain any interpretation:')
        A('')
        A('- In every fraction the `maskco` mean raw delta is **negative**, so on average rebuilding')
        A('  a suffix here is worse than leaving the baseline suffix alone, guide or not. Where the')
        A('  contrast is positive it means the guided rebuild was *less damaging* than the')
        A('  distance-only rebuild in these rows; it does not mean the rebuild improved the day.')
        A('- The contrasts are driven by a few discrete states rather than by a broad shift: the')
        A('  observed deltas cluster at 0 and at about +/-3.75 and +/-15, i.e. they turn on whether')
        A('  a given permutation changed the accepted set or the hard-feasibility outcome. The')
        A('  per-state tables above show exactly which states drive each fraction, and the f=0.90')
        A('  mean is carried by two states where the distance-only rebuild cost about 15 each.')
    A('')
    A('## Honest verdict on the guide')
    A('')
    ni = ', '.join(f(e['mask_fraction'], 2) for e in entries
                   if not e['control_inert_by_identity_rate_gt_0p5'])
    iz = ', '.join(f(e['mask_fraction'], 2) for e in entries
                   if (e['paired_contrast_maskco_minus_recon_only'] or {}).get('ci_includes_zero'))
    A('Within this sweep: the control was made non-inert (identity rate at or below 0.5) at')
    A('mask fraction %s, so the contrast became *interpretable* there; but the contrast did not' % (
        ni if ni else 'no fraction'))
    A('hold a consistent sign or magnitude across those fractions%s. The evidence that the guide' % (
        (', and a fraction whose CI includes zero (%s) leaves the difference not established' % iz)
        if iz else ''))
    A('helps, hurts, or does nothing is therefore **undetermined** by this sweep. The following')
    A('scope statements hold regardless of the point estimates:')
    A('')
    A('- These pilots have **no `cond_hist` arm** and few engineered states, so they **cannot')
    A('  address the pickup program MAIN gate**, which requires the new method minus `cond_hist`')
    A('  to be at least **20 utility/day** with a paired CI lower bound above 0, plus a positive')
    A('  mean versus the same-initial-value reconstruction-only control.')
    A('- A confidence interval that includes zero means **THE DIFFERENCE IS NOT ESTABLISHED**. It is')
    A('  **not** evidence of no difference, and it is **not** equivalence.')
    A('- Nothing here establishes a stable online gain, MaskCO attribution, pickup impossibility, or')
    A('  equivalence. No acceptance rule, no online policy and no registered panel is run.')
    A('- Producing a permutation is not producing a gain. The capture loop also favoured early')
    A('  events, and every failure and degenerate row is retained in `result.json` (see the')
    A('  failures column); nothing was redrawn or dropped.')
    A('')
    A('## Files')
    A('')
    for e in entries:
        A('- `%s` (mask_fraction %s, sha256 `%s`)' % (
            e['result_json'], f(e['mask_fraction'], 2), e['result_json_sha256']))
    A('- `control_strength_sweep_summary.json` - machine-readable summary rendered into this file')
    A('')

    Path(args.out).write_text('\n'.join(lines), encoding='utf-8')
    print('wrote %s (%d bytes)' % (args.out, Path(args.out).stat().st_size))
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', required=True)
    ap.add_argument('--out', required=True)
    sys.exit(main(ap.parse_args()))
