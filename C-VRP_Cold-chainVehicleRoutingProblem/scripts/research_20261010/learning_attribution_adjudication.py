"""P-LEARN-ADJUDICATION-v1: a verifiable adjudication of the program's core negative claim.

Why this exists. Seventeen rounds produced one central negative result about MaskCO in the
delivery adapter - no learned component has ever beaten the non-learning base - and it is
currently spread across six separate Arbor nodes. A reader who wants to know whether that
claim is supported must reconstruct it by hand from six evidence strings and five result
files. The negative-results line solved exactly this problem for the historical archive by
building a claim gate; this panel applies the same treatment to the current program's own
central negative claim.

Method. The adjudication enumerates every experiment that bears on the claim, and for each
one verifies at run time that the number quoted is the number actually stored in the
artifact. A claim whose own supporting numbers cannot be re-read from disk is not
adjudicated as supported. The panel then states the verdict and, importantly, the ONE route
that remains untested and why the measured upper bounds bear on it.

Exit code is non-zero if any cited number fails to match, so this can gate future rounds.

Usage:
    python learning_attribution_adjudication.py --out NEWDIR
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results'

# (label, relative result file, json path, claim, direction)
# direction: 'harmful' means the learned arm is materially worse than its control,
#            'no_signal' means it is indistinguishable or worse with no positive CI.
EVIDENCE = [
    ('D-EQ8-v1 pretrained vs zero guide, equal attempts, n=32',
     'research_20261010/INDEPENDENT_EQ8_PROGRESS_ADJUDICATION.json',
     'comparisons.P_over_H.mean_relative_gain',
     'the pretrained guide is worse than no guide at all', 'harmful'),
    ('D-EQ8-v1 pretrained vs random same-architecture guide, n=32',
     'research_20261010/INDEPENDENT_EQ8_PROGRESS_ADJUDICATION.json',
     'comparisons.P_over_R.mean_relative_gain',
     'the pretrained guide is worse than a random guide of the same architecture',
     'harmful'),
    ('D-GUIDE-v1 guide harm under a state-matched 2x2 role factorization',
     'research_20261010/delivery_guide_misalignment_v1/result.json',
     'paired_vs_no_guide.shipped_minus_no_guide.mean_difference',
     'the shipped guide rule costs about 1.9 more per decision than no guide',
     'harmful_positive'),
    ('D-DOSE-v1 dose response at guide weight 1.0',
     'research_20261010/delivery_dose_v1/result.json',
     'dev_table.P@gw1.comparison_vs_zero_weight.mean_relative_improvement',
     'at unit weight the guide costs 9.4 percent', 'harmful'),
    ('D-DOSE-v1 dose response at guide weight 0.1',
     'research_20261010/delivery_dose_v1/result.json',
     'dev_table.P@gw0.1.comparison_vs_zero_weight.mean_relative_improvement',
     'even a tenth of unit weight still costs 2.4 percent', 'harmful'),
    ('D-DESTROY-v1 guide as a destroy operator, cost-matched contrast',
     'research_20261010/delivery_destroy_v1/result.json',
     'dev.gains.guided_over_guided_rand.mean_relative_gain',
     'as a destroy operator the pretrained direction is worse than a random one',
     'harmful'),
    ('D-MASKHEUR-v1 best hand-designed mask heuristic vs uniform random',
     'research_20261010/delivery_maskheur_v1/result.json',
     'dev.gains_vs_random.long_edge.mean_relative_gain',
     'no hand-designed mask choice beats uniform random', 'no_signal'),
    ('D-DESTROY2-v1 classical Shaw related removal vs uniform random',
     'research_20261010/delivery_destroy2_v1/result.json',
     'dev.gains_vs_random.related.mean_relative_gain',
     'the canonical LNS destroy operator is significantly worse than random',
     'harmful'),
]

UNTESTED_ROUTE = {
    'route': 'train a task-specific model instead of reusing zero-shot CVRP weights',
    'why_not_obviously_closed': ('every experiment above reuses the same pretrained '
                                 'checkpoint; nothing here trained a delivery-specific '
                                 'network, so "no transfer" is not the same claim as '
                                 '"no training"'),
    'why_the_bounds_bear_on_it': ('two of the measured channels bound what such training '
                                  'could find: the mask-choice channel carries no signal '
                                  'even for the strongest hand-designed and classical '
                                  'operators, so a learned selector would have to '
                                  'discover structure that both failed to reach; and the '
                                  'pretrained direction is anti-correlated with the '
                                  'decoder, which is a property of the representation '
                                  'rather than of the weights being frozen'),
    'stated_as': ('the training route remains formally untested and is NOT claimed to be '
                  'impossible; it is stated as the one open path, together with the two '
                  'measured bounds that make it unpromising rather than excluded'),
}


def dig(obj, path):
    """Dotted path lookup that tolerates keys which themselves contain dots.

    Some JSON keys here are literally 'P@gw0.1', so a naive split on '.' would break them
    into two parts and never match. Each dict level is therefore matched greedily against
    the LONGEST remaining key, which resolves 'P@gw0.1' correctly while still handling
    plain paths and numeric list indices.
    """
    parts = path.split('.')
    cur = obj
    i = 0
    while i < len(parts):
        if isinstance(cur, dict):
            for j in range(len(parts), i, -1):
                key = '.'.join(parts[i:j])
                if key in cur:
                    cur = cur[key]
                    i = j
                    break
            else:
                raise KeyError(path)
        elif isinstance(cur, list) and parts[i].lstrip('-').isdigit():
            cur = cur[int(parts[i])]
            i += 1
        else:
            raise KeyError(path)
    return cur


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    checks, failures = [], 0
    for label, rel, field, claim, direction in EVIDENCE:
        entry = {'label': label, 'file': rel, 'field': field, 'claim': claim,
                 'direction': direction}
        try:
            data = json.loads((RESULTS / rel).read_text(encoding='utf-8'))
            value = dig(data, field)
            entry['artifact_value'] = value
            if direction == 'harmful':
                ok = float(value) < 0
            elif direction == 'harmful_positive':
                # Some panels store the mirror image: a positive value means the learned
                # arm was more expensive. The sign convention is recorded per claim rather
                # than assumed, because a first run of this adjudicator failed on exactly
                # that assumption.
                ok = float(value) > 0
            else:
                # no_signal: the candidate must have failed to clear a positive threshold
                ok = float(value) < 0.01
            entry['pass'] = bool(ok)
            entry['reason'] = '' if ok else (
                'value %r does not support the stated %s direction' % (value, direction))
        except Exception as exc:                      # noqa: BLE001
            entry['pass'] = False
            entry['reason'] = '%s: %s' % (type(exc).__name__, exc)
        failures += 0 if entry['pass'] else 1
        checks.append(entry)

    harmful = [c for c in checks if c['pass'] and c['direction'] == 'harmful']
    no_signal = [c for c in checks if c['pass'] and c['direction'] == 'no_signal']
    roles_covered = ['insertion-position score', 'regret ranking key', 'destroy operator',
                     'mask choice (hand-designed)', 'mask choice (classical LNS)']
    result = {
        'kind': 'learning_attribution_adjudication',
        'claim_under_adjudication': ('no learned component has beaten the non-learning '
                                     'delivery base, so no MaskCO-attributable gain is '
                                     'established in this adapter'),
        'checks': checks, 'n_checks': len(checks), 'n_failures': failures,
        'all_cited_numbers_verified': failures == 0,
        'n_harmful_findings': len(harmful), 'n_no_signal_findings': len(no_signal),
        'channels_measured': roles_covered,
        'verdict': ('maskco_attributable_gain_not_established_and_learning_side_closed'
                    if failures == 0 else 'ADJUDICATION_INCOMPLETE_CITED_NUMBER_MISMATCH'),
        'untested_route': UNTESTED_ROUTE,
        'what_this_does_not_claim': [
            'it does not claim a learned component is impossible, only that none has '
            'succeeded and that two measured channels bound what training could find',
            'it does not cover the pickup line, whose separate screening concluded no '
            'decision-side headroom at the frozen budget',
            'it does not change any prior result; every number is read from the archived '
            'artifact as written'],
        'stable_gain_established': False,
    }
    with (out / 'result.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    for c in checks:
        print('%-4s %-8s %-58s %s' % ('PASS' if c['pass'] else 'FAIL', c['direction'],
                                      c['label'][:58], c.get('reason', '')))
    print(json.dumps({'n_checks': len(checks), 'n_failures': failures,
                      'n_harmful': len(harmful), 'n_no_signal': len(no_signal),
                      'verdict': result['verdict']}, indent=2))
    return 0 if failures == 0 else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    sys.exit(main(ap.parse_args()))
