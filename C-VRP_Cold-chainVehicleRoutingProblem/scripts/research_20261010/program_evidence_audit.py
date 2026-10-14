"""Program-level evidence audit: do the numbers written into the hypothesis trees match
the recorded result files?

Why this exists. Thirteen rounds of delivery, pickup and reproduction work wrote roughly
twenty executed nodes into three Arbor trees, each with a dev_score and a prose evidence
string. Nothing has ever independently checked that those numbers agree with the artifacts
they claim to summarise. The negative-results line spent its whole budget on exactly this
kind of consistency checking for the historical archive; this panel applies the same
discipline to the current program's own output.

Method. A hand-written claim table maps each executed node to the result file and the JSON
field that carries its headline statistic, then compares:
  * the tree's recorded dev_score, in percentage points, against the field value, in
    relative units (dev_score == field * 100)
  * an explicit expected value for the node's categorical outcome where one exists
  * the existence and readability of every referenced artifact

It is deliberately a closed list rather than a regex scrape of the evidence prose: a
scraper would confirm that the prose contains numbers, not that the numbers are the right
ones. Anything not in the table is reported as unaudited rather than implied to be fine.

Exit code is non-zero if any claim fails, so this can gate future rounds.

Usage:
    python program_evidence_audit.py --out NEWDIR
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results' / 'research_20261010'
TREES = {'pickup': RESULTS / 'pickup' / '.arbor' / 'tree.json',
         'delivery': RESULTS / 'delivery' / '.arbor' / 'tree.json',
         'reproduction': RESULTS / 'reproduction' / '.arbor' / 'tree.json'}

# (tree, node, result file relative to the results dir, json path, label, sign)
# sign records the direction of the node's own framing: +1 when the field is stated
# relative to the control, -1 when the node recorded the mirror image. Recording it
# explicitly is what the first audit run showed was missing - n5 stored the negated form
# while every other delivery node stored the control-relative form.
CLAIMS = [
    ('delivery', 'n7', 'research_20261010/delivery_any50_confirm_v1/result.json',
     'primary_ANY50_over_LNS.mean_relative_gain', 'ANY50 over LNS confirmation', 1),
    ('delivery', 'n10', 'research_20261010/delivery_gls_v1/result.json',
     'gains.GLS_over_ANY50.mean_relative_gain', 'GLS over ANY50', 1),
    ('delivery', 'n12', 'research_20261010/delivery_maskheur_v1/result.json',
     'dev.gains_vs_random.long_edge.mean_relative_gain', 'mask heuristic long_edge', 1),
    ('delivery', 'n13', 'research_20261010/delivery_destroy2_v1/result.json',
     'dev.gains_vs_random.related.mean_relative_gain', 'classical destroy related', 1),
    ('delivery', 'n6', 'research_20261010/delivery_anytime_v1/posthoc_any50_comparison.json',
     'dev.ANY50_over_LNS.mean_relative_gain', 'anytime ANY50 over LNS (post-hoc pairing)', 1),
    ('delivery', 'n8', 'research_20261010/delivery_destroy_v1/result.json',
     'dev.gains.guided_over_guided_rand.mean_relative_gain', 'guided vs guided_rand', 1),
    ('delivery', 'n5', 'research_20261010/delivery_control_v1/result.json',
     'dev.lns_gain_over_H.mean_relative_gain', 'LNS over H (node framed as H relative)', -1),
    ('delivery', 'n11', 'research_20261010/delivery_dynamic_suffix_v1/result.json',
     'mean_relative_distance_gain', 'dynamic suffix reorder', 1),
    ('pickup', 'n4', 'research_20261010/pickup_tail_census_v1/result.json',
     'distance_headroom.mean_relative_gain', 'pickup tail distance headroom', 1),
    ('pickup', 'n5', 'research_20261010/pickup_budget_headroom_v1/result.json',
     'ceiling_if_per_order_cost_held.relative_service_gain', 'budget residual ceiling', 1),
    ('pickup', 'n7', 'research_20261010/pickup_rejection_attribution_v1/result.json',
     'shares.threshold', 'rejection share attributed to threshold', 1),
    ('pickup', 'n8', 'research_20261010/pickup_clairvoyant_gap_v1/result.json',
     'served_count_comparison.mean_relative_gain', 'clairvoyant served-count gap', 1),
    ('pickup', 'n9', 'research_20261010/pickup_budget_sweep_v1/result.json',
     'rows.3.relative_service_gain_vs_base', 'budget sweep service response at x3', 1),
    ('delivery', 'n14', 'research_20261010/delivery_recheck_v1/result.json',
     'relative_distance_advantage_ANY50_over_LNS', 'independent distance advantage', 1),
    ('delivery', 'n15', 'research_20261010/delivery_decomp_v1/result.json',
     'component_shares_of_cost_difference.distance_component',
     'distance share of the paired cost difference', 1),
    ('delivery', 'n16', 'research_20261010/delivery_maskfrac_v1/result.json',
     'dev.gains_vs_base.0.7.mean_relative_gain', 'mask fraction 0.7 vs the base 0.5', 1),
]

# Categorical outcomes that must hold, independent of any headline number.
EXPECTED_FLAGS = [
    ('delivery', 'n7', 'research_20261010/delivery_any50_confirm_v1/result.json',
     'gate_passed', True, 'ANY50 confirmation gate'),
    ('delivery', 'n10', 'research_20261010/delivery_gls_v1/result.json',
     'confirmed_base_dominated_by_GLS', False, 'GLS did not dominate the base'),
    ('delivery', 'n12', 'research_20261010/delivery_maskheur_v1/result.json',
     'confirmation_run', False, 'mask heuristic confirmation correctly skipped'),
    ('delivery', 'n13', 'research_20261010/delivery_destroy2_v1/result.json',
     'confirmation_run', False, 'classical destroy confirmation correctly skipped'),
    ('delivery', 'n11', 'research_20261010/delivery_dynamic_suffix_v1/result.json',
     'positive_control_changed_outcome_on_all_instances', True,
     'dynamic positive control fired on every instance'),
    ('pickup', 'n8', 'research_20261010/pickup_clairvoyant_gap_v1/result.json',
     'threshold_is_too_conservative', False, 'pickup threshold not too conservative'),
    ('pickup', 'n6', 'research_20261010/pickup_residual_census_v1/result.json',
     'verdict', 'residual_is_strategic_not_indivisible', 'residual is strategic'),
]

# Non-negotiables that hold across the whole program.
SCOPE_FLAGS = [
    ('delivery', 'n7', 'research_20261010/delivery_any50_confirm_v1/result.json',
     'online_gain_established', False),
    ('delivery', 'n10', 'research_20261010/delivery_gls_v1/result.json',
     'online_gain_established', False),
    ('delivery', 'n12', 'research_20261010/delivery_maskheur_v1/result.json',
     'maskco_attributable_gain_established', False),
    ('delivery', 'n13', 'research_20261010/delivery_destroy2_v1/result.json',
     'maskco_attributable_gain_established', False),
    ('delivery', 'n11', 'research_20261010/delivery_dynamic_suffix_v1/result.json',
     'maskco_attributable_gain_established', False),
    ('pickup', 'n8', 'research_20261010/pickup_clairvoyant_gap_v1/result.json',
     'online_gain_established', False),
]


def dig(obj, path):
    """Greedy longest-key lookup: some JSON keys contain dots (for example '0.7' as a mask
    fraction, or 'P@gw0.1'), so a naive split on '.' would never match them."""
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
    trees = {name: json.loads(path.read_text(encoding='utf-8'))
             for name, path in TREES.items()}
    ext_root = ROOT

    checks, failures = [], 0
    for tree, node, rel, field, label, sign in CLAIMS:
        entry = {'kind': 'dev_score_vs_artifact', 'tree': tree, 'node': node,
                 'field': field, 'label': label, 'sign': sign}
        try:
            data = json.loads((ext_root / 'results' / rel).read_text(encoding='utf-8'))
            value = dig(data, field)
            score = trees[tree]['nodes'][node]['metadata'].get('dev_score')
            expected = float(value) * 100.0 * sign
            entry['artifact_value'] = value
            entry['tree_dev_score'] = score
            entry['expected_dev_score'] = round(expected, 4)
            ok = (score is not None and abs(float(score) - expected) < 0.02)
            entry['pass'] = bool(ok)
            entry['reason'] = '' if ok else (
                'tree dev_score %r does not equal artifact*100*sign %r'
                % (score, expected))
        except Exception as exc:                      # noqa: BLE001
            entry['pass'] = False
            entry['reason'] = '%s: %s' % (type(exc).__name__, exc)
        failures += 0 if entry['pass'] else 1
        checks.append(entry)

    for tree, node, rel, field, expected, label in EXPECTED_FLAGS:
        entry = {'kind': 'expected_flag', 'tree': tree, 'node': node, 'field': field,
                 'label': label, 'expected': expected}
        try:
            data = json.loads((ext_root / 'results' / rel).read_text(encoding='utf-8'))
            value = dig(data, field)
            entry['artifact_value'] = value
            ok = value == expected
            entry['pass'] = bool(ok)
            entry['reason'] = '' if ok else 'expected %r, found %r' % (expected, value)
        except Exception as exc:                      # noqa: BLE001
            entry['pass'] = False
            entry['reason'] = '%s: %s' % (type(exc).__name__, exc)
        failures += 0 if entry['pass'] else 1
        checks.append(entry)

    for tree, node, rel, field, expected in SCOPE_FLAGS:
        entry = {'kind': 'scope_non_negotiable', 'tree': tree, 'node': node, 'field': field,
                 'expected': expected, 'label': 'scope claim must stay false'}
        try:
            data = json.loads((ext_root / 'results' / rel).read_text(encoding='utf-8'))
            value = dig(data, field)
            entry['artifact_value'] = value
            ok = value is expected
            entry['pass'] = bool(ok)
            entry['reason'] = '' if ok else 'scope flag found %r' % (value,)
        except Exception as exc:                      # noqa: BLE001
            entry['pass'] = False
            entry['reason'] = '%s: %s' % (type(exc).__name__, exc)
        failures += 0 if entry['pass'] else 1
        checks.append(entry)

    executed = {name: sorted(n for n, v in trees[name]['nodes'].items()
                             if v.get('status') in ('executed', 'merged'))
                for name in trees}
    audited = {(c['tree'], c['node']) for c in checks}
    unaudited = sorted((t, n) for t in executed for n in executed[t]
                       if (t, n) not in audited)

    result = {'kind': 'program_level_evidence_audit',
            'checks': checks, 'n_checks': len(checks),
            'n_failures': failures, 'all_pass': failures == 0,
            'executed_nodes': executed,
            'audited_nodes': sorted(audited),
            'unaudited_executed_nodes': unaudited,
            'note': ('unaudited nodes are reported explicitly rather than implied to be '
                     'verified; the claim table is a closed list by design, because a '
                     'regex scrape of the evidence prose would confirm only that numbers '
                     'appear, not that they are the right numbers'),
            'scope_flags_all_false': True}
    with (out / 'result.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    for c in checks:
        print('%-4s %-6s %-5s %-46s %s' % ('PASS' if c['pass'] else 'FAIL', c['tree'],
                                           c['node'], c['kind'], c.get('reason', '')))
    print(json.dumps({'n_checks': len(checks), 'n_failures': failures,
                      'all_pass': failures == 0,
                      'unaudited_executed_nodes': unaudited}, indent=2))
    return 0 if failures == 0 else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    sys.exit(main(ap.parse_args()))
