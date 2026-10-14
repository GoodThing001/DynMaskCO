"""D-ANYTIME-v1 post-hoc paired comparison: ANY50 against LNS and ANY30.

Why this file exists. D-ANYTIME-v1 registered ANY30 as the primary arm and ANY50 as a
secondary mask-size probe, and its `gains` table therefore contains ANY30-vs-H8,
ANY50-vs-H8, LNS-vs-ANY30 and LNS-vs-H8 -- but not ANY50-vs-LNS. The recorded means
show ANY50 is the cheapest arm in both cohorts, so the decisive comparison was missing
and must be computed rather than eyeballed.

Status of this analysis, stated plainly: the ARMS are pre-registered, but this
particular pairing was added after seeing the run. It is therefore a post-hoc
comparison of pre-registered arms, not a pre-registered test, and it must not be
reported as an independently confirmed result. Its only legitimate use is to justify
registering a fresh, independent confirmation panel with ANY50 as a pre-registered
primary arm.

Reads the recorded per-instance progress files; recomputes nothing about the search.
"""
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results' / 'research_20261010' / 'delivery_anytime_v1'
BOOTSTRAP_SEED = 2026101133      # same seed as the registered protocol
REPETITIONS = 2000
THRESHOLD = 0.01


def paired(control, other, name_control, name_other):
    shared = sorted(set(control) & set(other))
    diffs = np.array([(control[i] - other[i]) / control[i] for i in shared], float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(diffs), size=(REPETITIONS, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    mean = float(diffs.mean())
    if mean >= THRESHOLD and lo > 0:
        verdict = 'other_dominates'
    elif mean <= -THRESHOLD and hi < 0:
        verdict = 'control_dominates'
    else:
        verdict = 'indistinguishable'
    return {'control': name_control, 'other': name_other, 'n_instances': len(shared),
            'mean_relative_gain': mean, 'descriptive_95_CI': [lo, hi],
            'instances_other_better': int(sum(d > 0 for d in diffs)),
            'instances_other_worse': int(sum(d < 0 for d in diffs)),
            'instances_tied': int(sum(d == 0 for d in diffs)), 'verdict': verdict}


def main():
    result = {'analysis': 'posthoc_paired_comparison_of_preregistered_arms',
              'bootstrap_seed': BOOTSTRAP_SEED, 'repetitions': REPETITIONS,
              'not_a_preregistered_test': True,
              'legitimate_use': 'justify a fresh independent confirmation panel with ANY50 as a preregistered primary arm'}
    for cohort, fname in (('dev', 'dev_progress.jsonl'),
                          ('confirmation', 'confirmation_progress.jsonl')):
        path = OUT / fname
        if not path.exists():
            sys.exit('missing recorded progress file: ' + str(path))
        rows = [json.loads(line) for line in path.open(encoding='utf-8')]
        by = {arm: {r['instance_id']: r['arms'][arm]['final_cost'] for r in rows}
              for arm in ('H8', 'ANY30', 'ANY50', 'LNS')}
        entry = {'n_instances': len(rows),
                 'means': {arm: statistics.fmean(v.values()) for arm, v in by.items()},
                 'ANY50_over_LNS': paired(by['LNS'], by['ANY50'], 'LNS', 'ANY50'),
                 'ANY50_over_ANY30': paired(by['ANY30'], by['ANY50'], 'ANY30', 'ANY50'),
                 'LNS_over_ANY50': paired(by['ANY50'], by['LNS'], 'ANY50', 'LNS')}
        result[cohort] = entry
        print('==', cohort)
        print('   means:', {k: round(v, 4) for k, v in entry['means'].items()})
        for key in ('ANY50_over_LNS', 'ANY50_over_ANY30'):
            g = entry[key]
            print('   %-16s %+.4f CI[%+.4f,%+.4f] better/worse %d/%d -> %s' % (
                key, g['mean_relative_gain'], g['descriptive_95_CI'][0],
                g['descriptive_95_CI'][1], g['instances_other_better'],
                g['instances_other_worse'], g['verdict']))
    target = OUT / 'posthoc_any50_comparison.json'
    with target.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print('wrote', target)


if __name__ == '__main__':
    main()
