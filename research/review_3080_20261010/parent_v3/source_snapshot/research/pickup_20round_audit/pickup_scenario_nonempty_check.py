"""One-line decisive test for the pickup SAA baseline: do the scenarios contain any orders at all?

Round 15 narrowed four refuted mechanisms down to a single code-grounded candidate that would
explain EVERY pinned observation at once (the SAA arms stuck at exactly -376.40726526174694 across
1/8/60 history days, 2s/10s limits and n_neighbors 10/5): if `_scenario_from_day(day, clock)`
returns an empty order list, then `_build_space` takes its empty branch, the decision space equals
the real nodes and is therefore identical for ANY sampler.

This script answers that directly and needs no environment:
  * build the same history the smoke uses,
  * for a grid of clocks, call `_scenario_from_day` per history day,
  * report how many days yield a non-empty order list, and the total order count.

Read-only. It imports the frozen module and calls one function; nothing is trained, no arm is
compared, and no claim is made beyond the counts it prints.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def main(args):
    scripts = Path(args.scripts).resolve()
    for sub in ('evaluation', 'coldchain', 'simulation'):
        sys.path.insert(0, str(scripts / sub))
    from run_exp_energy_c0 import generate_dataset
    from scenario_saa import build_history, _scenario_from_day

    ds = generate_dataset(args.history_days, 200, 20260923)
    hist = build_history(ds)
    out = {'n_history_days': len(hist), 'per_clock': []}
    for clock in args.clocks:
        nonempty, totals, lengths = 0, 0, []
        for day in hist:
            try:
                scen = _scenario_from_day(day, float(clock))
                n = len(scen)
            except Exception as exc:                      # noqa: BLE001
                n = -1
                out.setdefault('errors', []).append('%s: %s' % (type(exc).__name__, exc))
            lengths.append(n)
            if n > 0:
                nonempty += 1
                totals += n
        out['per_clock'].append({
            'clock': float(clock), 'days_nonempty': nonempty,
            'days_total': len(hist), 'total_orders_across_days': totals,
            'mean_orders_per_day': (float(np.mean([x for x in lengths if x >= 0]))
                                    if lengths else None),
            'min_orders': int(min(lengths)) if lengths else None,
            'max_orders': int(max(lengths)) if lengths else None})
    out['verdict_hint'] = ('if days_nonempty is zero or near zero at the clocks the smoke used, the '
                           'sampler is irrelevant by construction and the whole SAA family is '
                           'unusable as a baseline in this configuration; if it is large, this '
                           'candidate is dead')
    out['what_this_is_not'] = ['not a baseline measurement and not an arm comparison',
                               'no interval, no alpha, no impossibility or equivalence claim']
    print(json.dumps(out, indent=2, default=str))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scripts', required=True)
    ap.add_argument('--history-days', type=int, default=8)
    ap.add_argument('--clocks', type=float, nargs='+', default=[0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    main(ap.parse_args())
