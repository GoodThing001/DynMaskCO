"""P-ATTRIB-v1: what actually binds the pickup accept decision?

Origin. P-BUDGET-v1 read the unspent 9.3 percent of the C0 energy budget as
indivisibility; P-RESIDUAL-v1 refuted that (the residual is about 18 times the cheapest
rejected round trip and rejected orders fit inside it on 36 of 40 instances). The
residual is therefore a symptom. This panel attributes the cause.

The reference policy rejects an order at reveal time for exactly one of three reasons,
and they are separable inside its own code path:
  threshold            REV[class] <= est_energy * lam - a deliberate reservation of
                       budget for orders that have not been revealed yet
  no_feasible_route    _try_insert found no vehicle and position where the order fits
                       the time windows and capacity
  budget_certify_failed  the order could be routed, but the true C0 certification of the
                       resulting plan exceeded the energy budget

Method. Subclasses the reference policy and reproduces its accept logic verbatim, with
one change: it decomposes _try_insert_certified into its two internal stages so the
failure can be attributed, and rolls the plan back exactly as the original does.
Reproduction check: the mean served count must reproduce the recorded 83.7 exactly; the
panel aborts rather than reports if it does not, because a changed behaviour would make
the attribution meaningless.

Scope. This is attribution over the reference policy's own decisions on the frozen A-v1
cohort. It identifies which constraint is binding; it does not test any intervention and
establishes no gain.

Usage:
    python pickup_rejection_attribution.py --gate GATE.json --out NEWDIR [--test-instances N]
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
for sub in ('evaluation', 'coldchain', 'simulation'):
    sys.path.insert(0, str(SCRIPTS / sub))

from coldchain_contract import ColdChainContractV2                 # noqa: E402
from coldchain_evaluator_a1 import REV, add_v2_initial_quality     # noqa: E402
from run_exp_energy_c0 import (BOOKING_HORIZON, KM_PER_UNIT, SPEED_KMH,
                               certify_plan, generate_dataset)       # noqa: E402
from run_exp_rule_control import UncondDynamicReplanner            # noqa: E402
from strict_online_env import StrictOnlineEnv                      # noqa: E402

REASONS = ('threshold', 'no_feasible_route', 'budget_certify_failed')

PROTOCOL = {
    'version': 'P-ATTRIB-v1',
    'kind': 'binding_constraint_attribution_over_the_reference_policy',
    'parents': ['P-BUDGET-v1', 'P-RESIDUAL-v1'],
    'policy': 'uncond_dynamic (non-learned reference), accept logic reproduced verbatim',
    'reasons': list(REASONS),
    'decomposition': ('_try_insert_certified split into _try_insert (time windows and '
                      'capacity) and certify_plan (true C0 energy budget) so the failure '
                      'stage is attributable; the plan is rolled back exactly as the '
                      'original does'),
    'budget_source': 'recorded A-v1 gate, not recomputed',
    'recorded_reference_mean_served': 83.7,
    'claims_online_gain': False,
    'tests_an_intervention': False,
}


class AttributionReplanner(UncondDynamicReplanner):
    """Reference accept logic, verbatim, with the certification stage decomposed."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.reasons = []

    def on_reveal(self, env, inst_idx, clock, vehicles, served_mask, visible_ids):
        self._reset_if_new(inst_idx)
        reserved = env.get_reserved_customers(vehicles)
        new_set = set(int(i) for i in visible_ids if not served_mask[i]
                      and int(i) not in reserved and int(i) not in self._accepted
                      and int(i) not in self._rejected)
        for o in sorted(new_set, key=lambda i: env.reveal_time[inst_idx, i]):
            e = self._est_energy(env, inst_idx, o)
            lam = self._lookahead_lambda(env, inst_idx, clock, o)
            if REV[int(env.dataset['temp_class'][inst_idx, o])] <= e * lam + 1e-9:
                self._rejected.add(o)
                self.reasons.append('threshold')
                continue
            saved_plan = {k: list(v) for k, v in self._plan.items()}
            if not self._try_insert(env, inst_idx, o, vehicles, served_mask, clock):
                self.reasons.append('no_feasible_route')
                self._rejected.add(o)
                continue
            try:
                ok, _ = certify_plan(env, inst_idx, clock, vehicles, served_mask,
                                     self._plan, self.contract, self.budget)
            except ValueError:
                ok = False
            if not ok:
                self._plan = saved_plan
                self.reasons.append('budget_certify_failed')
                self._rejected.add(o)
                continue
            self._accepted.add(o)


def main(args):
    started = time.perf_counter()
    gate_path = Path(args.gate)
    gate_raw = gate_path.read_bytes()
    gate = json.loads(gate_raw.decode('utf-8'))
    B = float(gate['budget']['B'])
    cooling_share = float(gate['budget']['cooling_share'])
    cfg = gate['config']
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    contract = ColdChainContractV2()
    contract.validate()
    train_ds = generate_dataset(cfg['train_instances'], cfg['n_orders'], cfg['seed'])
    n_test = cfg['test_instances'] if args.test_instances is None else args.test_instances
    test_ds = generate_dataset(n_test, cfg['n_orders'], cfg['seed'] + 1)
    test_ds = add_v2_initial_quality(test_ds, contract)
    hist = []
    for i in range(cfg['train_instances']):
        dem = train_ds['demands'][i]
        mask = dem[1:] > 0
        hist.append((train_ds['reveal_time'][i][1:][mask],
                     train_ds['temp_class'][i][1:][mask].astype(int)))

    replanner = AttributionReplanner(
        budget=B, capacity=cfg['capacity'], booking_horizon=BOOKING_HORIZON,
        contract=contract, cooling_share=cooling_share, hist=hist, k=cfg['k'])
    env = StrictOnlineEnv(test_ds, capacity=cfg['capacity'], num_vehicles=cfg['num_vehicles'],
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=replanner,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)

    rows = []
    for i in range(n_test):
        replanner.reasons = []
        traces, _ = env.run(i)
        counts = {r: replanner.reasons.count(r) for r in REASONS}
        rows.append({'instance': i, 'served': len(replanner._accepted),
                     'rejected': len(replanner._rejected),
                     'counts': counts,
                     'reject_rate_by_class': {
                         str(c): float(np.mean([1.0 if int(test_ds['temp_class'][i][o]) == c else 0.0
                                                for o in replanner._rejected])) if replanner._rejected else None
                         for c in range(3)}})
        print(json.dumps({'instance': i, 'served': len(replanner._accepted),
                          'rejected': len(replanner._rejected), **counts}), flush=True)

    mean_served = statistics.fmean(r['served'] for r in rows)
    recorded = PROTOCOL['recorded_reference_mean_served']
    full_cohort = args.test_instances is None
    reproduced = abs(mean_served - recorded) <= 0.05
    if full_cohort and not reproduced:
        raise RuntimeError('Attribution policy did not reproduce the recorded mean served '
                           '%.2f (got %.2f); behaviour changed, aborting' % (recorded,
                                                                             mean_served))

    totals = {r: sum(row['counts'][r] for row in rows) for r in REASONS}
    grand = sum(totals.values())
    result = {
        'protocol': PROTOCOL,
        'source_gate': str(gate_path).replace('\\', '/'),
        'source_gate_sha256': hashlib.sha256(gate_raw).hexdigest(),
        'budget_B': B, 'n_instances': len(rows),
        'reproduction': {'recorded_mean_served': recorded,
                         'observed_mean_served': mean_served,
                         'full_cohort_checked': full_cohort,
                         'reproduced_within_0.05': reproduced},
        'totals': totals,
        'shares': {r: (totals[r] / grand if grand else None) for r in REASONS},
        'mean_rejected_per_instance': statistics.fmean(r['rejected'] for r in rows),
        'binding_constraint': max(totals, key=totals.get) if grand else None,
        'per_instance': rows,
        'online_gain_established': False,
        'stable_gain_established': False,
        'wall_s': time.perf_counter() - started,
    }
    with (out / 'PROTOCOL.json').open('x', encoding='utf-8') as f:
        json.dump(PROTOCOL, f, indent=2)
    with (out / 'result.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({'reproduction': result['reproduction'], 'totals': totals,
                      'shares': result['shares'],
                      'binding_constraint': result['binding_constraint'],
                      'wall_s': round(result['wall_s'], 1)}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--test-instances', type=int, default=None)
    main(ap.parse_args())
