"""P-BUDGETSWEEP-v1: is the pickup ceiling set by decision quality or by budget size?

Origin. The pickup line's screening concluded "no capturable headroom" from four panels,
all of them run at the single frozen C0 energy budget B that covers only 40.0 percent of
an unconditional day. The binding constraint was attributed to the accept threshold
(92.71 percent of rejections), and that threshold beat a greedy full-information
reference. But every one of those statements is conditioned on that one B. If the
threshold is conservative because budget is scarce, then the ceiling is a function of
budget size rather than of decision quality, and the honest conclusion is "no headroom at
this budget" rather than "no headroom".

Method. Scales B by a registered multiplier and re-runs the same non-learned reference
policy, uncond_dynamic, on the same instances. B is taken from the recorded A-v1 gate and
multiplied rather than recomputed, so B is an explicit exogenous knob and the training-side
calibration is untouched. For each multiplier the panel records served count, A-v1 utility
under the main reject penalty, realised budget utilisation, and the share of rejections
attributable to the accept threshold versus true infeasibility - the same attribution
decomposition used by P-ATTRIB-v1, so the mechanism behind any change is visible rather
than inferred.

Registered reading: the ceiling is budget-set rather than decision-set if served count
rises materially with the multiplier (mean relative service gain >= 1 percent between
consecutive multipliers). If served count is flat while utilisation also stays flat, the
constraint is the decision rule and the earlier conclusion stands unqualified.

Scope. Secondary but genuinely new measurement: it changes one exogenous parameter on the
frozen configuration and measures the response. Non-learning throughout, snapshot setting
with the parent physics, no online claim and no MaskCO-attributable gain.

Usage:
    python pickup_budget_sweep.py --gate GATE.json --out NEWDIR [--instances N]
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

from coldchain_contract import ColdChainContractV2                       # noqa: E402
from coldchain_evaluator_a1 import REV, add_v2_initial_quality, \
    evaluate_trace_a1                                                    # noqa: E402
from run_exp_energy_c0 import (BOOKING_HORIZON, KM_PER_UNIT, SPEED_KMH,
                               certify_plan, generate_dataset)           # noqa: E402
from run_exp_rule_control import UncondDynamicReplanner                  # noqa: E402
from strict_online_env import StrictOnlineEnv                            # noqa: E402

PROTOCOL = {
    'version': 'P-BUDGETSWEEP-v1',
    'kind': 'exogenous_budget_response_not_gain_claim',
    'parents': ['P-ATTRIB-v1', 'P-CLAIR-v1'],
    'policy': 'uncond_dynamic (non-learned reference), unchanged',
    'budget_multipliers': [1.0, 1.5, 2.0, 3.0],
    'budget_source': 'recorded A-v1 gate B, multiplied, never recomputed',
    'main_penalty': {'0': 5.0, '1': 10.0, '2': 15.0},
    'reading': ('ceiling is budget-set if served count rises materially with the '
                'multiplier; decision-set if served count stays flat'),
    'claims_online_gain': False,
    'all_failures_retained': True,
}


class AttributionReplanner(UncondDynamicReplanner):
    """Reference accept logic with the certification stage decomposed for attribution."""

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
            saved = {k: list(v) for k, v in self._plan.items()}
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
                self._plan = saved
                self.reasons.append('budget_certify_failed')
                self._rejected.add(o)
                continue
            self._accepted.add(o)


def run_at(budget, test_ds, cfg, contract, cooling_share, hist, n_inst, p_c):
    rp = AttributionReplanner(budget=budget, capacity=cfg['capacity'],
                              booking_horizon=BOOKING_HORIZON, contract=contract,
                              cooling_share=cooling_share, hist=hist, k=cfg['k'])
    env = StrictOnlineEnv(test_ds, capacity=cfg['capacity'],
                          num_vehicles=cfg['num_vehicles'],
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    served, energy, costs, reasons = [], [], [], {'threshold': 0, 'no_feasible_route': 0,
                                                  'budget_certify_failed': 0}
    for i in range(n_inst):
        rp.reasons = []
        traces, _ = env.run(i)
        inst = {k: v[i] for k, v in test_ds.items()}
        e = evaluate_trace_a1(traces, inst, contract, set(rp._accepted),
                              set(rp._rejected), budget, p_c)
        served.append(len(rp._accepted))
        energy.append(float(e['energy_kwh']))
        costs.append(float(e['utility']))
        for r in rp.reasons:
            reasons[r] += 1
    return {'budget': budget, 'mean_served': statistics.fmean(served),
            'mean_utility': statistics.fmean(costs),
            'mean_budget_utilisation': statistics.fmean(energy) / budget,
            'total_rejections': sum(reasons.values()), 'rejections': reasons}


def main(args):
    started = time.perf_counter()
    gate_path = Path(args.gate)
    gate_raw = gate_path.read_bytes()
    gate = json.loads(gate_raw.decode('utf-8'))
    B0 = float(gate['budget']['B'])
    cooling_share = float(gate['budget']['cooling_share'])
    cfg = gate['config']
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    contract = ColdChainContractV2()
    contract.validate()
    train_ds = generate_dataset(cfg['train_instances'], cfg['n_orders'], cfg['seed'])
    n_test = min(cfg['test_instances'], args.instances)
    test_ds = generate_dataset(n_test, cfg['n_orders'], cfg['seed'] + 1)
    test_ds = add_v2_initial_quality(test_ds, contract)
    hist = []
    for i in range(cfg['train_instances']):
        dem = train_ds['demands'][i]
        mask = dem[1:] > 0
        hist.append((train_ds['reveal_time'][i][1:][mask],
                     train_ds['temp_class'][i][1:][mask].astype(int)))
    p_c = {int(k): v for k, v in PROTOCOL['main_penalty'].items()}

    rows = []
    for mult in PROTOCOL['budget_multipliers']:
        row = run_at(B0 * mult, test_ds, cfg, contract, cooling_share, hist, n_test, p_c)
        row['multiplier'] = mult
        row['budget_coverage_fraction'] = (B0 * mult) / float(
            gate['budget']['train_accept_all_c0_energy_mean'])
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in row.items() if k != 'rejections'}), flush=True)

    base = rows[0]
    for row in rows:
        row['relative_service_gain_vs_base'] = (
            (row['mean_served'] - base['mean_served']) / base['mean_served'])
        row['threshold_share'] = (row['rejections']['threshold'] / row['total_rejections']
                                  if row['total_rejections'] else None)
    monotone = all(rows[i + 1]['mean_served'] >= rows[i]['mean_served'] - 1e-9
                   for i in range(len(rows) - 1))
    material = any(r['relative_service_gain_vs_base'] >= 0.01 for r in rows[1:])
    result = {
        'protocol': PROTOCOL,
        'source_gate': str(gate_path).replace('\\', '/'),
        'source_gate_sha256': hashlib.sha256(gate_raw).hexdigest(),
        'base_budget_B': B0, 'n_instances': n_test, 'rows': rows,
        'served_count_monotone_in_budget': bool(monotone),
        'material_service_gain_exists': bool(material),
        'verdict': ('ceiling_is_budget_set' if material
                    else 'ceiling_is_decision_set_even_with_more_budget'),
        'online_gain_established': False,
        'stable_gain_established': False,
        'maskco_attributable_gain_established': False,
        'wall_s': time.perf_counter() - started,
    }
    with (out / 'PROTOCOL.json').open('x', encoding='utf-8') as f:
        json.dump(PROTOCOL, f, indent=2)
    with (out / 'result.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({'rows': [{k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in r.items() if k != 'rejections'}
                               for r in rows],
                      'monotone': monotone, 'material': material,
                      'verdict': result['verdict'],
                      'wall_s': round(result['wall_s'], 1)}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--instances', type=int, default=20)
    main(ap.parse_args())
