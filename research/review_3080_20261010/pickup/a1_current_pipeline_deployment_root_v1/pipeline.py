"""Current-source online pipeline entry point for the registered A-v1 deployment rule.

`run_a1_step2_gate.run_arm` can no longer be imported (scenario_saa.py dropped
SoftKNNHistoricalSampler and the anytime_vote path), so the recorded online pipeline is
not deployable against the current sources.  This module is a frozen, current-source
replacement that drives the same online loop for two arms under identical conditions:

  rule    the registered deployment rule (clock shadow-price budget-reserving admission)
  control the frozen base SAA with the identical sampler, seed, K and time limit

It never edits the old pipeline; it only imports current modules.  Day-level utility is
computed with the frozen evaluator, and every decision is logged per order.
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
V3 = ROOT / 'research/review_3080_20261010/pickup/a1_deployment_rule_identity_root_v3'
for sub in ('scripts', 'scripts/evaluation', 'scripts/coldchain', 'scripts/simulation'):
    sys.path.insert(0, str(EXT / sub))
sys.path.insert(0, str(V3))

import numpy as np  # noqa: E402


def run_arm(arm, gate_ds, days, contract, B, cooling_share, train_ds, cfg, rule):
    """Run one arm over `days`; return per-day rows plus the rule decision log."""
    from coldchain_evaluator_a1 import evaluate_trace_a1
    from scenario_saa import build_history, ExplicitFeatureSampler, SaaReplanner
    from strict_online_env import StrictOnlineEnv
    from run_exp_reserve import SPEED_KMH, KM_PER_UNIT, BOOKING_HORIZON
    import deployment_rule as DR  # the v3 frozen rule, reused read-only

    objective = cfg['objective']
    lam_grid = rule['lambda_grid']
    clock_grid = rule['clock_grid']
    g_table = rule['g_table']
    p_c = {int(k): float(v) for k, v in cfg['penalty_tiers'].items()}
    panel = {k: v[:max(days) + 1] for k, v in gate_ds.items()}
    log = []
    decisions = dict(total=0, admit=0, reject_price=0, reject_budget=0, base_accepted=0)
    events = []

    class RuleReplanner(SaaReplanner):
        def _saa_decide(self, env, inst_idx, clock, vehicles, served_mask, o, scenarios, t0):
            decisions['total'] += 1
            c = int(env.dataset['temp_class'][inst_idx, o])
            q = float(env.demands[inst_idx, o])
            e = DR.e_lower_of(c, q, objective)
            spent = 0.0
            for other in self._accepted:
                spent += DR.e_lower_of(int(env.dataset['temp_class'][inst_idx, other]),
                                       float(env.demands[inst_idx, other]), objective)
            b_rem = float(B) - spent
            admit, reason, lam, ti = DR.decide(e, DR.REV[c], clock, b_rem, lam_grid, clock_grid, g_table)
            entry = dict(arm=arm, inst=int(inst_idx), clock=float(clock), order=int(o), class_id=c, quantity=q,
                         e_lower=e, revenue=DR.REV[c], spent_before=spent, b_rem=b_rem,
                         bucket=(None if ti is None else int(ti)), lam=(None if lam is None else float(lam)),
                         rule_decision=('admit' if admit else 'reject'), reason=reason)
            if not admit:
                decisions['reject_budget' if reason == 'over_remaining_budget' else 'reject_price'] += 1
                self._rejected.add(o)
                entry['base_decision'] = 'not_reached'
                log.append(entry)
                return
            decisions['admit'] += 1
            super()._saa_decide(env, inst_idx, clock, vehicles, served_mask, o, scenarios, t0)
            if o in self._accepted:
                decisions['base_accepted'] += 1
                entry['base_decision'] = 'accepted'
            else:
                entry['base_decision'] = 'rejected'
            log.append(entry)

    kw = dict(budget=B, capacity=cfg['capacity'], booking_horizon=BOOKING_HORIZON, contract=contract,
              cooling_share=cooling_share, sampler=ExplicitFeatureSampler(build_history(train_ds)),
              reject_penalty=p_c, K=cfg['K'], time_limit=cfg['time_limit_s'], arm_seed=cfg['arm_seed'],
              shadow_mode=cfg['shadow_mode'], energy_pricing=cfg['energy_pricing'],
              future_policy=cfg['future_policy'], standby_orders=None, overage_price=None, incr_eval=False)
    rp = RuleReplanner(**kw) if arm == 'rule' else SaaReplanner(**kw)
    env = StrictOnlineEnv(panel, capacity=cfg['capacity'], num_vehicles=cfg['num_vehicles'],
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    rows = []
    for i in days:
        t0 = time.perf_counter()
        traces, _ = env.run(i)
        acc, rej = set(rp._accepted), set(rp._rejected)
        ev = evaluate_trace_a1(traces, {k: v[i] for k, v in panel.items()}, contract, acc, rej, B, p_c)
        elapsed = time.perf_counter() - t0
        rows.append(dict(arm=arm, day=int(i),
                         utility=(float(ev['utility']) if np.isfinite(ev['utility']) else None),
                         hard_feasible=bool(ev['hard_feasible']), failures=sorted(set(ev['failures'])),
                         n_served=int(ev['n_served']), revenue=float(ev['revenue']),
                         energy_kwh=float(ev['energy_kwh']), budget_violated=bool(ev['budget_violated']),
                         request_seconds=elapsed,
                         timeouts=int(getattr(rp, 'timeouts', 0)),
                         partial_commits=int(getattr(rp, 'partial_commits', 0)),
                         over_time_limit=bool(elapsed > cfg['time_limit_s'])))
    return rows, log, decisions
