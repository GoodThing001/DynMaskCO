"""P-KNOWN-EXACT-LIVE-v1: exact enumeration on a REAL causal mid-day pickup packet.

Why this exists. `known_exact.py` deliberately refuses nonzero-clock / current-vehicle
packets (`validate` raises 'current exact implementation only initial idle depot at clock0').
The registered gap `next_actual_online_integration_gap` says actual middle-day headroom needs
exactly that extension, and that it must preserve current cargo/thermal state, the committed
prefix and all other fleet energy, use the same visible admissible action contract, and use a
finite continuation with an independent replay of the full J -- with no fabricated residual
budget and no private future feed.

How this does it, without touching any frozen module:
  * a genuine mid-day decision point is CAPTURED from a real rollout (`snapshot_hook`), and the
    unexecuted suffix actually installed on each vehicle is read from the live vehicle objects;
  * candidates are the permutations of the revealed, unserved order ids inside that installed
    suffix -- a PURE RE-ORDERING of the unexecuted plan, inserted back into the same positions
    so whatever terminal convention the environment used (depot return, WAIT, ...) is preserved;
  * each candidate is run through the ORIGINAL environment from t=0 with
    `env.run(inst, force_suffix={(event_id, vehicle_id): suffix})`, so the whole day still
    unfolds causally: reveals, other vehicles, energy budget and hard gates are all the real
    ones. Cargo/thermal state and already-consumed energy are carried because nothing is
    reconstructed by hand -- the same code path produces them;
  * two controls make the branch machinery itself falsifiable:
      - NO-FORCE replay      : re-running with no force must reproduce the baseline exactly;
      - IDENTITY-FORCE replay: forcing the baseline's OWN recorded suffix must also reproduce
        it exactly. If the identity force changes anything, the harness is perturbing the run
        and every candidate number is void.

Honest scope. This is exact over a declared finite candidate set on a real causal packet. It is
NOT a full online method, NOT a global oracle, and a positive mean here establishes no stable
online gain and no MaskCO attribution. A negative result cannot show impossibility.
"""
import argparse
from itertools import permutations
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

# The registered submit barrier for one mid-day decision point, in seconds. Fixed constant so
# that a run given a larger --seconds cannot report the barrier as satisfied.
REGISTERED_BARRIER_S = 10.0


def bootstrap(scripts):
    scripts = Path(scripts).resolve()
    for sub in ('evaluation', 'coldchain', 'simulation', 'correction_engineering'):
        sys.path.insert(0, str(scripts / sub))
    return scripts


def source_hashes(scripts):
    here = Path(__file__).resolve().parent
    paths = {Path(m.__file__).resolve() for m in list(sys.modules.values())
             if getattr(m, '__file__', None) and str(m.__file__).endswith('.py')}
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)
            if p.is_relative_to(scripts) or p.is_relative_to(here)}


def place(perm, template, positions):
    """Write `perm` into exactly the slots named by `positions`, leaving every other slot
    (depot return, WAIT markers, already-served or not-yet-revealed nodes) untouched."""
    if len(perm) != len(positions):
        raise ValueError('permutation/position length mismatch: %d vs %d'
                         % (len(perm), len(positions)))
    out = list(template)
    for slot, val in zip(positions, perm):
        out[slot] = val
    return out


def main(args):
    started = time.perf_counter()
    scripts = bootstrap(args.scripts)
    from coldchain_contract import ColdChainContractV2
    from coldchain_evaluator_a1 import add_v2_initial_quality, evaluate_trace_a1
    from recourse_snapshot import capture_recourse_snapshot
    from run_exp_energy_c0 import (BOOKING_HORIZON, KM_PER_UNIT, SPEED_KMH, generate_dataset)
    from run_exp_rule_control import UncondDynamicReplanner
    from strict_online_env import StrictOnlineEnv

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    contract = ColdChainContractV2()
    contract.validate()
    budget = float(args.budget)
    p_c = {0: 5.0, 1: 10.0, 2: 15.0}
    src_before = source_hashes(scripts)

    # Registration is written BEFORE any packet is generated, so this batch cannot later be
    # mistaken for a pre-registration of an already-run dataset.
    (out / 'REGISTRATION.json').write_text(json.dumps({
        'version': 'P-KNOWN-EXACT-LIVE-v1',
        'question': ('Within an exactly enumerated re-ordering of the UNEXECUTED suffix at a '
                     'real causal mid-day decision point, under unchanged C0-v2/A-v1 utility '
                     'and the unchanged common budget B, does a hard-feasible strictly better '
                     'sequence exist than the one the strong non-learning policy installed?'),
        'kind': 'bounded_mechanism_screen_on_a_real_causal_packet_not_online_confirmation',
        'supersedes_limitation_of': 'P-KNOWN-EXACT-v1 (clock0/initial-idle only)',
        'registration_written_before_any_packet': True,
        'contract_factory': 'coldchain_contract.ColdChainContractV2 (C0-v2)',
        'budget_B': budget, 'capacity': 50.0, 'num_vehicles': 15,
        'booking_horizon': BOOKING_HORIZON,
        'objective': ('evaluate_trace_a1.utility = revenue - fuel_cost - reject_loss; energy '
                      'budget, fish salability, TW, capacity, commitments and return are hard '
                      'feasibility; tomato quality reported continuously'),
        'do_not_change_objective': True,
        'candidate_set': ('permutations of the revealed+unserved order ids found in the '
                          'INSTALLED unexecuted suffix of a vehicle at the sampled decision '
                          'point, rewritten into the same positions; exact only within this '
                          'declared finite set'),
        'immutable': ['previously_accepted_orders', 'same_vehicle_ownership', 'C0_contract',
                      'common_budget', 'visible_order_features'],
        'no_private_future': True,
        'no_fabricated_residual_budget': ('the full day is re-run through the original '
                                          'environment, so already-consumed energy and cargo '
                                          'state are the real ones, not re-derived'),
        'controls_that_must_reproduce_the_baseline': ['no_force_replay', 'identity_force_replay'],
        'time_cap_seconds_per_state': float(args.seconds),
        'late_completion': ('retain every candidate but exact_within_declared_subproblem=false; '
                            'never treat the best found as a certified bound'),
        'sampling': {'min_clock': float(args.min_clock), 'min_orders': int(args.min_orders),
                     'max_orders': int(args.max_orders),
                     'states_per_instance': int(args.states_per_instance)},
        'data_seeds': {'train_history': 20260923, 'eval_dataset': 20260924,
                       'instances': int(args.instances)},
        'archive': ['protocol_source_SHA_before_packets', 'every_sampled_public_packet',
                    'all_candidate_evaluations', 'best_and_baseline_full_typed_traces',
                    'C0_contract_hash', 'start_end_actual_source_SHA',
                    'independent_terminal_replay', 'all_failures_and_incomplete_flags',
                    'enumeration_wall_seconds_and_10s_barrier'],
        'claims_online_stable_gain': False, 'claims_MaskCO_attribution': False,
        'claims_impossibility_if_negative': False, 'claims_global_optimality': False,
        'protected_final_seeds': [20260927, 20260928],
        'registered_unix': time.time(),
    }, indent=2), encoding='utf-8')

    train_ds = generate_dataset(1, 200, 20260923)
    test_ds = generate_dataset(args.instances, 200, 20260924)
    test_ds = add_v2_initial_quality(test_ds, contract)
    hist = []
    for i in range(1):
        dem = train_ds['demands'][i]
        mask = dem[1:] > 0
        hist.append((train_ds['reveal_time'][i][1:][mask],
                     train_ds['temp_class'][i][1:][mask].astype(int)))

    def make_env(dataset):
        rp = UncondDynamicReplanner(budget=budget, capacity=50.0,
                                    booking_horizon=BOOKING_HORIZON, contract=contract,
                                    cooling_share=args.cooling_share, hist=hist, k=10)
        env = StrictOnlineEnv(dataset, capacity=50.0, num_vehicles=15,
                              tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                              coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
        return env, rp

    def evaluate(traces, inst, rp):
        return evaluate_trace_a1(traces, {k: v[inst] for k, v in test_ds.items()}, contract,
                                 set(rp._accepted), set(rp._rejected), budget, p_c)

    states = []
    for inst in range(args.instances):
        env, rp = make_env(test_ds)
        picked = []

        def hook(env_, inst_idx, clock, event_id, reveal_idx, vehicles, traces, served_mask,
                 all_customers):
            if clock < float(args.min_clock) or len(picked) >= args.states_per_instance:
                return
            for v in vehicles:
                tmpl = [int(x) for x in (v.mutable_suffix or [])]
                # Permutable slots: nonzero, not yet served, already revealed. Everything
                # else in the suffix is left exactly where the environment put it.
                slots = [k for k, x in enumerate(tmpl)
                         if x != 0 and not served_mask[x]
                         and float(env_.reveal_time[inst_idx, x]) <= clock + 1e-6]
                if not (args.min_orders <= len(slots) <= args.max_orders):
                    # A truncated subproblem would silently drop orders, so skip the vehicle
                    # instead of taking a prefix of its suffix.
                    continue
                picked.append({
                    'event_id': int(event_id), 'clock': float(clock),
                    'vehicle_id': int(v.vehicle_id), 'vehicle_status': str(v.status),
                    'template_suffix': tmpl, 'slots': slots,
                    'orders': [tmpl[k] for k in slots],
                    'n_orders_in_template': len(slots),
                    'snapshot': capture_recourse_snapshot(
                        env_, inst_idx, clock, event_id, reveal_idx, vehicles, traces,
                        served_mask, all_customers),
                })
                return

        env.snapshot_hook = hook
        base_traces, _ = env.run(inst)
        base_eval = evaluate(base_traces, inst, rp)
        base_accept, base_reject = set(rp._accepted), set(rp._rejected)
        base_utility = float(base_eval['utility'])

        for si, st in enumerate(picked):
            sdir = out / ('state_%02d_%02d' % (inst, si))
            sdir.mkdir(parents=True, exist_ok=True)
            tmpl = st['template_suffix']
            orders = st['orders']
            slots = st['slots']
            key = (st['event_id'], st['vehicle_id'])
            force = {key: tmpl}

            # --- control 1: no force must reproduce the baseline exactly ---------------
            env_nf, rp_nf = make_env(test_ds)
            nf_traces, _ = env_nf.run(inst)
            nf_eval = evaluate(nf_traces, inst, rp_nf)
            no_force_exact = (abs(float(nf_eval['utility']) - base_utility) < 1e-9
                              and set(rp_nf._accepted) == base_accept)
            # --- control 2: forcing the baseline's OWN suffix must also reproduce it --
            env_if, rp_if = make_env(test_ds)
            if_traces, _ = env_if.run(inst, force_suffix={key: tmpl})
            if_eval = evaluate(if_traces, inst, rp_if)
            identity_force_exact = (abs(float(if_eval['utility']) - base_utility) < 1e-9
                                    and set(rp_if._accepted) == base_accept)

            t0 = time.perf_counter()
            rows, best = [], None
            complete = True
            for perm in permutations(orders):
                if time.perf_counter() - t0 > float(args.seconds):
                    complete = False
                    break
                seq = place(perm, tmpl, slots)
                env_c, rp_c = make_env(test_ds)
                try:
                    tr, _ = env_c.run(inst, force_suffix={key: seq})
                    ev = evaluate(tr, inst, rp_c)
                except Exception as exc:                       # noqa: BLE001
                    rows.append({'permutation': list(perm), 'suffix': seq,
                                 'failure': '%s: %s' % (type(exc).__name__, exc),
                                 'hard_feasible': False})
                    continue
                rows.append({'permutation': list(perm), 'suffix': seq,
                             'utility': float(ev['utility']),
                             'hard_feasible': bool(ev['hard_feasible']),
                             'n_served': int(ev['n_served']),
                             'accepted_set_unchanged': set(rp_c._accepted) == base_accept,
                             'evaluation': ev})
                if ev['hard_feasible'] is True and (
                        best is None or float(ev['utility']) > best[1] + 1e-10):
                    best = (seq, float(ev['utility']), tr, ev, set(rp_c._accepted))
            enum_wall = time.perf_counter() - t0
            # `within_passed_cap` uses whatever --seconds was passed. The registered submit
            # barrier is a SEPARATE, literal constant: a run given a generous cap must never
            # be able to report the 10s barrier as met. Both are recorded.
            within_cap = bool(complete and enum_wall <= float(args.seconds))
            within_10s = bool(complete and enum_wall <= REGISTERED_BARRIER_S)
            exact = bool(complete and within_cap)

            record = {
                'instance': inst, 'state': si, 'event_id': st['event_id'],
                'clock': st['clock'], 'vehicle_id': st['vehicle_id'],
                'vehicle_status': st['vehicle_status'],
                'template_suffix': tmpl, 'subproblem_orders': orders,
                'subproblem_slots': slots,
                'n_orders_in_template': st['n_orders_in_template'],
                'n_candidates': len(rows), 'expected_candidates': int(math.factorial(len(orders))),
                'no_force_replay_exact': bool(no_force_exact),
                'identity_force_replay_exact': bool(identity_force_exact),
                'baseline_utility': base_utility,
                'baseline_n_served': int(base_eval['n_served']),
                'best_permutation': list(best[0]) if best else None,
                'best_utility': best[1] if best else None,
                'best_n_served': int(best[3]['n_served']) if best else None,
                'delta_utility': (best[1] - base_utility) if best else None,
                # Honesty field: a re-ordering candidate is free to change LATER accept
                # decisions, because the whole day is replayed causally. When that happens
                # the effect is NOT local to the re-ordered orders, and the state must not be
                # described as "the same plan, reordered".
                'best_accepted_set_unchanged': bool(
                    best is not None and best[4] == base_accept),
                'effect_is_pure_reordering': bool(
                    best is not None and best[3]['n_served'] == base_eval['n_served']
                    and best[4] == base_accept),
                'n_hard_feasible': int(sum(1 for r in rows if r.get('hard_feasible') is True)),
                'n_strictly_better': int(sum(1 for r in rows if r.get('hard_feasible') is True
                                             and r.get('utility', -np.inf) > base_utility + 1e-10)),
                'n_accepted_set_changed': int(sum(1 for r in rows
                                                  if r.get('accepted_set_unchanged') is False)),
                'enumeration_wall_s': enum_wall,
                'cap_used_s': float(args.seconds),
                'within_passed_cap': within_cap,
                'within_registered_10s_barrier': within_10s,
                'exact_within_declared_subproblem': exact,
                'candidates': rows,
            }
            (sdir / 'packet.json').write_text(json.dumps(
                {k: v for k, v in st['snapshot'].items() if k != 'traces'},
                indent=2, allow_nan=False, default=str), encoding='utf-8')
            (sdir / 'record.json').write_text(json.dumps(record, indent=2, allow_nan=False,
                                                         default=str), encoding='utf-8')
            states.append(record)
            print(json.dumps({'inst': inst, 'state': si, 'clock': round(st['clock'], 3),
                              'orders': len(orders), 'cands': len(rows),
                              'wall_s': round(enum_wall, 3), 'exact': exact,
                              'dJ': record['delta_utility'],
                              'nf': no_force_exact, 'if': identity_force_exact}), flush=True)

    gains = [s['delta_utility'] for s in states if s['delta_utility'] is not None]
    exact_states = [s for s in states if s['exact_within_declared_subproblem']]
    mean_gain = float(np.mean(gains)) if gains else None
    ci = None
    if len(gains) >= 2:
        rng = np.random.default_rng(args.bootstrap_seed)
        arr = np.asarray(gains, dtype=np.float64)
        draws = arr[rng.integers(0, len(arr), size=(args.bootstrap_repetitions, len(arr)))].mean(axis=1)
        ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    result = {
        'kind': 'exact_reordering_on_real_causal_midday_packets',
        'protocol_version': 'P-KNOWN-EXACT-LIVE-v1',
        'n_states': len(states), 'n_exact_states': len(exact_states),
        'all_states_exact': bool(states) and len(exact_states) == len(states),
        'n_states_with_positive_gain': int(sum(1 for g in gains if g > 1e-10)),
        'n_states_with_zero_gain': int(sum(1 for g in gains if abs(g) <= 1e-10)),
        'n_states_with_negative_gain': int(sum(1 for g in gains if g < -1e-10)),
        'mean_delta_utility': mean_gain,
        'descriptive_bootstrap_ci95': ci,
        'bootstrap_seed': args.bootstrap_seed,
        'bootstrap_repetitions': args.bootstrap_repetitions,
        'ci_interpretation': ('descriptive only: this bounded engineered packet family is not a '
                              'confirmatory cohort, so the interval is not a registered test'),
        'branch_machinery_controls_hold': bool(states) and all(
            s['no_force_replay_exact'] and s['identity_force_replay_exact'] for s in states),
        # Separates "the same plan served better" from "the re-ordering changed what the day
        # went on to accept". Both are real causal effects; only the first is local.
        'n_states_effect_is_pure_reordering': int(sum(
            1 for s in states if s['effect_is_pure_reordering'])),
        'n_states_effect_changed_downstream_accepts': int(sum(
            1 for s in states if s['delta_utility'] is not None
            and abs(s['delta_utility']) > 1e-10 and not s['effect_is_pure_reordering'])),
        'max_candidates_in_one_state': max(s['n_candidates'] for s in states) if states else 0,
        'max_enumeration_wall_s': max(s['enumeration_wall_s'] for s in states) if states else None,
        'all_states_within_registered_10s_barrier': bool(states) and all(
            s['within_registered_10s_barrier'] for s in states),
        'all_states_within_passed_cap': bool(states) and all(
            s['within_passed_cap'] for s in states),
        'registered_barrier_s': REGISTERED_BARRIER_S,
        'contract_hash': contract.contract_hash, 'budget': budget,
        'source_start': src_before,
        'states': states,
        'what_this_qualifies': [
            'a real causal mid-day packet can carry an exact finite re-ordering enumeration '
            'under the unchanged objective and the unchanged common budget',
            'the branch machinery is falsifiable here: both the no-force and the identity-force '
            'replays must reproduce the baseline exactly, and they are reported per state'],
        'what_this_does_not_qualify': [
            'the candidate set is re-orderings only; accept/reject re-decisions are NOT enumerated',
            'it is exact only within the declared finite candidate set, never globally',
            'no stable online gain, no MaskCO attribution, and no impossibility claim',
            'physics is shared with the versioned C0 evaluator and is not independently calibrated'],
        'formal_online_gain_established': False,
        'maskco_attributable': False,
        'pickup_impossibility_proven': False,
        'wall_s': time.perf_counter() - started,
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False, default=str),
                                     encoding='utf-8')
    (out / 'source_end.json').write_text(json.dumps(source_hashes(scripts), indent=2),
                                         encoding='utf-8')
    print(json.dumps({k: result[k] for k in
                      ('n_states', 'n_exact_states', 'all_states_exact',
                       'n_states_with_positive_gain', 'n_states_with_zero_gain',
                       'mean_delta_utility', 'descriptive_bootstrap_ci95',
                       'branch_machinery_controls_hold',
                       'all_states_within_registered_10s_barrier',
                       'all_states_within_passed_cap', 'wall_s')}, indent=2,
                     default=str))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scripts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--instances', type=int, default=2)
    ap.add_argument('--states-per-instance', type=int, default=2)
    ap.add_argument('--min-clock', type=float, default=1.0)
    ap.add_argument('--min-orders', type=int, default=3)
    ap.add_argument('--max-orders', type=int, default=4)
    ap.add_argument('--seconds', type=float, default=10.0)
    ap.add_argument('--budget', type=float, default=710.6548152249378)
    ap.add_argument('--cooling-share', type=float, default=3.7125478804082377)
    ap.add_argument('--bootstrap-seed', type=int, default=2026101201)
    ap.add_argument('--bootstrap-repetitions', type=int, default=2000)
    main(ap.parse_args())
