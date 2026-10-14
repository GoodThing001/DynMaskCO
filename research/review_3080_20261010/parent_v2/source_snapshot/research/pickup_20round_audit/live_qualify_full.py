"""P-LIVE-QUALIFY-FULL-v1: close the five registered engineering qualifications for the
live mid-day pickup prototype, in ONE artifact, each with a control that can fail.

The decision doc registers, before any new same-budget online panel:
  1. resume parity          - no-intervention continuation reproduces the original outcome
  2. whole-fleet energy     - the fleet energy ledger is complete for the WHOLE fleet
  3. commitment immutable   - the committed prefix cannot be modified
  4. future invariance      - changing unrevealed orders must not change the packet/choice
  5. full 10s submit path   - enumerate + certify + install + independent re-execution <= 10s

Items 1, 3 and 4 were previously checked in separate artifacts. Splitting them let two defects
survive: a sampler that took the first N decision points after min-clock (so every sample sat
at clock ~1.0 where only ONE vehicle is on the road) and a missing positive control on the
invariance check (a check that cannot fail proves nothing). This module re-runs all five with

  * a clock stride so samples really span the day,
  * an explicit dispatched-vehicle count so any "whole fleet" wording is scoped honestly,
  * a POSITIVE control for invariance (mutating a REVEALED order MUST change the packet),
  * a NEGATIVE control for commitment (a packet violating the committed/suffix split must be
    rejected by the validator), and
  * the submit path timed end-to-end, including installation and an independent re-execution.

Scope: engineering qualification of the prototype. No route is improved here, no arm is
compared, no stable online gain, no MaskCO attribution, no impossibility claim.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from itertools import permutations
from pathlib import Path
import sys
import time

import numpy as np

REGISTERED_BARRIER_S = 10.0


def bootstrap(scripts):
    scripts = Path(scripts).resolve()
    here = Path(__file__).resolve().parent
    for p in (here, scripts / 'evaluation', scripts / 'coldchain', scripts / 'simulation',
              scripts / 'correction_engineering'):
        sys.path.insert(0, str(p))
    return scripts


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def place(perm, template, slots):
    out = list(template)
    for slot, val in zip(slots, perm):
        out[slot] = val
    return out


def mutate_revealed(env, inst_idx, clock, served_mask):
    """POSITIVE control for invariance: alter orders that ARE revealed and unserved. A packet
    that claims to be causal must still change, otherwise the invariance check is inert."""
    clone = deepcopy(env)
    touched = 0
    for o in range(1, env.num_nodes):
        if bool(served_mask[o]):
            continue
        if float(env.reveal_time[inst_idx, o]) <= float(clock) + 1e-9:
            clone.demands[inst_idx, o] = float(env.demands[inst_idx, o]) + 7.0
            clone.coords[inst_idx, o, 0] = float(env.coords[inst_idx, o, 0]) + 123.0
            clone.coords[inst_idx, o, 1] = float(env.coords[inst_idx, o, 1]) - 45.0
            touched += 1
    return clone, touched


def main(args):
    started = time.perf_counter()
    scripts = bootstrap(args.scripts)
    from coldchain_contract import ColdChainContractV2
    from coldchain_evaluator_a1 import add_v2_initial_quality, evaluate_trace_a1
    from recourse_snapshot import (capture_recourse_snapshot, validate_snapshot_against_env)
    from run_exp_energy_c0 import (BOOKING_HORIZON, KM_PER_UNIT, SPEED_KMH, generate_dataset)
    from run_exp_rule_control import UncondDynamicReplanner
    from strict_online_env import StrictOnlineEnv
    import live_snapshot as LS

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    contract = ColdChainContractV2()
    contract.validate()
    budget = float(args.budget)
    p_c = {0: 5.0, 1: 10.0, 2: 15.0}
    fleet_size = int(args.num_vehicles)

    train_ds = generate_dataset(1, 200, 20260923)
    test_ds = generate_dataset(args.instances, 200, 20260924)
    test_ds = add_v2_initial_quality(test_ds, contract)
    hist = []
    for i in range(1):
        dem = train_ds['demands'][i]
        m = dem[1:] > 0
        hist.append((train_ds['reveal_time'][i][1:][m], train_ds['temp_class'][i][1:][m].astype(int)))

    def make_env():
        rp = UncondDynamicReplanner(budget=budget, capacity=50.0,
                                    booking_horizon=BOOKING_HORIZON, contract=contract,
                                    cooling_share=args.cooling_share, hist=hist, k=10)
        env = StrictOnlineEnv(test_ds, capacity=50.0, num_vehicles=fleet_size,
                              tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                              coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
        return env, rp

    def ev(traces, inst, rp):
        return evaluate_trace_a1(traces, {k: v[inst] for k, v in test_ds.items()}, contract,
                                 set(rp._accepted), set(rp._rejected), budget, p_c)

    records = []
    for inst in range(args.instances):
        env, rp = make_env()
        caps = []

        def hook(env_, inst_idx, clock, event_id, reveal_idx, vehicles, traces, served_mask,
                 all_customers):
            if clock < float(args.min_clock) or len(caps) >= args.captures_per_instance:
                return
            # "commitment immutable" can only be exercised where a committed prefix exists, and
            # it does not exist at every decision point. Collecting only on the clock stride
            # left it unexercised in 3 of 6 samples, which the aggregate then reported as a
            # FAILURE rather than as an untested item. Reserve slots for it explicitly.
            has_commit = any(getattr(v, 'committed_next', None) is not None for v in vehicles)
            n_commit = sum(1 for c in caps if c['has_commit'])
            take_for_commit = has_commit and n_commit < args.committed_captures
            if not take_for_commit:
                if caps and (float(clock) - caps[-1]['clock']) < float(args.clock_stride):
                    return
                if not has_commit and len(caps) + (args.committed_captures - n_commit) > \
                        args.captures_per_instance:
                    return
            caps.append(dict(clock=float(clock), event_id=int(event_id), has_commit=has_commit,
                             snapshot=capture_recourse_snapshot(
                                 env_, inst_idx, clock, event_id, reveal_idx, vehicles, traces,
                                 served_mask, all_customers),
                             vehicles=deepcopy(vehicles),
                             served=np.array(served_mask, copy=True),
                             accepted=set(rp._accepted)))

        env.snapshot_hook = hook
        base_traces, _ = env.run(inst)
        base_eval = ev(base_traces, inst, rp)
        base_accept = set(rp._accepted)
        base_u = float(base_eval['utility'])

        for ci, cap in enumerate(caps):
            rec = {'instance': inst, 'capture': ci, 'clock': cap['clock'],
                   'event_id': cap['event_id']}
            vehicles, served, clock = cap['vehicles'], cap['served'], cap['clock']

            # ---- 1. resume parity -------------------------------------------------
            try:
                renv, rrp = make_env()
                validate_snapshot_against_env(renv, cap['snapshot'])
                rtr, _ = renv.run_resumed(cap['snapshot'])
                re_ = ev(rtr, inst, rrp)
                rec['q1_resume_parity'] = bool(
                    abs(float(re_['utility']) - base_u) < 1e-9
                    and set(rrp._accepted) == base_accept)
                rec['q1_utility_delta'] = float(re_['utility']) - base_u
            except Exception as exc:                     # noqa: BLE001
                rec['q1_resume_parity'] = False
                rec['q1_failure'] = '%s: %s' % (type(exc).__name__, exc)
            # introspection outside the resume block
            statuses = list(cap['snapshot']['vehicle_status'])
            cstates = list(cap['snapshot']['vehicle_coldchain_state'])
            energies, missing = [], 0
            for s in cstates:
                if s is None:
                    continue
                v = (s.get('cumulative_energy_kwh') if isinstance(s, dict)
                     else getattr(s, 'cumulative_energy_kwh', None))
                if v is None:
                    missing += 1
                else:
                    energies.append(float(v))
            rec['snapshot_slots'] = len(statuses)
            rec['snapshot_non_idle'] = int(sum(1 for s in statuses if s != 'idle'))
            rec['snapshot_energy_entries'] = len(energies)
            rec['snapshot_energy_missing'] = missing

            # ---- 2. whole-fleet energy ledger -------------------------------------
            packet = LS.capture(env, inst, clock, vehicles, served, contract, budget,
                                cap['accepted'], args.depot_deadline)
            LS.validate_packet(packet)
            fleet = packet['fleet']
            dispatched = [f for f in fleet if abs(f['energy_used_kwh']) > 0
                          or f['status'] != 'idle']
            idle = [f for f in fleet if f not in dispatched]
            rec['packet_fleet_slots'] = len(fleet)
            rec['packet_dispatched'] = len(dispatched)
            rec['packet_idle'] = len(idle)
            rec['packet_energy_total_kwh'] = float(sum(f['energy_used_kwh'] for f in fleet))
            rec['q2_ledger_complete_all_slots_present'] = len(fleet) == fleet_size
            rec['q2_no_dispatched_vehicle_missing_energy'] = bool(
                all('energy_used_kwh' in f and np.isfinite(f['energy_used_kwh'])
                    for f in dispatched) and missing == 0
                and len(energies) == rec['snapshot_non_idle'])
            rec['q2_idle_vehicles_report_zero'] = bool(
                all(f['energy_used_kwh'] == 0.0 for f in idle))
            rec['q2_ledger_complete'] = bool(
                rec['q2_ledger_complete_all_slots_present']
                and rec['q2_no_dispatched_vehicle_missing_energy']
                and rec['q2_idle_vehicles_report_zero'])

            # ---- 3. commitment immutability ---------------------------------------
            committed = [f for f in fleet if f['committed_next'] is not None]
            rec['q3_committed_prefix_present'] = bool(committed)
            # negative control: the validator must REJECT a packet whose committed prefix has
            # been folded into the modifiable suffix.
            if committed:
                bad = deepcopy(packet)
                tgt = next(f for f in bad['fleet'] if f['committed_next'] is not None)
                tgt['mutable_suffix'] = list(tgt['mutable_suffix']) + [tgt['committed_next']]
                try:
                    LS.validate_packet(bad)
                    rec['q3_negative_control_rejected'] = False
                except ValueError:
                    rec['q3_negative_control_rejected'] = True
            else:
                rec['q3_negative_control_rejected'] = None
            rec['q3_no_committed_leak'] = bool(all(
                f['committed_next'] not in f['mutable_suffix'] for f in fleet))
            # Judged only on captures where a committed prefix exists; `None` means the item was
            # not exercisable there, which the aggregate treats as skipped, never as a pass.
            if not rec['q3_committed_prefix_present']:
                rec['q3_commitment_immutable'] = None
            else:
                rec['q3_commitment_immutable'] = bool(
                    rec['q3_no_committed_leak']
                    and rec['q3_negative_control_rejected'] is True)

            # ---- 4. future-mutation invariance + POSITIVE control -----------------
            clone, touched = LS.mutate_unrevealed(env, inst, clock)
            mut = LS.capture(clone, inst, clock, vehicles, served, contract, budget,
                             cap['accepted'], args.depot_deadline)
            rec['q4_unrevealed_touched'] = int(touched)
            rec['q4_packet_unchanged'] = digest(packet) == digest(mut)
            pclone, ptouched = mutate_revealed(env, inst, clock, served)
            pmut = LS.capture(pclone, inst, clock, vehicles, served, contract, budget,
                              cap['accepted'], args.depot_deadline)
            rec['q4_positive_control_revealed_touched'] = int(ptouched)
            rec['q4_positive_control_packet_changes'] = digest(packet) != digest(pmut)
            rec['q4_future_invariance'] = bool(
                rec['q4_unrevealed_touched'] > 0 and rec['q4_packet_unchanged']
                and rec['q4_positive_control_revealed_touched'] > 0
                and rec['q4_positive_control_packet_changes'])

            # ---- 5. full 10s submit path -----------------------------------------
            t0 = time.perf_counter()
            tgt = None
            for v in vehicles:
                tmpl = [int(x) for x in (v.mutable_suffix or [])]
                slots = [k for k, x in enumerate(tmpl)
                         if x != 0 and not served[x]
                         and float(env.reveal_time[inst, x]) <= clock + 1e-9]
                if args.min_orders <= len(slots) <= args.max_orders:
                    tgt = (v.vehicle_id, tmpl, slots, [tmpl[k] for k in slots])
                    break
            if tgt is None:
                rec['q5_submit_path'] = None
                rec['q5_skip_reason'] = 'no vehicle with %d..%d modifiable revealed orders' % (
                    args.min_orders, args.max_orders)
            else:
                vid, tmpl, slots, orders = tgt
                key = (cap['event_id'], vid)
                base_arms = []
                best = None
                for perm in permutations(orders):
                    seq = place(perm, tmpl, slots)
                    e2, r2 = make_env()
                    tr2, _ = e2.run(inst, force_suffix={key: seq})
                    e_ = ev(tr2, inst, r2)
                    base_arms.append(float(e_['utility']))
                    if e_['hard_feasible'] is True and (
                            best is None or float(e_['utility']) > best[1] + 1e-10):
                        best = (seq, float(e_['utility']))
                t_enum = time.perf_counter() - t0
                # independent re-execution of the SUBMITTED suffix, in a fresh env
                t1 = time.perf_counter()
                e3, r3 = make_env()
                tr3, _ = e3.run(inst, force_suffix={key: best[0]})
                e3v = ev(tr3, inst, r3)
                t_replay = time.perf_counter() - t1
                rec['q5_subproblem_orders'] = orders
                rec['q5_candidates'] = len(base_arms)
                rec['q5_enum_and_certify_s'] = t_enum
                rec['q5_independent_reexecution_s'] = t_replay
                rec['q5_submit_path_s'] = t_enum + t_replay
                rec['q5_submitted_utility'] = best[1]
                rec['q5_independent_reexecution_matches'] = bool(
                    abs(float(e3v['utility']) - best[1]) < 1e-9)
                rec['q5_within_registered_10s'] = bool(
                    rec['q5_submit_path_s'] <= REGISTERED_BARRIER_S
                    and len(base_arms) == math.factorial(len(orders)))

            records.append(rec)
            (out / ('capture_%02d_%02d.json' % (inst, ci))).write_text(
                json.dumps({k: v for k, v in rec.items()}, indent=2, allow_nan=False,
                           default=str), encoding='utf-8')
            print(json.dumps({'inst': inst, 'cap': ci, 'clock': round(clock, 3),
                              'nonidle': rec['snapshot_non_idle'],
                              'fleet_slots': rec['packet_fleet_slots'],
                              'q1': rec['q1_resume_parity'], 'q2': rec['q2_ledger_complete'],
                              'q3': rec['q3_commitment_immutable'],
                              'q4': rec['q4_future_invariance'],
                              'q5': rec.get('q5_within_registered_10s'),
                              'submit_s': (round(rec['q5_submit_path_s'], 3)
                                           if rec.get('q5_submit_path_s') else None)},
                             default=str), flush=True)

    def alltrue(field):
        vals = [r.get(field) for r in records if r.get(field) is not None]
        return bool(vals) and all(vals)

    q5 = [r for r in records if r.get('q5_submit_path_s') is not None]
    result = {
        'kind': 'live_midday_prototype_full_engineering_qualification',
        'protocol_version': 'P-LIVE-QUALIFY-FULL-v1',
        'n_captures': len(records),
        'clock_range': [min(r['clock'] for r in records), max(r['clock'] for r in records)],
        'max_non_idle_vehicles_observed': max(r['snapshot_non_idle'] for r in records),
        'fleet_size': fleet_size,
        'q1_resume_parity_holds': alltrue('q1_resume_parity'),
        'q2_whole_fleet_energy_ledger_complete': alltrue('q2_ledger_complete'),
        'q3_commitment_immutable': alltrue('q3_commitment_immutable'),
        'n_captures_exercising_commitment': int(sum(
            1 for r in records if r.get('q3_commitment_immutable') is not None)),
        'q4_future_mutation_invariance_holds': alltrue('q4_future_invariance'),
        'q5_submit_path_within_10s': bool(q5) and all(r['q5_within_registered_10s'] for r in q5),
        'q5_max_submit_path_s': (max(r['q5_submit_path_s'] for r in q5) if q5 else None),
        'q5_n_captures_measured': len(q5),
        'all_five_registered_items_pass': bool(
            alltrue('q1_resume_parity') and alltrue('q2_ledger_complete')
            and alltrue('q3_commitment_immutable') and alltrue('q4_future_invariance')
            and q5 and all(r['q5_within_registered_10s'] for r in q5)),
        'registered_barrier_s': REGISTERED_BARRIER_S,
        'records': records,
        'what_this_qualifies': [
            'the live prototype satisfies the five registered engineering preconditions for a '
            'same-budget online development panel, each with a control that can fail'],
        'what_this_does_not_qualify': [
            'no method is compared and no utility improvement is claimed on any day',
            'the energy ledger is complete for all %d fleet slots, but only %d vehicle(s) were '
            'ever dispatched in these instances, so the untouched remainder is not exercised'
            % (fleet_size, max(r['snapshot_non_idle'] for r in records)),
            'the submit path is qualified only for subproblems of size %d..%d orders'
            % (args.min_orders, args.max_orders),
            'physics is shared with the versioned C0 evaluator and is not independently calibrated',
            'no stable online gain, no MaskCO attribution, no impossibility claim'],
        'stable_online_gain_established': False,
        'maskco_attributable': False,
        'pickup_impossibility_proven': False,
        'wall_s': time.perf_counter() - started,
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False, default=str),
                                     encoding='utf-8')
    print(json.dumps({k: result[k] for k in
                      ('n_captures', 'clock_range', 'max_non_idle_vehicles_observed',
                       'q1_resume_parity_holds', 'q2_whole_fleet_energy_ledger_complete',
                       'q3_commitment_immutable', 'q4_future_mutation_invariance_holds',
                       'q5_submit_path_within_10s', 'q5_max_submit_path_s',
                       'all_five_registered_items_pass', 'wall_s')}, indent=2, default=str))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scripts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--instances', type=int, default=3)
    ap.add_argument('--captures-per-instance', type=int, default=3)
    ap.add_argument('--committed-captures', type=int, default=1,
                    help='reserved capture slots for decision points that actually HAVE a '
                         'committed prefix, so the immutability item is exercised not skipped')
    ap.add_argument('--min-clock', type=float, default=3.0)
    ap.add_argument('--clock-stride', type=float, default=4.0)
    ap.add_argument('--min-orders', type=int, default=3)
    ap.add_argument('--max-orders', type=int, default=3)
    ap.add_argument('--num-vehicles', type=int, default=15)
    ap.add_argument('--budget', type=float, default=710.6548152249378)
    ap.add_argument('--cooling-share', type=float, default=3.7125478804082377)
    ap.add_argument('--depot-deadline', type=float, default=22.0)
    main(ap.parse_args())
