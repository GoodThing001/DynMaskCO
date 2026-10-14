"""Live mid-day causal pickup snapshot: capture a public packet from a real non-zero-clock
state, and prove by future-mutation invariance that it carries no hidden-future information.

Why this exists. The registered exact pickup work is deliberately restricted to an initial
idle depot at clock 0, and its protocol names the gap: actual middle-day headroom requires a
causal live snapshot that preserves the current cargo and thermal state, the committed
prefix, every other vehicle's consumed energy and the genuinely modifiable suffix - with no
fabricated residual budget and no private future feed. This module implements the CAPTURE
and the INVARIANCE PROOF half of that gap. It does not implement the exact solve over live
packets, and it claims no utility improvement.

What the invariance proof is. A public packet is a function of the state a policy can
legitimately see. If it is causal, then mutating anything the policy cannot see - the
quantity, coordinates, class, time windows, service time or identity of orders that are not
yet revealed - must leave the packet byte-identical. That is checked directly by rebuilding
the packet from a mutated environment and comparing serialised forms, rather than argued.

Scope: engineering qualification on a real StrictOnlineEnv state, using the unmodified
A-v1 contract and the unmodified reference replanner. Non-learning. No gain is claimed and
no exact upper bound is established here.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / 'C-VRP_Cold-chainVehicleRoutingProblem' / 'scripts'


def bootstrap(scripts):
    scripts = Path(scripts).resolve()
    for sub in ('evaluation', 'coldchain', 'simulation'):
        sys.path.insert(0, str(scripts / sub))
    return scripts


PACKET_FIELDS = ('clock', 'depot', 'depot_deadline', 'orders', 'accepted', 'fleet', 'budget')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def capture(env, inst_idx, clock, vehicles, served_mask, contract, budget, accepted,
            depot_deadline):
    """Export a causal public packet from a real mid-day state.

    Orders that are not yet revealed are omitted entirely, not masked, so the packet cannot
    leak their count, identity, geometry or readiness. depot_deadline is taken as a public
    contract parameter rather than derived from the order arrays: deriving it as
    max(tw_end) over all orders was the first version of this function and the
    future-mutation check caught it leaking unrevealed orders into the packet.
    """
    nodes = env.num_nodes
    orders = []
    for o in range(1, nodes):
        if bool(served_mask[o]):
            continue
        reveal = float(env.reveal_time[inst_idx, o])
        if reveal > float(clock) + 1e-9:
            continue                      # unrevealed: excluded by construction
        orders.append({'id': int(o), 'reveal': reveal,
                       'x': float(env.coords[inst_idx, o, 0]),
                       'y': float(env.coords[inst_idx, o, 1]),
                       'demand': float(env.demands[inst_idx, o]),
                       'temp_class': int(env.temp_class[inst_idx, o]),
                       'tw_start': float(env.tw_start[inst_idx, o]),
                       'tw_end': float(env.tw_end[inst_idx, o]),
                       'service_time': float(env.service_time[inst_idx, o])})
    fleet = []
    for v in vehicles:
        st = getattr(v, 'coldchain_state', None)
        cnext = getattr(v, 'committed_next', None)
        fleet.append({'vehicle_id': int(v.vehicle_id), 'status': str(v.status),
                      'current_node': int(v.current_node),
                      'current_load': float(v.current_load),
                      'committed_next': (None if cnext in (None,) else int(cnext)),
                      'mutable_suffix': [int(x) for x in v.mutable_suffix if int(x) != 0],
                      'energy_used_kwh': (float(st.cumulative_energy_kwh)
                                          if st is not None else 0.0)})
    return {'clock': float(clock),
            'depot': [float(env.coords[inst_idx, 0, 0]), float(env.coords[inst_idx, 0, 1])],
            'depot_deadline': float(depot_deadline),
            'orders': orders,
            'accepted': sorted(int(a) for a in accepted),
            'fleet': fleet,
            'budget': float(budget)}


def validate_packet(packet):
    """Reject any undeclared field, any unrevealed order, and any non-finite numeric."""
    if set(packet) != set(PACKET_FIELDS):
        raise ValueError('undeclared public field in live packet')
    if not np.isfinite(packet['clock']) or packet['clock'] < 0:
        raise ValueError('invalid clock')
    for o in packet['orders']:
        if set(o) != {'id', 'reveal', 'x', 'y', 'demand', 'temp_class', 'tw_start',
                      'tw_end', 'service_time'}:
            raise ValueError('undeclared order field')
        if o['reveal'] > packet['clock'] + 1e-9:
            raise ValueError('future order present in packet')
        if any(not np.isfinite(float(o[k])) for k in
               ('reveal', 'x', 'y', 'demand', 'tw_start', 'tw_end', 'service_time')):
            raise ValueError('non-finite order numeric')
    for f in packet['fleet']:
        if f['committed_next'] is not None and f['committed_next'] in f['mutable_suffix']:
            raise ValueError('committed prefix leaked into the modifiable suffix')
        if any(not np.isfinite(float(f[k])) for k in ('current_load', 'energy_used_kwh')):
            raise ValueError('non-finite fleet numeric')
    return True


def mutate_unrevealed(env, inst_idx, clock):
    """Return a copy of the environment with every UNREVEALED order altered beyond
    recognition. A causal packet must be identical against it."""
    clone = deepcopy(env)
    nodes = env.num_nodes
    touched = 0
    for o in range(1, nodes):
        if float(env.reveal_time[inst_idx, o]) > float(clock) + 1e-9:
            clone.demands[inst_idx, o] = float(env.demands[inst_idx, o]) + 7.0
            clone.coords[inst_idx, o, 0] = float(env.coords[inst_idx, o, 0]) + 123.0
            clone.coords[inst_idx, o, 1] = float(env.coords[inst_idx, o, 1]) - 45.0
            clone.temp_class[inst_idx, o] = (int(env.temp_class[inst_idx, o]) + 1) % 3
            clone.tw_end[inst_idx, o] = float(env.tw_end[inst_idx, o]) + 3.0
            clone.service_time[inst_idx, o] = float(env.service_time[inst_idx, o]) + 0.02
            touched += 1
    return clone, touched


def main(args):
    started = time.perf_counter()
    bootstrap(args.scripts)
    from coldchain_contract import ColdChainContractV2
    from coldchain_evaluator_a1 import add_v2_initial_quality
    from run_exp_energy_c0 import BOOKING_HORIZON, KM_PER_UNIT, SPEED_KMH, generate_dataset
    from run_exp_rule_control import UncondDynamicReplanner
    from strict_online_env import StrictOnlineEnv

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    contract = ColdChainContractV2()
    contract.validate()

    budget = float(args.budget)
    train_ds = generate_dataset(1, 200, 20260923)
    test_ds = generate_dataset(args.instances, 200, 20260924)
    test_ds = add_v2_initial_quality(test_ds, contract)
    hist = []
    for i in range(1):
        dem = train_ds['demands'][i]
        mask = dem[1:] > 0
        hist.append((train_ds['reveal_time'][i][1:][mask],
                     train_ds['temp_class'][i][1:][mask].astype(int)))

    replanner = UncondDynamicReplanner(
        budget=budget, capacity=50.0, booking_horizon=BOOKING_HORIZON, contract=contract,
        cooling_share=args.cooling_share, hist=hist, k=10)
    env = StrictOnlineEnv(test_ds, capacity=50.0, num_vehicles=15,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=replanner,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)

    captured = []

    def hook(env_, inst_idx, clock, event_id, reveal_idx, vehicles, traces, served_mask,
             all_customers):
        if clock <= 0 or len(captured) >= args.captures:
            return
        if any(c['clock'] == clock for c in captured):
            return
        accepted = set(getattr(replanner, '_accepted', set()))
        packet = capture(env_, inst_idx, clock, vehicles, served_mask, contract, budget,
                         accepted, args.depot_deadline)
        validate_packet(packet)
        clone, touched = mutate_unrevealed(env_, inst_idx, clock)
        mutated = capture(clone, inst_idx, clock, vehicles, served_mask, contract, budget,
                          accepted, args.depot_deadline)
        captured.append({'clock': float(clock), 'inst_idx': int(inst_idx),
                         'unrevealed_orders_mutated': int(touched),
                         'packet_sha256': digest(packet),
                         'mutated_packet_sha256': digest(mutated),
                         'future_mutation_invariant': digest(packet) == digest(mutated),
                         'n_visible_orders': len(packet['orders']),
                         'n_vehicles_with_cargo': sum(
                             1 for f in packet['fleet'] if f['current_load'] > 0),
                         'n_committed_prefixes': sum(
                             1 for f in packet['fleet'] if f['committed_next'] is not None),
                         'total_energy_used_kwh': float(sum(
                             f['energy_used_kwh'] for f in packet['fleet'])),
                         'packet': packet if args.keep_packets else None})
        (out / ('packet_%02d.json' % len(captured))).write_text(
            json.dumps(packet, indent=2, allow_nan=False), encoding='utf-8')

    env.snapshot_hook = hook
    for i in range(args.instances):
        env.run(i)

    ok = [c for c in captured if c['future_mutation_invariant']]
    result = {
        'kind': 'live_midday_causal_snapshot_capture_and_invariance',
        'n_captures': len(captured), 'n_invariant': len(ok),
        'all_captures_future_mutation_invariant': bool(captured) and len(ok) == len(captured),
        'captures': [{k: v for k, v in c.items() if k != 'packet'} for c in captured],
        'mechanism_present': {
            'non_zero_clock': all(c['clock'] > 0 for c in captured),
            'cargo_in_transit_observed': any(c['n_vehicles_with_cargo'] > 0 for c in captured),
            'committed_prefix_observed': any(c['n_committed_prefixes'] > 0
                                             for c in captured),
            'energy_already_consumed_observed': any(c['total_energy_used_kwh'] > 0
                                                    for c in captured)},
        'what_this_does_not_do': [
            'it does not solve the live subproblem or claim any utility improvement',
            'it does not establish an exact or certified upper bound anywhere',
            'it does not fabricate a residual budget and does not read any unrevealed order',
            'it uses the shared C0 evaluator, so its physics is not independently calibrated'],
        'online_gain_established': False,
        'stable_gain_established': False,
        'maskco_attributable': False,
        'wall_s': time.perf_counter() - started,
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False),
                                     encoding='utf-8')
    print(json.dumps({k: result[k] for k in
                      ('n_captures', 'n_invariant',
                       'all_captures_future_mutation_invariant', 'mechanism_present',
                       'wall_s')}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scripts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--instances', type=int, default=2)
    ap.add_argument('--captures', type=int, default=6)
    ap.add_argument('--budget', type=float, default=710.6548152249378)
    ap.add_argument('--cooling-share', type=float, default=3.7125478804082377)
    ap.add_argument('--depot-deadline', type=float, default=22.0,
                    help='public contract parameter, never derived from the order arrays')
    ap.add_argument('--keep-packets', action='store_true')
    main(ap.parse_args())
