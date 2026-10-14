"""One archived public pending request; independent admission component, no science day."""
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
import qualify_pickup_joint_policy_transaction_root_v3 as old
import numpy as np
from rolling_optimizer import RollingOptimizeReplanner
from run_exp_rule_control import UncondDynamicReplanner
from strict_online_env import StrictOnlineEnv
from recourse_snapshot import restore_recourse_snapshot, snapshot_state_hash
from pickup_undecided_public_v1 import validate_undecided
from pickup_partial_validation_v1 import validate_partial
from pickup_pending_transaction_v1 import certificate
import hard_admission as hd

HERE = Path(__file__).resolve().parent
INPUT = old.P / 'undecided_public_input_root_v2/RESULT.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(name, value):
    with (HERE / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)

def main():
    reg = json.loads((HERE / 'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
    assert sha(__file__) == reg['probe_sha256']
    assert sha(INPUT) == reg['input_sha256']
    assert reg['record_position'] == 6 and reg['penalties'] == [0.0, 0.0, 0.0]
    _, decode = old.codecs()
    row = json.loads(INPUT.read_text(encoding='utf-8'))['records'][6]
    assert row['kind'] == 'actual_first_reveal_hook' and row['variant'] == 0
    packet = decode(row['packet'])
    validate_undecided(packet)
    before = old.pf.digest(packet)
    order = packet['admission_input']['current_order']
    snap = packet['snapshot']
    penalties = {0: 0., 1: 0., 2: 0.}
    rp = RollingOptimizeReplanner(packet['budget'], packet['capacity'], packet['booking_horizon'], packet['contract'], 0., penalties, 8.)
    rp.restore_state(copy.deepcopy(snap['replanner_state']))
    env = StrictOnlineEnv(packet['dataset'], packet['capacity'], packet['tw_speed'], snap['num_vehicles'], rp, packet['contract'], packet['booking_horizon'])
    env.replan_count = snap['replan_count']
    fleet, traces, served = restore_recourse_snapshot(snap)
    original_state = copy.deepcopy(rp.export_state())
    t0 = time.perf_counter()
    rp.on_reveal(env, 0, snap['clock'], fleet, served, [order])
    elapsed = time.perf_counter() - t0
    state = rp.export_state()
    accepted = set(state['accepted'])
    rejected = set(state['rejected'])
    if (accepted - {order} != set(original_state['accepted']) or
        rejected - {order} != set(original_state['rejected']) or
        (order in accepted) == (order in rejected)):
        raise ValueError('not a current-only independent admission transition')
    plan = hd.canonical(packet, state['plan'])
    state['plan'] = plan
    current = copy.deepcopy(packet)
    current['schema'] = 'pickup-public-partial-admissions-v1'
    current['admission_input'].update(pending=sorted(set(packet['admission_input']['pending']) - {order}),
        current_decision='accepted' if order in accepted else 'rejected')
    current['plan'] = plan
    current['ownership'] = {str(o): int(v) for v, route in plan.items() for o in route}
    current['snapshot']['replanner_state'] = state
    current['snapshot']['state_hash'] = snapshot_state_hash(current['snapshot'])
    current['penalties'] = penalties
    validate_partial(current)
    cert, raw = certificate(current, plan)
    assert old.pf.digest(packet) == before
    assert RollingOptimizeReplanner.on_reveal is not UncondDynamicReplanner.on_reveal
    encode, _ = old.codecs()
    write('RESULT.json', dict(record_position=6, original_input_sha256=sha(INPUT), input_digest=before,
        order=order, clock=float(snap['clock']), elapsed_seconds=elapsed,
        current_decision=current['admission_input']['current_decision'],
        previous_accepted=sorted(original_state['accepted']), previous_rejected=sorted(original_state['rejected']),
        accepted=sorted(accepted), rejected=sorted(rejected), pending=current['admission_input']['pending'],
        plan=plan, certificate=encode(cert), raw_digest=old.pf.digest(raw),
        input_unchanged=old.pf.digest(packet) == before,
        independent_admission_method=True, same_objective_penalties_zero=True,
        initial_observed_public_packet_only=True, actual_live_owner=False,
        complete_request_budget_qualified=False, formal_strong_dynamic_qualified=False,
        scientific_data_generated=False, stable_online_gain=False))

if __name__ == '__main__':
    main()
