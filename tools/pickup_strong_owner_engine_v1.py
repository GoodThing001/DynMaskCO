"""Current-only independent rolling admission inside the original private owner.

Engineering-only variant of the already sealed pending transaction. The
proposal and K10 scoring stay identical; only admission is independently
selected by the original RollingOptimizeReplanner with a zero rejection loss.
"""
import copy
import time

import public_fleet as pf
from pickup_partial_validation_v1 import validate_partial
from pickup_pending_transaction_v1 import certificate


def decide(packet, history, scripts, deadline):
    from pickup_undecided_public_v1 import validate_undecided
    from strict_online_env import StrictOnlineEnv
    from recourse_snapshot import restore_recourse_snapshot, snapshot_state_hash
    from rolling_optimizer import RollingOptimizeReplanner
    import hard_admission as hd

    validate_undecided(packet)
    order = int(packet['admission_input']['current_order'])
    prior = packet['snapshot']['replanner_state']
    penalty = {0: 0., 1: 0., 2: 0.}
    if time.perf_counter() >= deadline:
        raise TimeoutError('independent rolling admission start')
    rp = RollingOptimizeReplanner(packet['budget'], packet['capacity'],
        packet['booking_horizon'], packet['contract'], 0., penalty,
        max(0., deadline-time.perf_counter()),
        hist=history['history_times_classes'], k=10)
    rp.restore_state(copy.deepcopy(prior))
    env = StrictOnlineEnv(packet['dataset'], packet['capacity'], packet['tw_speed'],
        packet['snapshot']['num_vehicles'], rp, packet['contract'],
        packet['booking_horizon'])
    env.replan_count = packet['snapshot']['replan_count']
    fleet, traces, served = restore_recourse_snapshot(packet['snapshot'])
    rp.on_reveal(env, 0, packet['snapshot']['clock'], fleet, served, [order])
    state = rp.export_state()
    accepted, rejected = set(state['accepted']), set(state['rejected'])
    if (accepted-{order} != set(prior['accepted']) or
        rejected-{order} != set(prior['rejected']) or
        (order in accepted) == (order in rejected) or accepted & rejected):
        raise ValueError('current-only independent admission transition')
    plan = hd.canonical(packet, state['plan'])
    state['plan'] = plan
    current = copy.deepcopy(packet)
    current['schema'] = 'pickup-public-partial-admissions-v1'
    current['admission_input'].update(
        pending=sorted(set(packet['admission_input']['pending'])-{order}),
        current_decision='accepted' if order in accepted else 'rejected')
    current['plan'] = plan
    current['ownership'] = {str(o): int(v) for v, route in plan.items() for o in route}
    current['snapshot']['replanner_state'] = state
    current['snapshot']['state_hash'] = snapshot_state_hash(current['snapshot'])
    current['penalties'] = penalty
    validate_partial(current)
    cert, raw = certificate(current, plan)
    return dict(qualified=cert['qualified'] and time.perf_counter()<deadline,
        state=state, plan=plan, certificate=cert, known_raw=raw,
        current_packet=current, independent_admission_class='RollingOptimizeReplanner',
        zero_reject_penalties=True)


def compute(packet, history, scripts, deadline):
    from pickup_partial_common_v1 import score_partial_common
    from joint_install import compile_state, digest, state_dict

    started = time.perf_counter()
    base = decide(packet, history, scripts, deadline)
    if not base['qualified']:
        raise ValueError('independent admitted incumbent physical failure')
    current = base['current_packet']
    proposal = copy.deepcopy(current['plan'])
    scores = score_partial_common(current,
        {'own_incumbent': current['plan'], 'candidate': proposal},
        history, deadline, seed=202610111301)
    own = scores['candidate_scores']['own_incumbent']
    cand = scores['candidate_scores']['candidate']
    change = cand['qualified'] and cand['mean_shadow_utility']>own['mean_shadow_utility']+1e-10
    selected = proposal if change else current['plan']
    cert, raw = certificate(current, selected)
    state = copy.deepcopy(base['state'])
    state['plan'] = selected
    compiled = compile_state(state)
    finished = time.perf_counter()
    return dict(kind='response', request_id=str(packet['admission_input']['current_order']),
        packet_digest=pf.digest(packet), deadline=deadline,
        certified=cert['qualified'] and finished<deadline,
        compiled_state=compiled, compiled_state_sha256=digest(state_dict(compiled)),
        base_response=base, scoring_packet=current, proposal=proposal,
        common_scores=scores, selected_certificate=cert, selected_raw=raw,
        accepted_route=change, worker_started=started, worker_finished=finished,
        actual_live_owner=False, formal_learned_controls=False,
        full_live_10s_qualified=False, independent_admission_class='RollingOptimizeReplanner')
