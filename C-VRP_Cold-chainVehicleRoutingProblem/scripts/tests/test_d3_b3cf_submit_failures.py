#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B3-CF: submit failure-path counterexample verification (READ-ONLY).

Pre-registered in `B3CF_COUNTEREXAMPLE_SPEC.json` BEFORE this file was written.  Every
counterexample is constructed and executed ONLY inside an isolated copy of the real
StrictOnlineEnv, and the ACTUAL before/after state difference is recorded.

Reporting rule (pre-registered): CP-F10 is EXPECTED to reproduce the bypass.  A reproduced defect is
recorded as COUNTEREXAMPLE_REPRODUCED and must NOT be reported as "G-L passes".

Nothing here modifies `_commit` / `_plan_and_commit`, wires real D3 write-back, relaxes D8, or runs a
scientific effect experiment.

Run: python scripts/tests/test_d3_b3cf_submit_failures.py [--instances 0] [--heavy-cap 6]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent / 'simulation'
for _p in (SIM, HERE.parent, HERE, HERE.parent / 'coldchain', HERE.parent / 'evaluation'):
    sys.path.insert(0, str(_p))

import numpy as np                                                     # noqa: E402
import d3_state_assembly as A                                          # noqa: E402
import d3_credential as CR                                             # noqa: E402
import d3_install_gate as GATE                                         # noqa: E402
from strict_online_env import StrictOnlineEnv, VehicleState            # noqa: E402
from jf1h_repair import make_continuation                              # noqa: E402
from coldchain_contract import (default_pilot_contract,                # noqa: E402
                                apply_objective_profile, ObjectiveProfile)
from coldchain_state import transition_segment                          # noqa: E402

EXT = HERE.parent.parent
CAL_NPZ = EXT / 'data' / 'm0_scale' / 'dcc_50_r1_edod05_cal_teacher.npz'
PROFILE_JSON = EXT / 'results' / 'o0cc' / 'scale_v2' / 'objective_profile.json'
CAPACITY, NUM_VEHICLES = 50.0, 25
PREREGISTERED = ['CP-F10', 'CP-F8', 'CP-F9', 'CP-A-c7', 'CP-A-c8', 'CP-A-c5']
RESULTS = []


def chk(cid, name, verdict, detail):
    RESULTS.append({'cid': cid, 'name': name, 'verdict': verdict, 'detail': detail})


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_profile():
    d = json.loads(PROFILE_JSON.read_text(encoding='utf-8'))
    p = ObjectiveProfile(name=d['name'], distance_scale=float(d['distance_scale']),
                         quality_scale=float(d['quality_scale']),
                         energy_scale=float(d['energy_scale']),
                         lambda_quality=float(d['lambda_quality']),
                         lambda_energy=float(d['lambda_energy']),
                         scale_source=d.get('scale_source', 'pilot'),
                         dev_statistics=d.get('dev_statistics'))
    assert p.profile_hash == d.get('profile_hash')
    return p


def make_env(dataset, contract, hook=None):
    env = StrictOnlineEnv(dataset, capacity=CAPACITY, tw_speed=1.0, num_vehicles=NUM_VEHICLES,
                          replanner=make_continuation(), coldchain_contract=contract,
                          booking_horizon=None)
    env.snapshot_hook = hook
    return env


class Grab(object):
    def __init__(self, cap=6):
        self.cap, self.items, self.n_calls, self.visible = cap, [], 0, {}

    def hook(self, env, inst_idx, clock, event_id, reveal_idx, vehicles, traces, served_mask,
             all_customers=None):
        self.n_calls += 1
        if len(self.items) >= self.cap:
            return
        snap, h = A.capture_decision_state(env, inst_idx, clock, event_id, reveal_idx, vehicles,
                                          traces, served_mask, all_customers)
        cv, ct, csm = A.clone_state(snap)
        vis = [int(i) for i in all_customers if env.reveal_time[inst_idx, i] <= clock + 1e-6]
        cv, plans, _r = A.prepare_clone(env, inst_idx, clock, cv, csm, vis)
        self.items.append({'env': env, 'inst': int(inst_idx), 'clock': float(clock),
                           'event': int(event_id), 'vehicles': cv, 'traces': ct, 'served_mask': csm,
                           'plans': plans, 'snapshot': snap, 'state_hash': h, 'visible': vis,
                           'state_after': A.live_state_fingerprint(
                               env, inst_idx, clock, event_id, reveal_idx, vehicles, traces,
                               served_mask, all_customers)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instances', type=int, nargs='+', default=[0])
    ap.add_argument('--heavy-cap', type=int, default=6)
    args = ap.parse_args()

    profile = load_profile()
    dataset = dict(np.load(CAL_NPZ))
    raw = default_pilot_contract()
    eff = apply_objective_profile(raw, profile)
    inst = args.instances[0]

    g = Grab(cap=args.heavy_cap)
    env = make_env(dataset, eff, hook=g.hook)
    traces, served = env.run(inst)
    items = g.items
    base = items[0]

    def pick(pred):
        """Select the decision point each counterexample actually NEEDS.

        Fixture-selection defect found on the first run: using items[0] (clock 0) made three cases
        NOT_VERIFIABLE for the wrong reason -- at clock 0 nothing is served, no vehicle carries
        cargo and no vehicle is 'ready' at a customer.  Each case now selects a decision point that
        really exhibits the precondition, and falls back to NOT_VERIFIABLE only if the captured
        window contains none.
        """
        return next((it for it in items if pred(it)), None)

    chk('PRE', 'isolated copies captured from a real run, and the live state was not written',
        'SETUP_OK' if items and all(it['state_after'] for it in items) else 'SETUP_FAILED',
        {'n_items': len(items), 'n_calls': g.n_calls,
         'first_clock': items[0]['clock'] if items else None,
         'last_clock': items[-1]['clock'] if items else None})

    # ================= CP-F10: bypass -- an illegal suffix is executed with NO gate =========
    it10 = pick(lambda it: any(bool(it['served_mask'][c]) for c in range(1, env.num_nodes)))
    cv, ct, csm = A.clone_state((it10 or base)['snapshot'])
    served_nodes = {int(sr.node) for tr in ct for sr in tr.services}
    served_in_mask = {c for c in range(1, env.num_nodes) if bool(csm[c])}
    illegal = sorted(served_nodes | served_in_mask)
    f10 = {'certifier_called': False, 'decision_point': ((it10 or base)['clock'],
                                                         (it10 or base)['event'])}
    if illegal:
        victim = next(v for v in cv if v.status in ('idle', 'ready'))
        # a replanner would write exactly this; needs_replan=False so the replanner is NOT invoked
        for v in cv:
            v.needs_replan = False
        victim.mutable_suffix = [int(illegal[0])]
        before = {'status': victim.status, 'committed_next': victim.committed_next,
                  'suffix': list(victim.mutable_suffix)}
        all_customers = [int(c) for c in range(1, env.num_nodes) if bool(csm[c])] + illegal
        env._plan_and_commit(inst, (it10 or base)['clock'], cv, ct, csm,
                             sorted(set(all_customers)))            # the env's OWN commit loop
        after = {'status': victim.status,
                 'committed_next': (None if victim.committed_next is None
                                    else int(victim.committed_next)),
                 'suffix': list(victim.mutable_suffix)}
        wrote = (after['committed_next'] == int(illegal[0]))
        f10 = {'served_customer_used': int(illegal[0]), 'vehicles_needs_replan_cleared': True,
               'before': before, 'after': after, 'state_changed': before != after,
               'illegal_head_committed': bool(wrote),
               'composed_gate_invoked': False, 'certifier_called': f10['certifier_called'],
               'note': ('the env committed a suffix containing an already-SERVED customer with no '
                        'certification anywhere in the path; the before/after diff is the actual '
                        'state change, not an inference')}
        chk('CP-F10', 'CP-F10 the real commit loop executes an illegal suffix with no gate',
            'COUNTEREXAMPLE_REPRODUCED' if wrote else 'NOT_REPRODUCED', f10)
    else:
        chk('CP-F10', 'CP-F10 the real commit loop executes an illegal suffix with no gate',
            'NOT_VERIFIABLE', {'why': 'no served customer available at this decision point',
                               'n_served': len(served_in_mask)})

    # ================= CP-F8: TOCTOU between approval and the first real write ==============
    cv8, ct8, csm8 = A.clone_state(base['snapshot'])
    cv8b, ct8b, csm8b = A.clone_state(base['snapshot'])
    vis = base['visible']
    cv8, plans8, _r = A.prepare_clone(env, inst, base['clock'], cv8, csm8, vis)
    snap8, h8 = A.capture_decision_state(env, inst, base['clock'], base['event'], 0, cv8, ct8, csm8)
    cred_s1 = CR.make_credential(env, inst, base['clock'], base['event'], snap8, h8, plans8,
                                {'schema': 'cp'}, raw_contract=raw, profile=profile)
    allowed = GATE.pre_install_reverify(cred_s1, cred_s1, deadline_ok=True,
                                        certification_complete=True)
    # change the state on the copy AFTER approval: clear a mutable tail and advance the clock
    victim8 = next(v for v in cv8 if v.mutable_suffix)
    tail_before = list(victim8.mutable_suffix)
    if victim8.mutable_suffix:
        victim8.mutable_suffix = []                                   # exactly what a reveal does
    cv8, plans8b, _r2 = A.prepare_clone(env, inst, base['clock'] + 1.0, cv8, csm8, vis)
    snap8b, h8b = A.capture_decision_state(env, inst, base['clock'] + 1.0, base['event'] + 1, 0,
                                           cv8, ct8, csm8)
    state_changed = (h8 != h8b)
    # the S1-approved plan, written by the real primitive, with NO re-verification
    target8 = next(v for v in cv8 if v.status in ('idle', 'ready'))
    head = int(plans8[target8.vehicle_id].suffix[0]) if plans8.get(target8.vehicle_id) and \
        plans8[target8.vehicle_id].suffix else None
    wrote8 = None
    if head is not None:
        b8 = {'status': target8.status, 'committed_next': target8.committed_next}
        env._commit(inst, base['clock'] + 1.0, target8, ct8, head)
        wrote8 = {'status': target8.status,
                  'committed_next': (None if target8.committed_next is None
                                     else int(target8.committed_next))}
    chk('CP-F8', 'CP-F8 the approval and the first real write are separated by an uncontrolled '
        'state change',
        'COUNTEREXAMPLE_REPRODUCED' if (allowed['allow'] and state_changed and wrote8
                                        and wrote8['committed_next'] == head)
        else 'NOT_REPRODUCED',
        {'approved_at_S1': allowed['allow'], 'S1_state_hash': h8[:12], 'S2_state_hash': h8b[:12],
         'state_changed_after_approval': state_changed,
         'tail_cleared_on_the_copy': {'before': tail_before, 'after': []},
         'write_after_state_change_without_reverification': wrote8,
         'head_written': head,
         'note': ('re-comparing three hashes does NOT make the commit atomic: nothing binds the '
                  're-check to the write, and no other writer is excluded')})

    # ================= CP-F9: does a real reveal/replan invalidate the OLD credential =======
    reveal_transitions, refusals = [], []
    prev = None
    for it in items:
        cur = {'clock': it['clock'], 'state_hash': it['state_hash'],
               'plan_hash': CR.plan_digest_anchor_v1(it['plans']),
               'n_visible': len(it['visible'])}
        if prev is not None and cur['n_visible'] > prev['n_visible']:
            reveal_transitions.append({'from_clock': prev['clock'], 'to_clock': cur['clock'],
                                      'n_visible': [prev['n_visible'], cur['n_visible']],
                                      'state_hash_changed': prev['state_hash'] != cur['state_hash'],
                                      'plan_digest_changed': prev['plan_hash'] != cur['plan_hash']})
        prev = cur
    if reveal_transitions:
        snapA, hA = A.capture_decision_state(env, inst, items[0]['clock'], items[0]['event'], 0,
                                             items[0]['vehicles'], items[0]['traces'],
                                             items[0]['served_mask'])
        snapB, hB = A.capture_decision_state(env, inst, items[-1]['clock'], items[-1]['event'], 0,
                                             items[-1]['vehicles'], items[-1]['traces'],
                                             items[-1]['served_mask'])
        cA = CR.make_credential(env, inst, items[0]['clock'], items[0]['event'], snapA, hA,
                                items[0]['plans'], {'schema': 'cp'}, raw_contract=raw,
                                profile=profile)
        cB = CR.make_credential(env, inst, items[-1]['clock'], items[-1]['event'], snapB, hB,
                                items[-1]['plans'], {'schema': 'cp'}, raw_contract=raw,
                                profile=profile)
        res = GATE.pre_install_reverify(cA, cB, deadline_ok=True, certification_complete=True)
        refusals = res['refusals']
        chk('CP-F9', 'CP-F9 a later real reveal invalidates the old credential (mechanism)',
            'MECHANISM_SATISFIED_ON_REAL_PATH'
            if (reveal_transitions and res['allow'] is False
                and 'state_hash_mismatch' in res['refusals'])
            else 'NOT_REPRODUCED',
            {'n_reveal_transitions': len(reveal_transitions), 'examples': reveal_transitions[:3],
             'reverify_allow': res['allow'], 'refusals': refusals,
             'scope_caveat': ('the mechanism refuses, but NO production path calls it, so this is '
                              'NOT evidence that continuous safety (G-C) holds')})
    else:
        chk('CP-F9', 'CP-F9 a later real reveal invalidates the old credential (mechanism)',
            'NOT_VERIFIABLE', {'why': 'no reveal transition captured in the first N decision points'})

    # ================= CP-A-c7: _validate_state does not raise on a thermal excursion =======
    import coldchain_state as CS
    it7 = pick(lambda it: any(v.coldchain_state is not None for v in it['vehicles']))
    cc = None
    for v in (it7 or base)['vehicles']:
        if v.coldchain_state is not None:
            cc = v.coldchain_state
            break
    a7 = {'excursion_built': False, 'decision_point': ((it7 or base)['clock'],
                                                       (it7 or base)['event'])}
    if cc is not None:
        hi = [b[1] for b in eff.thermal.hard_temperature_bounds_c]
        hot = list(cc.compartment_temperature_c)
        hot[0] = hi[0] + 5.0
        bad = CS.VehicleColdChainState(**{**cc.__dict__, 'compartment_temperature_c': tuple(hot)})
        raised = None
        try:
            CS._validate_state(bad, eff)
        except BaseException as exc:                                  # noqa: BLE001
            raised = '%s: %s' % (type(exc).__name__, str(exc)[:80])
        cnt, dur = CS._thermal_violations(cc.compartment_temperature_c, tuple(hot), 1.0, eff)
        a7 = {'excursion_temperature_c': hot[0], 'hard_upper_c': hi[0],
              'validate_state_raised': raised, 'violations_counted': int(cnt),
              'excursion_built': True}
    chk('CP-A-c7', 'CP-A-c7 _validate_state does not raise on a thermal excursion; it is only '
        'counted', 'COUNTEREXAMPLE_REPRODUCED' if (a7['excursion_built']
        and a7['validate_state_raised'] is None and a7['violations_counted'] > 0)
        else 'NOT_VERIFIABLE', a7)

    # ================= CP-A-c8: can the zone capacity floor bind ============================
    zones = eff.thermal.supported_temp_classes
    rows8 = []
    for z in zones:
        z = int(z)
        cap_c = float(eff.thermal.thermal_capacity_kwh_per_c[z])
        hi = float(eff.thermal.hard_temperature_bounds_c[z][1])
        tgt = float(eff.thermal.target_temperature_c[z])
        # the door event warms by door_heat/cap_c per pickup; the number of pickups per vehicle is
        # bounded by the vehicle's eventual load / smallest order quantity -> use the largest
        # single-order door heat as the per-event step and report the slack against hard_upper
        step = float(eff.thermal.door_heat_kwh[z]) / cap_c
        max_events = int((hi - tgt) // step) if step > 0 else None
        rows8.append({'zone': z, 'target': tgt, 'hard_upper': hi, 'door_step_c': step,
                      'max_door_events_before_hard_upper': max_events,
                      'binds_free': bool(max_events is not None and max_events < 1)})
    chk('CP-A-c8', 'CP-A-c8 zone capacity/door-warming floor analysis: can hard_upper be reached',
        'CONCLUSION_RECORDED',
        {'per_zone': rows8,
         'note': ('max_door_events is the number of pickups a single vehicle could absorb before '
                  'the door warming alone would push its zone to hard_upper; it bounds whether the '
                  'capacity floor can bind at all')})

    # ================= CP-A-c5: force _force_return_all and measure the time gap ============
    it5 = pick(lambda it: any(v.status == 'ready' and v.current_node != 0
                              and v.coldchain_state is not None for v in it['vehicles']))
    cv5, ct5, csm5 = A.clone_state((it5 or base)['snapshot'])
    target5 = None
    for v in cv5:
        if v.status == 'ready' and v.current_node != 0 and v.coldchain_state is not None:
            target5 = v
            break
    a5 = {'forced': False, 'decision_point': ((it5 or base)['clock'], (it5 or base)['event'])}
    if target5 is not None:
        cc0 = target5.coldchain_state
        node_before = int(target5.current_node)
        n_before = len(cc0.delivered_to_depot)
        # `coldchain_updated_time` lives on VehicleState, NOT on VehicleColdChainState (bug found
        # by the crash on the first run of this block)
        cc_before_t = float(getattr(target5, 'coldchain_updated_time', None) or 0.0)
        travel_h = float(env._travel(inst, node_before, 0))
        mask = env._coldchain_active_mask()
        env._force_return_all(inst, (it5 or base)['clock'], cv5, ct5, csm5)
        cc1 = target5.coldchain_state
        recs = cc1.delivered_to_depot
        new = list(recs[n_before:]) if len(recs) > n_before else []
        cc_after_t = float(getattr(target5, 'coldchain_updated_time', None) or 0.0)
        rec_times = [float(r.return_arrival_time) for r in new]
        arrival = max(rec_times) if rec_times else None
        depart = (arrival - travel_h) if arrival is not None else None

        # DECISIVE TEST.  The registered suspicion is "the recorded unload time stops at `depart`
        # while the return travel is not elapsed, so the recorded quality is optimistic".  Make the
        # two candidate physics chains explicit as independent read-only projections of the SAME
        # pre-call state `cc0` (pure function; `cc0` is an immutable frozen dataclass):
        #   chain_A = advance to `depart` only      (what the defect predicts)
        #   chain_B = advance to `arrival`          (what correct physics predicts)
        # then compare the ACTUALLY recorded `delivered_quality` against both.
        def _q_at(state, oid):
            for lot in state.cargo_manifest:
                if int(lot.order_id) == int(oid):
                    return float(lot.quality_remaining)
            for rec in state.delivered_to_depot:
                if int(rec.order_id) == int(oid):
                    return float(rec.delivered_quality)
            return None

        chain_A = chain_B = None
        if depart is not None and depart >= cc_before_t - 1e-9:
            chain_A, _ = transition_segment(cc0, depart_time=cc_before_t, arrival_time=depart,
                                            service_finish=depart, served_customer=None,
                                            active_zone_mask=mask, contract=eff)
        if arrival is not None:
            chain_B, _ = transition_segment(cc0, depart_time=cc_before_t, arrival_time=arrival,
                                            service_finish=arrival, served_customer=None,
                                            active_zone_mask=mask, contract=eff)

        rows = []
        for r in new:
            oid = int(r.order_id)
            q_rec = float(r.delivered_quality)
            qA = _q_at(chain_A, oid) if chain_A is not None else None
            qB = _q_at(chain_B, oid) if chain_B is not None else None
            rows.append({'order_id': oid,
                         'predicted_quality_if_recorded_at_depart': qA,
                         'predicted_quality_if_recorded_at_arrival': qB,
                         'recorded_delivered_quality': q_rec,
                         'matches_depart_chain': (None if qA is None else abs(qA - q_rec) <= 1e-9),
                         'matches_arrival_chain': (None if qB is None else abs(qB - q_rec) <= 1e-9),
                         'return_leg_decay_arrival_minus_depart': (None if (qA is None or qB is None)
                                                                   else qA - qB)})
        n_measured = len(rows)
        all_arrival = bool(rows) and all(x['matches_arrival_chain'] for x in rows)
        any_depart = any(x['matches_depart_chain'] for x in rows)
        leg_positive = (arrival is not None and depart is not None and arrival - depart > 1e-9)
        a5 = {'forced': True, 'vehicle_id': int(target5.vehicle_id),
              'status_after': target5.status,
              'node_before_return': node_before,
              'return_travel_h': travel_h,
              'coldchain_updated_time_before': cc_before_t,
              'coldchain_updated_time_after': cc_after_t,
              'recorded_return_arrival_times': rec_times,
              'implied_depart_time': depart,
              'return_leg_duration_h': (None if (arrival is None or depart is None)
                                        else arrival - depart),
              'n_new_delivery_records': n_measured,
              'clock_advanced_to_arrival': (None if arrival is None
                                            else abs(cc_after_t - arrival) <= 1e-6),
              'recorded_quality_matches_arrival_chain': all_arrival,
              'recorded_quality_matches_depart_chain': any_depart,
              'rows': rows,
              'return_finish_read_back': target5.return_finish,
              'note': ('return_finish is CONSUMED by _advance_fleet (strict_online_env.py:428-429), '
                       'so it reads back as None after the call; the physics unload time is the '
                       'recorded return_arrival_time, and the two chains discriminate the claim')}
        # Only a POSITIVE measurement may produce NOT_REPRODUCED.
        if not n_measured or not leg_positive or not rows or chain_A is None or chain_B is None:
            verdict5 = 'NOT_VERIFIABLE'
        elif all_arrival and not any_depart:
            verdict5 = 'NOT_REPRODUCED'
        elif any_depart and not all_arrival:
            verdict5 = 'COUNTEREXAMPLE_REPRODUCED'
        else:
            verdict5 = 'NOT_VERIFIABLE'
    else:
        verdict5 = 'NOT_VERIFIABLE'
    chk('CP-A-c5', 'CP-A-c5 forcing _force_return_all exposes the physics-vs-recorded time gap',
        'NOT_VERIFIABLE' if not a5['forced'] else verdict5, a5)

    ids = [r['cid'] for r in RESULTS]
    missing = [i for i in PREREGISTERED if i not in ids]
    chk('PREREG', 'every pre-registered counterexample is present; none dropped',
        'OK' if not missing else 'MISSING', {'preregistered': PREREGISTERED, 'covered': ids,
                                             'missing': missing})

    # ================= AUDIT: the probes left the production path unchanged ==================
    # An EMPIRICAL read-only check, not the structural argument "everything ran on clones": the
    # same instance is run again in a FRESH env with NO hook at all, and the full production
    # outcome (per-vehicle service sequence, times, picked orders, depot-unload records and the
    # served mask) must be bit-identical to the probed run.
    def _outcome_fp(tr, sm):
        # `float()`/`int()` coercion is REQUIRED, not cosmetic: later instances carry np.float32
        # times (the dataset arrays differ per instance) and json.dumps refuses them (found by the
        # robustness sweep over instances 1-5).
        def f(x):
            return None if x is None else float(x)

        rows = []
        for t in sorted(tr, key=lambda x: int(x.vehicle_id)):
            rows.append({
                'v': int(t.vehicle_id), 'dispatch': f(t.dispatch_time),
                'services': [[int(s.node), f(s.depart_time), f(s.arrival_time),
                              f(s.service_finish),
                              (None if s.picked_order_id is None else int(s.picked_order_id))]
                             for s in t.services],
                'return_depart': f(t.return_depart), 'return_arrival': f(t.return_arrival),
                'unload': [[int(r.order_id), f(r.return_arrival_time), f(r.delivered_quality)]
                           for r in t.depot_unload_records]})
        return hashlib.sha256(json.dumps(
            {'rows': rows, 'served': [int(i) for i in np.where(sm)[0]]},
            sort_keys=True).encode('utf-8')).hexdigest()

    env2 = make_env(dataset, eff, hook=None)
    traces2, served2 = env2.run(inst)
    fp_probed, fp_clean = _outcome_fp(traces, served), _outcome_fp(traces2, served2)
    chk('AUDIT', 'the counterexample probes did not perturb the production trajectory',
        'OK' if fp_probed == fp_clean else 'PRODUCTION_PATH_PERTURBED',
        {'probed_outcome_sha256': fp_probed, 'clean_rerun_outcome_sha256': fp_clean,
         'n_vehicles': len(traces), 'n_served': int(np.count_nonzero(served)),
         'note': ('a fresh env with NO hook reproduces the probed run exactly, so every probe '
                  'acted on a detached clone')})

    counts = {}
    for r in RESULTS:
        counts[r['verdict']] = counts.get(r['verdict'], 0) + 1
    out = {
        'stage': 'D3_G0_Evidence_B3CF_submit_failure_paths',
        'read_only': True, 'production_state_written': False, 'commit_modified': False,
        'real_writeback_enabled': False, 'd8_relaxed': False,
        'gates': {'G-L': 'NOT_ESTABLISHED', 'G-C': 'NOT_ESTABLISHED'},
        'critical_rule_honoured': ('CP-F10 reproduces a DEFECT and is reported as '
                                   'COUNTEREXAMPLE_REPRODUCED; it is NOT reported as G-L passing'),
        'preregistered': PREREGISTERED, 'covered': ids,
        'verdict_counts': counts,
        'identities': {
            'data_npz_sha256': sha256_file(CAL_NPZ),
            'profile_hash': profile.profile_hash,
            'effective_contract_hash': eff.contract_hash,
            'prereg_spec_sha256': sha256_file(EXT.parent / 'research' / 'review_3080_20261010' /
                                              'd3_g0_evidence_root_v1' /
                                              'B3CF_COUNTEREXAMPLE_SPEC.json'),
            'code': {
                'scripts/tests/test_d3_b3cf_submit_failures.py': sha256_file(Path(__file__)),
                'scripts/simulation/d3_install_gate.py': sha256_file(SIM / 'd3_install_gate.py'),
                'scripts/simulation/strict_online_env.py':
                    sha256_file(SIM / 'strict_online_env.py'),
                'scripts/simulation/d3_composed_certifier.py':
                    sha256_file(SIM / 'd3_composed_certifier.py'),
                'scripts/coldchain/coldchain_state.py':
                    sha256_file(EXT / 'scripts' / 'coldchain' / 'coldchain_state.py'),
            },
        },
        'results': RESULTS,
    }
    for r in RESULTS:
        print('  %-9s %-19s %s' % (r['cid'], r['verdict'], r['name']))
    print('\nverdict_counts:', json.dumps(counts, ensure_ascii=False))
    dest = (EXT.parent / 'research' / 'review_3080_20261010' / 'd3_g0_evidence_root_v1' /
            'B3CF_FAILURE_PATH_MATRIX.json')
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + '\n',
                    encoding='utf-8', newline='\n')
    print('wrote', dest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
