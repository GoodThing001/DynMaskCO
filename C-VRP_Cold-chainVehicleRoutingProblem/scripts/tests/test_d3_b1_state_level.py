#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B1: state-level counterexamples C1-C5 on the REAL environment.

Authorised scope (user ruling 2026-10-14): LIMITED B1. Real `StrictOnlineEnv` + real
`VehicleState`; READ-ONLY capture and eligibility; zero production write-back; no `_commit`
modification; no composed certifier; no cross-process hash protocol; no persistent receipt; no
training; no effect experiment.

Acceptance criteria this file is written against (user ruling):
  1. all five counterexamples reach a definite conclusion; NOT_VERIFIABLE is allowed and no
     failing state may be filtered out;
  2. everything uses real `VehicleState` -- no fabricated `committed_next`-style attributes;
  3. the zero-write-back no-op path agrees with the untouched baseline;
  4. strict-online visible information is explicitly separated from the certification state;
  5. an empty `pool_B` is PRESERVED, never dropped after the fact;
  6. identities, environment and code SHA are registered;
  7. historical frozen trees, old adjudications and original tests are untouched.

Stop condition honoured: state is captured only through existing sanctioned entry points
(`capture_recourse_snapshot` / `restore_recourse_snapshot` / `prepare_decision_point` on a
CLONE). If that had been impossible, this file reports the gap (`stop_condition_triggered`)
instead of editing the env.

Run: python scripts/tests/test_d3_b1_state_level.py [--instances 0 1] [--heavy-cap 25]
Exit: 0 when every criterion has a definite conclusion, else 1.
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
import d3_geo_adapter as G                                             # noqa: E402
import d3_state_assembly as A                                          # noqa: E402
import action_contract as AC                                          # noqa: E402
from strict_online_env import StrictOnlineEnv, VehicleState            # noqa: E402
from jf1h_repair import make_continuation                             # noqa: E402
from coldchain_contract import (default_pilot_contract,               # noqa: E402
                                apply_objective_profile, ObjectiveProfile)
from recourse_snapshot import snapshot_state_hash                     # noqa: E402
from dynmaskco_cc_context import mutable_scope                        # noqa: E402

EXT = HERE.parent.parent
CAL_NPZ = EXT / 'data' / 'm0_scale' / 'dcc_50_r1_edod05_cal_teacher.npz'
PROFILE_JSON = EXT / 'results' / 'o0cc' / 'scale_v2' / 'objective_profile.json'
CAPACITY = 50.0
NUM_VEHICLES = 25

RESULTS = []


def chk(cid, name, ok, detail=None):
    RESULTS.append({'cid': cid, 'name': name, 'ok': bool(ok), 'detail': detail})


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_profile():
    d = json.loads(PROFILE_JSON.read_text(encoding='utf-8'))
    if d.get('name') == 'o0cc-pilot-devmean-equal-v1':
        raise SystemExit('refusing the INVALIDATED v1 profile')
    p = ObjectiveProfile(name=d['name'], distance_scale=float(d['distance_scale']),
                         quality_scale=float(d['quality_scale']),
                         energy_scale=float(d['energy_scale']),
                         lambda_quality=float(d['lambda_quality']),
                         lambda_energy=float(d['lambda_energy']),
                         scale_source=d.get('scale_source', 'pilot'),
                         dev_statistics=d.get('dev_statistics'))
    if p.profile_hash != d.get('profile_hash'):
        raise SystemExit('profile hash mismatch: %s vs %s' % (p.profile_hash[:12],
                                                             str(d.get('profile_hash'))[:12]))
    return p


def make_env(dataset, profile, hook=None):
    contract = apply_objective_profile(default_pilot_contract(), profile)
    env = StrictOnlineEnv(dataset, capacity=CAPACITY, tw_speed=1.0, num_vehicles=NUM_VEHICLES,
                          replanner=make_continuation(), coldchain_contract=contract,
                          booking_horizon=None)
    env.snapshot_hook = hook
    return env


def trace_projection(traces, served_mask):
    return {
        'served_mask': [bool(x) for x in served_mask],
        'services': [[int(sr.node) for sr in tr.services] for tr in traces],
        'dispatch_time': [None if tr.dispatch_time is None else float(tr.dispatch_time)
                          for tr in traces],
        'return_depart': [None if tr.return_depart is None else float(tr.return_depart)
                          for tr in traces],
        'return_arrival': [None if tr.return_arrival is None else float(tr.return_arrival)
                           for tr in traces],
        'dispatch_energy': [float(tr.dispatch_preconditioning_energy_kwh) for tr in traces],
        'return_energy': [float(tr.return_segment_energy_kwh) for tr in traces],
    }


class Probe(object):
    """Read-only decision-point probe. Writes NOTHING to the live state (self-verified)."""

    def __init__(self, heavy_cap=25):
        self.heavy_cap = heavy_cap
        self.rows = []
        self.objects = []          # in-memory companions (plans / vehicles); never serialised
        self.live_writes_detected = 0
        self.n_calls = 0
        self._heavy_per_inst = {}

    def hook(self, env, inst_idx, clock, event_id, reveal_idx, vehicles, traces, served_mask,
             all_customers=None):
        self.n_calls += 1
        pre_hash = A.live_state_fingerprint(env, inst_idx, clock, event_id, reveal_idx, vehicles,
                                           traces, served_mask, all_customers)
        snap, _ = A.capture_decision_state(env, inst_idx, clock, event_id, reveal_idx, vehicles,
                                          traces, served_mask, all_customers)
        row = {
            'inst_idx': int(inst_idx), 'event_id': int(event_id), 'clock': float(clock),
            'pre_prepare_state_hash': pre_hash,
            'pre_prepare_context': {
                'clock': float(clock),
                'served_count': int(np.sum(served_mask)),
                'visible_ids': [int(i) for i in all_customers
                                if env.reveal_time[inst_idx, i] <= clock + 1e-6],
                'needs_replan_ids': sorted(int(v.vehicle_id) for v in vehicles
                                           if v.status in ('idle', 'ready') and v.needs_replan),
                'suffix_len_by_vid': {int(v.vehicle_id): len(v.mutable_suffix)
                                      for v in vehicles},
            },
            'live_suffix_len_total': int(sum(len(v.mutable_suffix) for v in vehicles)),
            'committed_next_live': {int(v.vehicle_id): (None if v.committed_next is None
                                                        else int(v.committed_next))
                                    for v in vehicles},
            'served_route_nodes': {int(v.vehicle_id): [int(x) for x in v.served_route
                                                       if int(x) > 0] for v in vehicles},
            'replan_reason_reveal_ids': sorted(int(v.vehicle_id) for v in vehicles
                                               if v.replan_reason == 'reveal'),
            'status_by_vid': {int(v.vehicle_id): v.status for v in vehicles},
        }
        heavy_no = self._heavy_per_inst.get(int(inst_idx), 0)
        if heavy_no < self.heavy_cap:
            self._heavy_per_inst[int(inst_idx)] = heavy_no + 1
            clone_v, clone_t, clone_sm = A.clone_state(snap)
            visible_ids = [int(i) for i in all_customers
                           if env.reveal_time[inst_idx, i] <= clock + 1e-6]
            cv, plans, replan_ids = A.prepare_clone(env, inst_idx, clock, clone_v, clone_sm,
                                                    visible_ids)
            post_hash = snapshot_state_hash(A.capture_decision_state(
                env, inst_idx, clock, event_id, reveal_idx, cv, clone_t, clone_sm,
                all_customers)[0])
            mut = mutable_scope(cv)
            # PROTECTED selection rule, fixed a priori and applied uniformly: designate the
            # MUTABLE vehicle with the longest NON-EMPTY plan suffix (ties -> larger vehicle id).
            # Protecting a vehicle with an empty suffix would exclude nothing and make the check
            # vacuous, which is why an arbitrary (e.g. highest-id idle) vehicle is not used here.
            with_suffix = [(len(plans[vid].suffix), int(vid)) for vid in mut
                           if vid in plans and len(plans[vid].suffix) > 0]
            protected = frozenset({max(with_suffix)[1]}) if with_suffix else frozenset()
            elig = A.mask_eligibility(env, inst_idx, clock, cv, clone_sm, visible_ids, plans,
                                      mut, protected)
            pools = A.pool_report(env, inst_idx, clock, cv, clone_sm, visible_ids, plans, mut,
                                 protected, elig)
            cred = A.decision_credential(env, inst_idx, clock, event_id, post_hash, plans,
                                        pre_prepare_hash=pre_hash)
            remaining = pools['pool_B'][:8]
            legal = A.collect_legal_order(env, inst_idx, plans, remaining, mut)
            row.update({
                'post_prepare_state_hash': post_hash,
                'pre_vs_post_differ': pre_hash != post_hash,
                'mutable_ids': sorted(int(x) for x in mut),
                'protected_ids': sorted(int(x) for x in protected),
                'eligibility': elig,
                'pools': pools,
                'credential': cred,
                'clone_vehicle_class': sorted({type(v).__module__ + '.' + type(v).__name__
                                                for v in cv}),
                'legal_probe': {'n_probed_customers': len(remaining),
                                'n_legal_actions': len(legal),
                                'legal_order_head': legal[:12]},
            })
            self.objects.append({'inst': int(inst_idx), 'event': int(event_id),
                                 'plans': plans, 'vehicles': cv, 'clone_traces': clone_t,
                                 'clone_served_mask': clone_sm, 'clock': float(clock),
                                 'credential': cred, 'snapshot': snap,
                                 'mutable_ids': sorted(int(x) for x in mut),
                                 'protected_ids': sorted(int(x) for x in protected),
                                 'visible_ids': visible_ids, 'env': env,
                                 'post_prepare_state_hash': post_hash,
                                 'pre_prepare_state_hash': pre_hash})
        else:
            row['heavy_work_skipped'] = True
        after_hash = A.live_state_fingerprint(env, inst_idx, clock, event_id, reveal_idx, vehicles,
                                             traces, served_mask, all_customers)
        row['readonly_fingerprint_ok'] = (pre_hash == after_hash)
        if not row['readonly_fingerprint_ok']:
            self.live_writes_detected += 1
        self.rows.append(row)


# ---------------------------------------------------------------------------
# C1-C3: eligibility exclusions, plus a search for the NAIVE-ACCEPT configuration
# ---------------------------------------------------------------------------
def _constructed_naive_accept(kind, objects):
    """Construct, on a DETACHED clone, the configuration the user warned about.

    A customer can end up in a replannable vehicle's plan suffix while ALSO being committed
    (or already served). A union-of-suffix membership test would ACCEPT it; the per-vehicle +
    committed/served rule must REJECT it. Natural runs rarely produce this state, so it is
    CONSTRUCTED here -- on a clone, so nothing live is touched -- and the two rules are compared
    on the same state version.
    """
    for o in objects:
        plans, cv, mut = o['plans'], o['vehicles'], o['mutable_ids']
        if not mut or not plans:
            continue
        if kind == 'committed':
            victims = {int(v.committed_next) for v in cv if v.committed_next not in (None, 0)}
            why = 'committed'
        else:
            victims = set()
            for v in cv:
                for x in v.served_route:
                    if int(x) > 0:
                        victims.add(int(x))
            for c in range(1, o['env'].num_nodes):
                if bool(o['clone_served_mask'][c]):
                    victims.add(int(c))
            why = 'served'
        victims -= set(plans.get(mut[0]).suffix if mut[0] in plans else [])
        if not victims:
            continue
        c = sorted(victims)[0]
        vid = mut[0]
        p = plans[vid]
        injected = dict(plans)
        injected[vid] = AC.VehiclePlan(p.vehicle_id, p.anchor_node, p.anchor_time,
                                       p.anchor_load, tuple(list(p.suffix) + [c]))
        elig = A.mask_eligibility(o['env'], o['inst'], o['clock'], cv,
                                 o['clone_served_mask'], o['visible_ids'], injected,
                                 set(mut), frozenset(o['protected_ids']))
        return {
            'constructed_on_detached_clone': True, 'kind': kind,
            'customer': c, 'injected_into_mutable_vehicle': int(vid),
            'union_of_suffix_membership_would_accept':
                c in elig['b_eligible_by_customer_vehicle'],
            'per_vehicle_rule_rejects': why in elig['exclusion_reasons'].get(c, []),
            'reason_recorded': elig['exclusion_reasons'].get(c, []),
            'eligible_after_injection': c in elig['eligible'],
            'inst': o['inst'], 'event': o['event'],
        }
    return {'constructed_on_detached_clone': True, 'kind': kind,
            'union_of_suffix_membership_would_accept': False,
            'per_vehicle_rule_rejects': False,
            'note': 'NOT_CONSTRUCTIBLE_IN_SAMPLE: no such customer existed in the sampled states'}


def c1_committed(heavy, objects):
    owners, excluded, naive, violations = 0, 0, [], []
    for r in heavy:
        e = r['eligibility']
        b_elig = e['b_eligible_by_customer_vehicle']
        for c in e['committed_next_by_customer_vehicle']:
            owners += 1
            why = e['exclusion_reasons'].get(c, [])
            if 'committed' in why:
                excluded += 1
            else:
                violations.append({'inst': r['inst_idx'], 'event': r['event_id'], 'customer': c,
                                   'reasons': why})
            if c in b_elig:
                naive.append({'inst': r['inst_idx'], 'event': r['event_id'], 'customer': c,
                              'in_mutable_suffix_of': b_elig[c],
                              'why': 'a union-of-suffix membership test would ACCEPT this '
                                     'committed customer; the per-vehicle + committed rule '
                                     'REJECTS it'})
    naive_constructed = _constructed_naive_accept('committed', objects)
    ok = (owners and not violations)
    return {'status': ('CONFIRMED' if ok else
                       'NOT_VERIFIABLE_IN_SAMPLE' if not owners else 'COUNTEREXAMPLE'),
            'narrative': ('every committed customer observed was excluded by the per-vehicle '
                          'eligibility rule' if ok else
                          'no committed customer occurred in this sample' if not owners else
                          'a committed customer was eligible'),
            'n_committed_observed': owners, 'n_correctly_excluded': excluded,
            'violations': violations, 'naive_accept_configurations': naive,
            'constructed_naive_accept_demo': naive_constructed}


def c2_served(heavy, objects=None):
    excluded, naive, violations, disc = 0, [], [], 0
    for r in heavy:
        e = r['eligibility']
        if e['served_mask_vs_served_route_discrepancy']:
            disc += 1
        for c, why in e['exclusion_reasons'].items():
            if 'served' in why:
                excluded += 1
        # REAL violation check: no eligible customer may be in the independently computed
        # served set (from served_mask UNION every vehicle's served_route)
        overlap = sorted(set(e['eligible']) & set(e['served_nodes']))
        if overlap:
            violations.append({'inst': r['inst_idx'], 'event': r['event_id'],
                               'served_but_eligible': overlap})
        # served customers that a union-of-suffix membership test would have ACCEPTED
        for c, vids in e['b_eligible_by_customer_vehicle'].items():
            if 'served' in e['exclusion_reasons'].get(c, []):
                naive.append({'inst': r['inst_idx'], 'event': r['event_id'], 'customer': c,
                              'in_mutable_suffix_of': vids})
    naive_constructed = _constructed_naive_accept('served', objects or [])
    return {'status': ('CONFIRMED' if excluded and not violations else
                       'NOT_VERIFIABLE_IN_SAMPLE' if not excluded else 'COUNTEREXAMPLE'),
            'narrative': ('no served customer was eligible at any probed decision point'
                          if excluded and not violations else
                          'no served customer occurred in this sample' if not excluded else
                          'a served customer was eligible'),
            'n_served_excluded': excluded, 'served_mask_vs_route_discrepancy_events': disc,
            'violations': violations, 'n_naive_accept_configurations': len(naive),
            'naive_accept_configurations': naive[:20],
            'constructed_naive_accept_demo': naive_constructed}


def c3_protected(heavy):
    excluded, violations, flag_cases = 0, [], []
    for r in heavy:
        e = r['eligibility']
        if not e['protected_ids']:
            continue
        for c in e['protected_tail_by_customer_vehicle']:
            if 'protected' in e['exclusion_reasons'].get(c, []):
                excluded += 1
            else:
                violations.append({'inst': r['inst_idx'], 'event': r['event_id'], 'customer': c})
    # validate_repair_plan must flag a plan that changes a protected vehicle
    for o in [o for o in _OBJECTS if o.get('plans')]:
        plans = o['plans']
        prot = [vid for vid in plans if vid == max(plans)] if plans else []
        if not prot:
            continue
        changed = {vid: AC.VehiclePlan(p.vehicle_id, p.anchor_node, p.anchor_time,
                                       p.anchor_load, p.suffix) for vid, p in plans.items()}
        vid = prot[0]
        pv = changed[vid]
        changed[vid] = AC.VehiclePlan(pv.vehicle_id, pv.anchor_node, pv.anchor_time,
                                      pv.anchor_load, tuple(list(pv.suffix) + [999]))
        _ok, detail = _validate_repair_plan(changed, plans, [], prot)
        flag_cases.append({'inst': o['inst'], 'event': o['event'],
                           'protected_vehicles': list(prot),
                           'protected_vehicle_changed': detail['protected_vehicle_changed']})
        break
    return {'status': ('CONFIRMED' if excluded and not violations and flag_cases and
                       flag_cases[0]['protected_vehicle_changed'] else
                       'NOT_VERIFIABLE_IN_SAMPLE' if not excluded else 'COUNTEREXAMPLE'),
            'narrative': ('protected-tail customers were excluded and a changed protected '
                          'vehicle is flagged by validate_repair_plan'
                          if excluded and not violations and flag_cases and
                          flag_cases[0]['protected_vehicle_changed'] else
                          'no protected-tail customer occurred in this sample'),
            'n_protected_customers_excluded': excluded, 'violations': violations,
            'repair_plan_flag_case': flag_cases[0] if flag_cases else None}


# ---------------------------------------------------------------------------
# C4: state drift observable + plan_hash blind spot (D4)
# ---------------------------------------------------------------------------
def c4_state_version(heavy, objects):
    by_inst = {}
    comparisons = changes = 0
    for r in heavy:
        prev = by_inst.get(r['inst_idx'])
        if prev is not None:
            comparisons += 1
            if r['credential']['state_hash'] != prev['credential']['state_hash']:
                changes += 1
        by_inst[r['inst_idx']] = r

    # reveal detection must NOT use `replan_reason`: it is STICKY (the env sets it at a reveal
    # and never clears it), so counting it over-reports. We instead detect a reveal as GROWTH of
    # the visible set, which is the causally meaningful event, and record the stickiness as a
    # finding in its own right.
    prev_visible, reveal_events, reveal_tail_cleared = {}, 0, 0
    sticky_reason_rows = sum(1 for r in heavy if r['replan_reason_reveal_ids'])
    for r in heavy:
        vis = set(r['pre_prepare_context']['visible_ids'])
        pv = prev_visible.get(r['inst_idx'])
        if pv is not None and len(vis) > len(pv):
            reveal_events += 1
            non_closed_nonempty = sum(
                1 for vid, n in r['pre_prepare_context']['suffix_len_by_vid'].items()
                if n > 0 and r['status_by_vid'].get(vid) != 'closed')
            if non_closed_nonempty == 0:
                reveal_tail_cleared += 1
        prev_visible[r['inst_idx']] = vis

    # D4 fixture from a REAL plan set
    d4 = None
    if objects:
        base = objects[0]['plans']
        alt_time = {v: AC.VehiclePlan(p.vehicle_id, p.anchor_node, p.anchor_time + 1.0,
                                      p.anchor_load, p.suffix) for v, p in base.items()}
        alt_load = {v: AC.VehiclePlan(p.vehicle_id, p.anchor_node, p.anchor_time,
                                      p.anchor_load + 5.0, p.suffix) for v, p in base.items()}
        d4 = {
            'plan_hash_equal_after_anchor_time_shift':
                AC.plan_hash(base) == AC.plan_hash(alt_time),
            'anchor_digest_differs_after_anchor_time_shift':
                A.plan_digest_anchor_v1(base) != A.plan_digest_anchor_v1(alt_time),
            'plan_hash_equal_after_anchor_load_shift':
                AC.plan_hash(base) == AC.plan_hash(alt_load),
            'anchor_digest_differs_after_anchor_load_shift':
                A.plan_digest_anchor_v1(base) != A.plan_digest_anchor_v1(alt_load),
            'plan_hash_len': len(AC.plan_hash(base)),
        }

    # the EXISTING install gate cannot see state drift for the SAME plan. Isolated properly:
    # both credentials are built from the SAME plan object, differing only in state_hash.
    blind = None
    if objects:
        o = objects[0]
        c_same = A.decision_credential(o['env'], o['inst'], o['clock'], o['event'],
                                       o['post_prepare_state_hash'], o['plans'])
        c_other = A.decision_credential(o['env'], o['inst'], o['clock'] + 1.0, o['event'] + 99,
                                        'f' * 64, o['plans'])
        led = G.InstallLedger(install_fn=lambda p: None)
        installed = led.finish(failure_reason=None, plan=o['plans'], original_plan=o['plans'],
                               certified=True, certified_plan_hash=c_same['plan_hash'],
                               certifier='b1.state_test', deadline_ok=True)
        blind = {
            'constructed': 'two credentials over the SAME plan object, different state_hash',
            'same_plan_hash': c_same['plan_hash'] == c_other['plan_hash'],
            'different_state_hash': c_same['state_hash'] != c_other['state_hash'],
            'credential_diff': A.credential_diff(c_same, c_other),
            'existing_gate_installed_despite_state_difference': bool(installed),
            'existing_gate_refusal_reason': (led.events[-1] or {}).get('reason'),
            'note': ('the gate compares plan hash vs certified plan hash only, so a '
                     'same-plan / different-state install PASSES it'),
        }
    ok = (comparisons > 0 and changes > 0 and reveal_events > 0 and reveal_tail_cleared > 0
          and d4 and d4['plan_hash_equal_after_anchor_time_shift']
          and d4['anchor_digest_differs_after_anchor_time_shift']
          and blind and blind['existing_gate_installed_despite_state_difference'])
    return {'status': 'CONFIRMED' if ok else 'NOT_VERIFIABLE',
            'narrative': ('state drift is observable through the capture path, a reveal clears '
                          'the mutable tail, plan_hash cannot see anchor changes, and the '
                          'existing install gate passes a same-plan/different-state credential'
                          if ok else 'one of the required observations was absent'),
            'consecutive_comparisons': comparisons, 'consecutive_state_hash_changes': changes,
            'reveal_events_detected_by_visible_growth': reveal_events,
            'reveal_events_with_cleared_tail': reveal_tail_cleared,
            'replan_reason_reveal_rows': sticky_reason_rows,
            'finding_replan_reason_is_sticky': bool(sticky_reason_rows > reveal_events),
            'D4_plan_hash_blind_spot': d4, 'existing_gate_blind_to_state_drift': blind}


# ---------------------------------------------------------------------------
# C5: commit behaviour on a DETACHED copy (no live write)
# ---------------------------------------------------------------------------
def c5_commit(dataset, profile, inst_idx):
    env = make_env(dataset, profile)
    holder = {}

    def grab(e, inst, clock, event_id, reveal_idx, veh, traces, sm, all_customers=None):
        if not holder:
            holder['row'] = (inst, clock, event_id, reveal_idx, all_customers)
            holder['vehicles'] = veh
            holder['traces'] = traces
            holder['sm'] = sm
    env.snapshot_hook = grab
    env.run(inst_idx)
    if not holder:
        return {'conclusion': 'NOT_VERIFIABLE: no decision point reached'}
    inst, clock, event_id, reveal_idx, all_customers = holder['row']
    snap, _ = A.capture_decision_state(env, inst, clock, event_id, reveal_idx,
                                      holder['vehicles'], holder['traces'], holder['sm'],
                                      all_customers)
    live_before = A.live_state_fingerprint(env, inst, clock, event_id, reveal_idx,
                                          holder['vehicles'], holder['traces'], holder['sm'],
                                          all_customers)
    # TWO independent clones: one to observe the commit loop's guard, one to observe `_commit`.
    # Using a single clone would let our own commit change the statuses we then report on.
    cv_guard, _ct_g, _csm_g = A.clone_state(snap)
    cv, ct, csm = A.clone_state(snap)

    # (a) the commit loop's own guard, measured BEFORE any commit of ours
    guard = {}
    for v in cv_guard:
        guard.setdefault(v.status, v.status in ('idle', 'ready'))
    n_idle_ready = sum(1 for v in cv_guard if v.status in ('idle', 'ready'))

    # (b) commit a SERVED customer -- an executed-prefix violation -- on the detached copy
    served_nodes = {int(sr.node) for tr in ct for sr in tr.services}
    served_in_mask = {c for c in range(1, env.num_nodes) if bool(csm[c])}
    violation_victim = sorted(served_nodes | served_in_mask)
    a = None
    target = next((v for v in cv if v.status in ('idle', 'ready')), None)
    if target is not None and violation_victim:
        c = violation_victim[0]
        before = (target.status, target.committed_next)
        env._commit(inst, clock, target, ct, c)
        a = {'customer': int(c), 'served_in_mask': bool(csm[c]),
             'in_served_route': c in served_nodes,
             'status_before': before[0], 'committed_next_before': before[1],
             'status_after': target.status,
             'committed_next_after': (None if target.committed_next is None
                                      else int(target.committed_next)),
             'committed_despite_being_served': bool(target.committed_next == c)}

    live_after = A.live_state_fingerprint(env, inst, clock, event_id, reveal_idx,
                                         holder['vehicles'], holder['traces'], holder['sm'],
                                         all_customers)
    ok = (a is not None and a['committed_despite_being_served']
          and n_idle_ready > 0 and any(not v for v in guard.values())
          and live_before == live_after)
    return {'status': 'CONFIRMED' if ok else 'NOT_VERIFIABLE',
            'narrative': ('`_commit` accepted a SERVED customer and the commit loop silently '
                          'skips every vehicle whose status is not idle/ready, so "install '
                          'callback invoked" need not equal "state committed"'
                          if ok else 'the detached-copy observation did not reproduce'),
            'detached_copy_only': True, 'live_state_unchanged': live_before == live_after,
            'served_prefix_violation': a,
            'n_idle_ready_vehicles_in_clone': n_idle_ready,
            'loop_guard_status_in_idle_ready': guard,
            'loop_would_skip_statuses': sorted(s for s, v in guard.items() if not v)}


_OBJECTS = []
_validate_repair_plan = None


def main():
    global _OBJECTS, _validate_repair_plan
    ap = argparse.ArgumentParser()
    ap.add_argument('--instances', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--heavy-cap', type=int, default=25)
    args = ap.parse_args()

    from mpre_policy import validate_repair_plan as _vrp
    _validate_repair_plan = _vrp

    profile = load_profile()
    dataset = dict(np.load(CAL_NPZ))
    identities = {
        'data_npz': str(CAL_NPZ.relative_to(EXT)).replace('\\', '/'),
        'data_npz_sha256': sha256_file(CAL_NPZ),
        'profile_json': str(PROFILE_JSON.relative_to(EXT)).replace('\\', '/'),
        'profile_json_sha256': sha256_file(PROFILE_JSON),
        'profile_name': profile.name, 'profile_hash': profile.profile_hash,
        'capacity': CAPACITY, 'num_vehicles': NUM_VEHICLES,
        'code': {
            'scripts/simulation/d3_state_assembly.py': sha256_file(SIM / 'd3_state_assembly.py'),
            'scripts/simulation/d3_geo_adapter.py': sha256_file(SIM / 'd3_geo_adapter.py'),
            'scripts/simulation/strict_online_env.py': sha256_file(SIM / 'strict_online_env.py'),
            'scripts/simulation/recourse_snapshot.py': sha256_file(SIM / 'recourse_snapshot.py'),
            'scripts/simulation/dynmaskco_cc_context.py':
                sha256_file(SIM / 'dynmaskco_cc_context.py'),
            'scripts/tests/test_d3_b1_state_level.py': sha256_file(Path(__file__)),
        },
        'history_untouched_sha256': {
            n: sha256_file(HERE / n) for n in
            ('test_d3_geo_adapter.py', 'test_d3_adapter_risks.py',
             'test_d3_assembly_invariants.py', 'test_d3_g0_red_phase.py',
             'test_d3_g0_green_phase.py')},
    }
    env0 = make_env(dataset, profile)
    identities['scenario_identity'] = A.scenario_identity(env0)
    identities['contract_identity'] = A.contract_identity(env0)
    identities['runtime'] = {
        'python': sys.version.split()[0],
        'platform': sys.platform,
        'numpy': np.__version__,
        'executable': sys.executable,
        'scope': A.B1_SCOPE,
        'writes_production_state': A.B1_WRITES_PRODUCTION_STATE,
    }

    # ---- acceptance #7: historical artefacts unchanged since the freeze -------------------
    freeze_path = (EXT.parent / 'research' / 'review_3080_20261010' /
                   'd3_g0_green_phase_root_v1' / 'PREFIX_FREEZE.json')
    green_id_path = (EXT.parent / 'research' / 'review_3080_20261010' /
                     'd3_g0_green_phase_root_v1' / 'GREEN_PHASE_IDENTITY.json')
    hist = {}
    prod = {}
    if freeze_path.is_file():
        fr = json.loads(freeze_path.read_text(encoding='utf-8'))
        for name, rec in fr['files'].items():
            # manifest paths are REPO-ROOT relative, not extension relative
            live = EXT.parent / rec['path']
            got = sha256_file(live) if live.is_file() else None
            is_test = '/tests/' in rec['path']
            if is_test:
                hist[rec['path']] = {
                    'kind': 'historical_test_original',
                    'expected_sha256': rec['sha256_at_base_commit'], 'live_sha256': got,
                    'unchanged': got == rec['sha256_at_base_commit']}
            else:
                # the PRODUCTION source is EXPECTED to have changed in G0-Safety batch 1; the
                # requirement is that it changed to the REGISTERED post-fix hash, not that it
                # still equals the pre-fix one.
                prod[rec['path']] = {'pre_fix_sha256': rec['sha256_at_base_commit'],
                                     'live_sha256': got,
                                     'differs_from_pre_fix': got != rec['sha256_at_base_commit']}
    gp_hist = {}
    gp_prod = {}
    if green_id_path.is_file():
        gi = json.loads(green_id_path.read_text(encoding='utf-8'))
        for rel, want in gi['tests']['unchanged_from_base_commit'].items():
            got = sha256_file(EXT.parent / rel)
            gp_hist[rel] = {'kind': 'historical_test_original', 'expected_sha256': want,
                            'live_sha256': got, 'unchanged': got == want}
        ps = gi['production_source']
        rel = ps['path']
        got = sha256_file(EXT.parent / rel)
        gp_prod[rel] = {'expected_post_fix_sha256': ps['post_fix_sha256'], 'live_sha256': got,
                        'matches_registered_post_fix': got == ps['post_fix_sha256']}
    tests_ok = (bool(hist) and all(v['unchanged'] for v in hist.values())
                and bool(gp_hist) and all(v['unchanged'] for v in gp_hist.values()))
    prod_ok = (bool(prod) and all(v['differs_from_pre_fix'] for v in prod.values())
               and bool(gp_prod) and all(v['matches_registered_post_fix'] for v in gp_prod.values()))
    chk('HIST', 'historical test files byte-identical to the frozen pre-fix record; production '
        'source equals the REGISTERED post-fix hash (not the pre-fix one)',
        tests_ok and prod_ok,
        {'historical_tests_checked': len(hist) + len(gp_hist),
         'test_mismatches': [k for k, v in list(hist.items()) + list(gp_hist.items())
                             if not v['unchanged']],
         'production_source': {'differs_from_pre_fix': prod_ok,
                               'detail': {**prod, **gp_prod}}})
    identities['history_unchanged_check'] = {'prefix_freeze_tests': hist, 'green_phase_tests': gp_hist,
                                             'production_source': {**prod, **gp_prod}}

    # ---- precondition: the baseline must be deterministic --------------------------------
    proj = []
    for _ in range(2):
        e = make_env(dataset, profile)
        tr, sm = e.run(args.instances[0])
        proj.append(json.dumps(trace_projection(tr, sm), sort_keys=True))
    chk('P0', 'baseline run is deterministic (two fresh runs identical)', len(set(proj)) == 1)

    # ---- no-op equivalence (acceptance #3) + read-only self-verification ------------------
    eq_rows, probes = [], {}
    for inst in args.instances:
        e_base = make_env(dataset, profile)
        tr_b, sm_b = e_base.run(inst)
        pr = Probe(heavy_cap=args.heavy_cap)
        e_probe = make_env(dataset, profile, hook=pr.hook)
        tr_p, sm_p = e_probe.run(inst)
        probes[inst] = pr
        same = (json.dumps(trace_projection(tr_b, sm_b), sort_keys=True) ==
                json.dumps(trace_projection(tr_p, sm_p), sort_keys=True))
        eq_rows.append({'inst': inst, 'identical': same, 'n_decision_points': pr.n_calls,
                        'live_writes_detected': pr.live_writes_detected})
    chk('EQ', 'no-op probe path is identical to the untouched baseline (traces + served_mask)',
        all(r['identical'] for r in eq_rows), eq_rows)
    chk('RO', 'the probe wrote NOTHING to live state at any decision point',
        all(r['live_writes_detected'] == 0 for r in eq_rows), eq_rows)

    all_rows = [r for pr in probes.values() for r in pr.rows]
    heavy = [r for r in all_rows if 'eligibility' in r]
    _OBJECTS = [o for pr in probes.values() for o in pr.objects]

    # ---- acceptance #4: pre-prepare vs post-prepare separation ---------------------------
    differ = [r for r in heavy if r['pre_vs_post_differ']]
    chk('SEP', 'pre-prepare (visible) and post-prepare (certification) states are DISTINCT',
        bool(heavy) and bool(differ),
        {'n_heavy': len(heavy), 'n_differing': len(differ),
         'example_pair': ([heavy[0]['pre_prepare_state_hash'][:12],
                           heavy[0]['post_prepare_state_hash'][:12]] if heavy else None)})
    ctx_keys = sorted(all_rows[0]['pre_prepare_context']) if all_rows else []
    chk('SEP2', 'the visible context carries NO post-prepare-only field',
        ctx_keys == ['clock', 'needs_replan_ids', 'served_count', 'suffix_len_by_vid',
                     'visible_ids'], ctx_keys)

    # ---- acceptance #2: real VehicleState only -------------------------------------------
    classes = sorted({c for r in heavy for c in r['clone_vehicle_class']})
    chk('REAL', 'all state is real strict_online_env.VehicleState (no fabricated attribute)',
        classes == ['strict_online_env.VehicleState'],
        {'vehicle_classes': classes,
         'note': 'vehicles come from recourse_snapshot.restore_recourse_snapshot; no '
                 'committed_next-style attribute is injected anywhere (contrast: Red Phase A-3b)'})

    # ---- C1-C5 ---------------------------------------------------------------------------
    r1 = c1_committed(heavy, _OBJECTS)
    chk('C1', 'C1 committed orders -- definite conclusion (%s)' % r1['status'],
        r1['status'] != 'COUNTEREXAMPLE', r1)
    r2 = c2_served(heavy, _OBJECTS)
    chk('C2', 'C2 frozen executed prefix -- definite conclusion (%s)' % r2['status'],
        r2['status'] != 'COUNTEREXAMPLE', r2)
    r3 = c3_protected(heavy)
    chk('C3', 'C3 protected vehicles -- definite conclusion (%s)' % r3['status'],
        r3['status'] != 'COUNTEREXAMPLE', r3)
    r4 = c4_state_version(heavy, _OBJECTS)
    chk('C4', 'C4 state-version change + D4 plan-hash blind spot (%s)' % r4['status'],
        r4['status'] == 'CONFIRMED', r4)
    r5 = c5_commit(dataset, profile, args.instances[0])
    chk('C5', 'C5 post-certification commit failure on a detached copy (%s)' % r5['status'],
        r5['status'] == 'CONFIRMED', r5)

    # ---- acceptance #5: empty pool_B preserved -------------------------------------------
    empties = [r for r in heavy if r['pools']['empty']]
    chk('POOL', 'empty pool_B is PRESERVED and counted (decision point not dropped)',
        bool(empties),
        {'n_empty_decision_points': len(empties), 'n_heavy_decision_points': len(heavy),
         'all_empty_are_literal_empty_lists': all(r['pools']['pool_B'] == [] for r in empties),
         'example': ({'inst': empties[0]['inst_idx'], 'event': empties[0]['event_id'],
                      'reason': 'pool_B == [] recorded with empty=True, state kept'}
                     if empties else None)})

    # ---- acceptance #6: pool authority + reserved diagnostic (R2) ------------------------
    inter = [r for r in heavy if r['pools']['diag_pool_B_inter_reserved']]
    auth_ok = all(r['pools']['authoritative_rule'] == 'mask_candidate_pool' for r in heavy)
    # R2 property: the reserved intersection is RECORDED but never used to shrink pool_B, so
    # every member of the recorded intersection is still present in the authoritative pool.
    not_shrunk = all(set(r['pools']['diag_pool_B_inter_reserved']) <= set(r['pools']['pool_B'])
                     for r in heavy)
    subset_ok = all(set(r['pools']['pool_B']) <= set(r['pools']['decision_pool'])
                    for r in heavy)
    chk('R2', 'pool authority is mask_candidate_pool; reserved only a diagnostic (never shrinks '
        'the main pool)', auth_ok and not_shrunk and subset_ok,
        {'n_decision_points_with_nonempty_intersection': len(inter),
         'example_intersection': (inter[0]['pools']['diag_pool_B_inter_reserved'][:10]
                                  if inter else []),
         'intersection_members_still_in_pool_B': not_shrunk,
         'pool_B_subset_of_decision_pool': subset_ok,
         'note': 'a non-empty intersection is expected and is NOT a violation (R2)'})

    out = {
        'stage': 'D3_G0_Evidence_B1_state_level',
        'scope': A.B1_SCOPE,
        'production_state_written': False,
        'commit_modified': False,
        'real_env_used': True,
        'scientific_data_used': False,
        'effect_experiment_run': False,
        'instances': args.instances,
        'n_decision_points_probed': len(all_rows),
        'n_heavy_decision_points': len(heavy),
        'identities': identities,
        'results': RESULTS,
        'conclusions': {'C1': r1['status'], 'C2': r2['status'], 'C3': r3['status'],
                        'C4': r4['status'], 'C5': r5['status']},
        'narratives': {'C1': r1['narrative'], 'C2': r2['narrative'], 'C3': r3['narrative'],
                       'C4': r4['narrative'], 'C5': r5['narrative']},
        'c1_naive_accept_configurations': len(r1['naive_accept_configurations']),
        'c2_naive_accept_configurations': r2['n_naive_accept_configurations'],
        'findings': {
            'replan_reason_is_sticky': r4['finding_replan_reason_is_sticky'],
            'served_mask_vs_served_route_discrepancy_events':
                r2['served_mask_vs_route_discrepancy_events'],
        },
        'all_conclusions_definite': all(r['ok'] for r in RESULTS),
        'stop_condition_triggered': any(r['live_writes_detected'] for r in eq_rows),
        'not_done_this_round': ['real install write-back', 'modifying _commit',
                                'composed certifier taking over production decisions',
                                'cross-process state hash protocol',
                                'persistent P_after_install', 'training', 'effect experiment'],
    }
    for r in RESULTS:
        print('  %s %-5s %s' % ('PASS' if r['ok'] else 'FAIL', r['cid'], r['name']))
        if not r['ok']:
            print('        <- %s' % json.dumps(r['detail'], ensure_ascii=False, default=str)[:500])
    print('\nconclusions: %s' % json.dumps(out['conclusions'], ensure_ascii=False))
    print('decision points probed=%d heavy=%d' % (len(all_rows), len(heavy)))
    print('all_conclusions_definite:', out['all_conclusions_definite'])
    dest = (EXT.parent / 'research' / 'review_3080_20261010' / 'd3_g0_evidence_root_v1' /
            'B1_STATE_LEVEL_MATRIX.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + '\n',
                    encoding='utf-8', newline='\n')
    print('wrote', dest)
    return 0 if out['all_conclusions_definite'] else 1


if __name__ == '__main__':
    sys.exit(main())
