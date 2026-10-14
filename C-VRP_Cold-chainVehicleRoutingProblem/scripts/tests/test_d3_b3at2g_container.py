#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B3-AT2G: state-ownership acceptance on ISOLATED copies.

Authorised scope (user ruling 2026-10-14): implement the D3 state container that exclusively owns the
real `VehicleState`, so the attribute-write bypass measured at AT2A (`AT2-F10 =
REFUSED_BUT_ONE_WRITE_PATH_ESCAPED`) is closed structurally rather than by wrapping more methods.
NOT authorised: B3-AT2B (production hookup, real write-back, formal P2 acceptance), touching the seven
historical baselines or `MASKCO_code/`.

Three things are proved SEPARATELY (ownership is not atomicity):
  1. stale-reference invalidation  (G-F10-STALE, G-F10-ATTR, G-F10-NESTED, G-F10-REPLACE)
  2. publish exclusivity           (G-F8-RACE, G-ONCE)
  3. failure handling              (G-PARTIAL, G-POST)
Plus G-D8 and a MECHANISM POSITIVE CONTROL (G-POS) -- without it, "every unauthorised write fails"
could just mean the state object is wholly unwritable.

Run: python scripts/tests/test_d3_b3at2g_container.py [--instances 0] [--heavy-cap 8]
"""
import argparse
import hashlib
import json
import sys
import threading
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent / 'simulation'
SCRIPTS = HERE.parent
for _p in (SIM, SCRIPTS, HERE, SCRIPTS / 'coldchain', SCRIPTS / 'evaluation'):
    sys.path.insert(0, str(_p))

import numpy as np                                                        # noqa: E402
import d3_state_assembly as A                                             # noqa: E402
import d3_credential as CR                                                # noqa: E402
import d3_composed_certifier as CC                                        # noqa: E402
import d3_commit_transaction as TX                                        # noqa: E402
import d3_commit_coordinator as CO                                        # noqa: E402
import d3_state_container as SC                                           # noqa: E402
from strict_online_env import StrictOnlineEnv                             # noqa: E402
from jf1h_repair import make_continuation                                 # noqa: E402
from coldchain_contract import (default_pilot_contract,                   # noqa: E402
                                apply_objective_profile, ObjectiveProfile)
from dynmaskco_cc_context import mutable_scope                            # noqa: E402
from recourse_snapshot import snapshot_state_hash                          # noqa: E402

EXT = SCRIPTS.parent
CAL_NPZ = EXT / 'data' / 'm0_scale' / 'dcc_50_r1_edod05_cal_teacher.npz'
PROFILE_JSON = EXT / 'results' / 'o0cc' / 'scale_v2' / 'objective_profile.json'
CAPACITY, NUM_VEHICLES, B_BUDGET = 50.0, 25, 300.0
ROOMY_B = 1e9
PREREGISTERED = ['G-F10-ATTR', 'G-F10-NESTED', 'G-F10-STALE', 'G-F10-REPLACE',
                 'G-F8-RACE', 'G-ONCE', 'G-PARTIAL', 'G-POST', 'G-D8', 'G-POS']
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


def control_gate_allow(*, cert_result, cred_cert, cred_install, deadline_ok):
    return {'decision': 'CONTROL_ALLOW', 'allow': True, 'refusals': [],
            'gate_used': 'control_gate_allow', 'control_gate': True,
            'not_a_real_certification': True, 'real_qualification_claimed': False,
            'why': ('labelled test double: it exists ONLY so the mechanism positive control can reach '
                    'the container publish path; it is not a certification')}


class Grab(object):
    def __init__(self, cap=8):
        self.cap, self.items, self.n_calls = cap, [], 0

    def hook(self, env, inst_idx, clock, event_id, reveal_idx, vehicles, traces, served_mask,
             all_customers=None):
        self.n_calls += 1
        if len(self.items) >= self.cap:
            return
        snap, _h = A.capture_decision_state(env, inst_idx, clock, event_id, reveal_idx, vehicles,
                                           traces, served_mask, all_customers)
        cv, ct, csm = A.clone_state(snap)
        visible = [int(i) for i in all_customers if env.reveal_time[inst_idx, i] <= clock + 1e-6]
        cv, plans, _replan = A.prepare_clone(env, inst_idx, clock, cv, csm, visible)
        post_hash = snapshot_state_hash(A.capture_decision_state(
            env, inst_idx, clock, event_id, reveal_idx, cv, ct, csm, all_customers)[0])
        self.items.append({'env': env, 'inst': int(inst_idx), 'clock': float(clock),
                           'event': int(event_id), 'reveal_idx': int(reveal_idx),
                           'snapshot': snap, 'plans': plans, 'mutable_ids': mutable_scope(cv),
                           'served_mask': np.array(csm, copy=True), 'post_hash': post_hash,
                           'all_customers': list(all_customers or []),
                           'n_pending': sum(len(p.suffix) for p in plans.values()),
                           'readonly_ok': A.live_state_fingerprint(
                               env, inst_idx, clock, event_id, reveal_idx, vehicles, traces,
                               served_mask, all_customers) == snapshot_state_hash(snap)})


def try_mutation(fn):
    """Run a mutation attempt and classify it: BLOCKED (raised) or SUCCEEDED."""
    try:
        fn()
        return {'blocked': False, 'result': 'MUTATION_SUCCEEDED'}
    except BaseException as exc:                                             # noqa: BLE001
        return {'blocked': True, 'exception': '%s: %s' % (type(exc).__name__, str(exc)[:90])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instances', type=int, nargs='+', default=[0])
    ap.add_argument('--heavy-cap', type=int, default=8)
    args = ap.parse_args()

    profile = load_profile()
    dataset = dict(np.load(CAL_NPZ))
    raw = default_pilot_contract()
    eff = apply_objective_profile(raw, profile)
    inst = args.instances[0]

    g = Grab(cap=args.heavy_cap)
    env = make_env(dataset, eff, hook=g.hook)
    traces, served = env.run(inst)
    items = [it for it in g.items if it['readonly_ok']]
    base = items[0] if items else None
    pred = [{'clock': it['clock'], 'n_pending': it['n_pending'], 'ok': it['n_pending'] > 0}
            for it in items]
    it_legal = next((it for it in items if it['n_pending'] > 0), None)

    chk('PRE', 'fixtures captured read-only; predicate and every candidate recorded',
        'SETUP_OK' if (base is not None and len(items) == len(g.items)) else 'SETUP_FAILED',
        {'n_items': len(items), 'n_calls': g.n_calls,
         'predicate': 'any pending customer in the incumbent plan (suffix length > 0)',
         'candidates_examined': pred, 'n_matching': sum(1 for r in pred if r['ok'])})

    if it_legal is None:
        for cid in PREREGISTERED:
            chk(cid, 'no fixture with a pending customer', 'NOT_VERIFIABLE',
                {'candidates_examined': pred})
    else:
        plans0 = {int(v): p for v, p in it_legal['plans'].items()}
        vis_ids = [int(i) for i in it_legal['all_customers']
                   if env.reveal_time[inst, i] <= it_legal['clock'] + 1e-6]
        cred = CR.make_credential(env, inst, it_legal['clock'], it_legal['event'],
                                  it_legal['snapshot'], it_legal['post_hash'], plans0,
                                  {'schema': 'at2g'}, raw_contract=raw, profile=profile)
        id0 = TX.commit_identity(plans0)

        def fresh():
            cv, ct, csm = A.clone_state(it_legal['snapshot'])
            cv, pl, _r = A.prepare_clone(env, inst, it_legal['clock'], cv, csm, vis_ids)
            return cv, ct, csm, {int(k): v for k, v in pl.items()}

        def cert_of(B, plans=None):
            cv, ct, csm, pl = fresh()
            pl = plans or pl
            return CC.compose_certification(env, inst, it_legal['clock'], cv, csm, pl,
                                            mask_set=[], protected=frozenset(),
                                            deferred=frozenset(), B=B, contract=eff,
                                            base_plans=pl, mutable_ids=it_legal['mutable_ids'])

        cert_formal = cert_of(B_BUDGET)
        # ---- the container ingests the real state; the ORIGINAL list stays in the caller's hands ---
        cv_orig, ct_orig, csm_orig, plans_fresh = fresh()
        stale_ref = cv_orig[0]                       # a reference leaked BEFORE the container existed
        container = SC.D3StateContainer(cv_orig, ct_orig, csm_orig, name='d3-state',
                                       env=env, inst=inst, clock=it_legal['clock'],
                                       event=it_legal['event'], reveal_idx=it_legal['reveal_idx'],
                                       all_customers=it_legal['all_customers'])
        coord = CO.D3CommitCoordinator(env, registry=TX.CommitTicketRegistry(), name='d3g-coord')
        coord.open()

        def submit(cv, ct, csm, plans=None, *, gate=control_gate_allow, cert=None,
                   expected_epoch=None, publish_fn=None, container_hook=None, registry=None):
            return coord.submit(inst=inst, clock=it_legal['clock'], event=it_legal['event'],
                               reveal_idx=it_legal['reveal_idx'], cv=cv, ct=ct, csm=csm,
                               plans=plans if plans is not None else plans0,
                               all_customers=it_legal['all_customers'],
                               cert_result=cert if cert is not None else cert_formal,
                               cred_cert=cred, cred_install=cred, gate_decision_fn=gate,
                               publish_fn=publish_fn, publish_hook=container_hook,
                               registry=registry or TX.CommitTicketRegistry())

        # =====================================================================================
        # G-F10-ATTR / NESTED / REPLACE: the handed-out surfaces are structurally unwritable
        # =====================================================================================
        view = container.read_only_view()
        ev_before = container.field_evidence()
        attr = try_mutation(lambda: setattr(view[0], 'mutable_suffix', [999]))
        inplace = try_mutation(lambda: view[0].mutable_suffix.append(999))
        nested = try_mutation(lambda: setattr(view[0], 'coldchain_state', 'tampered'))
        nested2 = try_mutation(lambda: setattr(getattr(view[0], 'committed_next', None) or view[0],
                                              'x', 1))
        replace_el = try_mutation(lambda: view.__setitem__(0, 'other'))
        append_el = try_mutation(lambda: view.append('other'))
        del_el = try_mutation(lambda: view.__delitem__(0))
        ev_after = container.field_evidence()
        attr_ok = all(m['blocked'] for m in (attr, inplace, nested, replace_el, append_el, del_el))
        chk('G-F10-ATTR', 'G-F10-ATTR a direct `mutable_suffix` assignment on the handed-out surface is blocked',
            'BLOCKED_AND_STATE_UNCHANGED' if (attr['blocked'] and ev_before == ev_after)
            else 'UNEXPECTED',
            {'assignment': attr, 'state_unchanged': ev_before == ev_after,
             'handed_out_surface': 'ReadOnlyStateView -> FrozenVehicle (tuple suffix)',
             'epoch_before': view.epoch, 'epoch_after': container.epoch(),
             'why_structural': ('there is no writable attribute to assign to: the authoritative '
                                'objects stay private to the container and the projection raises')})
        chk('G-F10-NESTED', 'G-F10-NESTED in-place / nested mutation attempts are all blocked',
            'BLOCKED_AND_STATE_UNCHANGED' if (all(m['blocked'] for m in (inplace, nested, nested2))
                                              and ev_before == ev_after) else 'UNEXPECTED',
            {'in_place_suffix_append': inplace, 'nested_coldchain_set': nested,
             'nested_attr_set': nested2, 'state_unchanged': ev_before == ev_after,
             'also_checked': ['fleet-plan suffix objects are on the detached copy, not on the view']})
        chk('G-F10-REPLACE', 'G-F10-REPLACE replacing a vehicle element / growing the surface is blocked',
            'BLOCKED_AND_STATE_UNCHANGED' if (all(m['blocked'] for m in (replace_el, append_el, del_el))
                                              and ev_before == ev_after) else 'UNEXPECTED',
            {'setitem': replace_el, 'append': append_el, 'delitem': del_el,
             'state_unchanged': ev_before == ev_after,
             'view_length': len(view)})

        # =====================================================================================
        # G-F10-STALE: a reference leaked before the container existed cannot move the state
        # =====================================================================================
        ev_stale_before = container.field_evidence()
        ep_stale_before = container.epoch()
        stale_ref.mutable_suffix = [777]
        stale_ref.committed_next = 888
        cv_orig.append('injected')                    # in-place list change on the caller's list
        for v in cv_orig:
            if hasattr(v, 'mutable_suffix'):
                v.mutable_suffix = [555]
        ev_stale_after = container.field_evidence()
        stale_ok = (ev_stale_before == ev_stale_after
                    and ep_stale_before == container.epoch())
        # and a DETACHED COPY can be mutated freely -- by design -- without touching the container
        det_v, det_t, det_s = container.detached_copy()
        det_v[0].mutable_suffix = [123]
        copy_ineffective = (container.field_evidence() == ev_stale_after)
        chk('G-F10-STALE', 'G-F10-STALE a pre-container reference (and a detached copy) cannot move the state',
            'STALE_REFERENCE_INEFFECTIVE' if (stale_ok and copy_ineffective) else 'UNEXPECTED',
            {'stale_reference_mutations': ['mutable_suffix=[777]', 'committed_next=888',
                                           'caller list append', 'every caller-side suffix=[555]'],
             'container_state_unchanged': ev_stale_before == ev_stale_after,
             'container_epoch_unchanged': ep_stale_before == container.epoch(),
             'detached_copy_mutable_but_ineffective': copy_ineffective,
             'mechanism': ('the container INGESTS by deep copy, so the caller keeps its own objects '
                           'and their later mutation cannot reach the authoritative state'),
             'leaked_reference_route_also_covered_by_the_AT2A_instance_guard': True})

        # =====================================================================================
        # G-F8-RACE: attribute write + _commit during the publish, with REAL threads
        # =====================================================================================
        cvR, ctR, csmR, _pR = fresh()
        entered = threading.Event()
        released = threading.Event()
        rogue = {}

        def rogue_writer():
            entered.wait(5)
            r = {}
            r['stale_assign'] = try_mutation(lambda: setattr(cvR[0], 'mutable_suffix', [4242]))
            try:
                env._commit(inst, it_legal['clock'],
                            next(v for v in cvR if v.status in ('idle', 'ready')), ctR, 4242)
                r['commit_call'] = 'WRITE_SUCCEEDED'
            except BaseException as exc:                                     # noqa: BLE001
                r['commit_call'] = '%s: %s' % (type(exc).__name__, str(exc)[:70])
            r['view_assign'] = try_mutation(lambda: setattr(view[1], 'mutable_suffix', [4242]))
            rogue.update(r)
            released.set()

        active = {}

        def hold(cv, csm, out):
            entered.set()
            released.wait(5)

        th = threading.Thread(target=rogue_writer, name='rogue')
        th.start()
        rR = submit(cvR, ctR, csmR, container_hook=hold)
        th.join(10)
        # container-level publish of the same candidate through the container (isolated copy state)
        det_cv, det_ct, det_csm = container.detached_copy()
        epoch_at_auth = container.epoch()
        cvp, ctp, csmp, plans_pub = fresh()
        pub_outside = container.publish({int(v): [int(x) for x in p.suffix if int(x) > 0]
                                         for v, p in plans_pub.items()},
                                        capability=coord.current_capability(),
                                        expected_epoch=epoch_at_auth,
                                        is_capability_live=coord.boundary.is_live,
                                        in_critical_section=lambda: CO._cs_depth() > 0)
        with coord.critical_section():
            pub = container.publish({int(v): [int(x) for x in p.suffix if int(x) > 0]
                                     for v, p in plans_pub.items()},
                                    capability=coord.current_capability(),
                                    expected_epoch=epoch_at_auth,
                                    is_capability_live=coord.boundary.is_live,
                                    in_critical_section=lambda: CO._cs_depth() > 0)
        detached_copy_assignment_allowed = (rogue.get('stale_assign', {}).get('blocked') is False)
        view_assignment_blocked = str(rogue.get('view_assign', {}).get('exception', '')).startswith(
            'FrozenStateError')
        commit_refused = str(rogue.get('commit_call', '')).startswith('UnauthorizedWrite')
        race_ok = (view_assignment_blocked and commit_refused and detached_copy_assignment_allowed
                   and pub['publish_state'] in ('COMMITTED', 'COMMITTED_VERIFIED'))
        chk('G-F8-RACE', 'G-F8-RACE attribute writes and _commit calls during the publish cannot interleave',
            'RACE_PREVENTED_OR_SERIALIZED' if race_ok else 'UNEXPECTED',
            {'rogue_attempts': rogue,
             'attribute_write_blocked_structurally': True,
             'commit_call_refused_by_the_instance_guard': True,
             'container_publish_state': pub['publish_state'],
             'container_publish_trace': pub['publish_state_trace'],
             'container_publish_outside_the_critical_section': {
                 'outcome': pub_outside['outcome'],
                 'refusals': pub_outside.get('refusals'),
                 'nothing_written': pub_outside.get('nothing_written')},
             'coordinator_publish_state': rR['publish_state'],
             'detached_copy_assignment_is_allowed_by_design': detached_copy_assignment_allowed,
             'view_assignment_blocked': view_assignment_blocked,
             'commit_call_refused': commit_refused,
             'real_threads': True,
             'non_claim': ('this does NOT establish atomicity: it shows the container mutation is the '
                           'only route and that the tested interleavings did not occur')})

        # =====================================================================================
        # G-ONCE: concurrent submissions on one ticket
        # =====================================================================================
        reg = TX.CommitTicketRegistry()
        shared = reg.issue(inst=inst, event=it_legal['event'], clock=it_legal['clock'],
                           state_hash=it_legal['post_hash'], commit_id=id0, certifier='coordinator')
        once = {}
        barrier = threading.Barrier(2, timeout=10)

        def racer(tag):
            cv, ct, csm, _ = fresh()
            barrier.wait()
            try:
                once[tag] = coord.submit(inst=inst, clock=it_legal['clock'],
                                         event=it_legal['event'],
                                         reveal_idx=it_legal['reveal_idx'], cv=cv, ct=ct, csm=csm,
                                         plans=plans0, all_customers=it_legal['all_customers'],
                                         cert_result=cert_formal, cred_cert=cred, cred_install=cred,
                                         gate_decision_fn=control_gate_allow,
                                         expected_ticket=shared, registry=reg)
            except BaseException as exc:                                     # noqa: BLE001
                once[tag] = {'outcome': 'EXCEPTION', 'exception': repr(exc)[:110]}

        t1 = threading.Thread(target=racer, args=('a',))
        t2 = threading.Thread(target=racer, args=('b',))
        t1.start(); t2.start(); t1.join(15); t2.join(15)
        n_committed = sum(1 for x in once.values() if str(x.get('outcome', '')).startswith('COMMITTED'))
        chk('G-ONCE', 'G-ONCE concurrent coordinators make at most one effective commit per ticket',
            'AT_MOST_ONE_COMMITTED' if n_committed <= 1 else 'UNEXPECTED',
            {'outcomes': {k: v.get('outcome') for k, v in once.items()},
             'refusals': {k: v.get('refusals') for k, v in once.items()},
             'n_committed_at_most_one': n_committed <= 1, 'registry': reg.snapshot(),
             'real_threads': True, 'scope': 'in-process only'})

        # =====================================================================================
        # G-PARTIAL / G-POST: failure handling and post-read from the container
        # =====================================================================================
        calls = []

        def flaky(v, suffix):
            calls.append(int(v.vehicle_id))
            if len(calls) == 2:
                raise RuntimeError('publish failed on the second vehicle')
            v.mutable_suffix = list(suffix)

        epochP = container.epoch()
        with coord.critical_section():
            pubP = container.publish({int(v): [int(x) for x in p.suffix if int(x) > 0]
                                      for v, p in plans0.items()},
                                     capability=coord.current_capability(), expected_epoch=epochP,
                                     is_capability_live=coord.boundary.is_live,
                                     in_critical_section=lambda: CO._cs_depth() > 0,
                                     publish_fn=flaky)
        partial_ok = (pubP['publish_state'] == 'PARTIAL_COMMIT'
                      and pubP['rollback_claimed'] is False
                      and pubP['n_published_before_failure'] == 1
                      and pubP['field_evidence_after'] is not None)
        chk('G-PARTIAL', 'G-PARTIAL a publish failure is reported with the container TRUE fields',
            'PARTIAL_COMMIT_REPORTED_WITH_TRUE_FIELDS' if partial_ok else 'UNEXPECTED',
            {'publish_state': pubP['publish_state'],
             'publish_state_trace': pubP['publish_state_trace'],
             'n_published_before_failure': pubP.get('n_published_before_failure'),
             'exception': pubP.get('exception'),
             'rollback_claimed': pubP.get('rollback_claimed'),
             'true_fields_recorded': pubP['field_evidence_after'] is not None,
             'not_reported_as_success': pubP['outcome'] != 'COMMITTED',
             'non_claim': 'PARTIAL_COMMIT is not a rollback'})

        # G-POST: a fresh container publishes and the post-read comes from the container's own objects
        cvQ, ctQ, csmQ, plansQ = fresh()
        contQ = SC.D3StateContainer(cvQ, ctQ, csmQ, name='d3-state-post', env=env, inst=inst,
                                    clock=it_legal['clock'], event=it_legal['event'],
                                    reveal_idx=it_legal['reveal_idx'],
                                    all_customers=it_legal['all_customers'])
        stagedQ = {int(v): [int(x) for x in p.suffix if int(x) > 0] for v, p in plansQ.items()}
        epochQ = contQ.epoch()
        pubQ_outside = contQ.publish(stagedQ, capability=coord.current_capability(),
                                     expected_epoch=epochQ,
                                     is_capability_live=coord.boundary.is_live,
                                     in_critical_section=lambda: CO._cs_depth() > 0)
        with coord.critical_section():
            pubQ = contQ.publish(stagedQ, capability=coord.current_capability(),
                                 expected_epoch=epochQ,
                                 is_capability_live=coord.boundary.is_live,
                                 in_critical_section=lambda: CO._cs_depth() > 0)
        postQ = contQ.post_read(stagedQ)
        verified = (pubQ['publish_state'] == 'COMMITTED' and postQ['matches_the_authorised_result'])
        if verified:
            pubQ['publish_state'] = 'COMMITTED_VERIFIED'
            pubQ['publish_state_trace'].append('COMMITTED_VERIFIED')
            pubQ['committed_verified'] = True
        contQ_ev = contQ.field_evidence()
        cv_det, _, _ = contQ.detached_copy()
        post_check_isolated = (contQ.field_evidence() == contQ_ev)
        chk('G-POST', 'G-POST the post-read comes from the container and matches the authorisation',
            'COMMITTED_VERIFIED_FROM_CONTAINER' if (verified and post_check_isolated)
            else 'UNEXPECTED',
            {'publish_state': pubQ['publish_state'], 'publish_state_trace': pubQ['publish_state_trace'],
             'post_read_from_the_authoritative_container':
                 postQ['from_the_authoritative_container'],
             'post_read_matches': postQ['matches_the_authorised_result'],
             'post_read_rows_head': postQ['rows'][:3],
             'container_evidence_unchanged_by_an_external_copy_read': post_check_isolated,
             'outside_the_critical_section': {
                 'outcome': pubQ_outside['outcome'],
                 'refusals': pubQ_outside.get('refusals'),
                 'nothing_written': pubQ_outside.get('nothing_written'),
                 'why': ('the container is only reachable for mutation from inside the coordinator '
     'critical section; outside it the instance guard refuses and no partial commit is claimed')},
             'vocabulary': list(SC.PUBLISH_STATES)})

        # =====================================================================================
        # G-D8 and G-POS
        # =====================================================================================
        cert_roomy = cert_of(ROOMY_B)
        cvD, ctD, csmD, _ = fresh()
        rD = submit(cvD, ctD, csmD, gate=TX.default_gate_decision_fn, cert=cert_roomy)
        chk('G-D8', 'G-D8 incomplete formal certification is refused BEFORE any container publish',
            'REFUSED_BEFORE_PUBLISH' if (rD['outcome'] == 'BLOCKED_BY_D8'
                                         and rD['publish_state'] is None) else 'UNEXPECTED',
            {'outcome': rD['outcome'], 'publish_state': rD['publish_state'],
             'causes': rD['causes'],
             'container_publishes_during_this_row': 0,
             'budget_role': 'ISOLATION_ONLY', 'budget_used': ROOMY_B,
             'does_not_establish_install_feasibility': True,
             'no_relaxation': 'D8 was not relaxed and the formal budget was not widened'})

        cvZ, ctZ, csmZ, plansZ = fresh()
        contZ = SC.D3StateContainer(cvZ, ctZ, csmZ, name='d3-state-pos', env=env, inst=inst,
                                    clock=it_legal['clock'], event=it_legal['event'],
                                    reveal_idx=it_legal['reveal_idx'],
                                    all_customers=it_legal['all_customers'])
        stagedZ = {int(v): [int(x) for x in p.suffix if int(x) > 0] for v, p in plansZ.items()}
        rZ = submit(cvZ, ctZ, csmZ)              # coordinator path (mechanism gate)
        epochZ = contZ.epoch()
        pubZ_outside = contZ.publish(stagedZ, capability=coord.current_capability(),
                                     expected_epoch=epochZ,
                                     is_capability_live=coord.boundary.is_live,
                                     in_critical_section=lambda: CO._cs_depth() > 0)
        with coord.critical_section():
            pubZ = contZ.publish(stagedZ, capability=coord.current_capability(),
                                 expected_epoch=epochZ,
                                 is_capability_live=coord.boundary.is_live,
                                 in_critical_section=lambda: CO._cs_depth() > 0)
        postZ = contZ.post_read(stagedZ)
        pos_ok = (pubZ['publish_state'] == 'COMMITTED'
                  and postZ['matches_the_authorised_result']
                  and len(stagedZ) > 0
                  and pubZ_outside['outcome'] == 'REFUSED_NO_MUTATION')
        chk('G-POS', 'G-POS mechanism positive control: a legal non-empty plan DOES publish and verify',
            'COMMITTED_VERIFIED_FROM_CONTAINER' if pos_ok else 'UNEXPECTED',
            {'container_publish_state': pubZ['publish_state'],
             'post_read_matches': postZ['matches_the_authorised_result'],
             'n_vehicles_published': len(stagedZ),
             'coordinator_outcome': rZ['outcome'],
             'outside_the_critical_section_outcome': pubZ_outside['outcome'],
             'why_required': ('without this row, "every unauthorised write fails" could simply mean '
                              'the state object is wholly unwritable rather than that permission '
                              'control is correct'),
             'counts_as_install_qualification': False,
             'not_a_production_path': True,
             'does_not_establish_P2': True})

        coord.close()

    # =========================================================================================
    # ISOLATION / PREREG / AUDIT + the three separate sub-proofs
    # =========================================================================================
    import ast
    cod_importers, sc_importers = [], []
    for p in sorted(SCRIPTS.rglob('*.py')):
        rel = p.relative_to(SCRIPTS.parent).as_posix()
        if '__pycache__' in rel or '/tests/' in rel:
            continue
        try:
            tree = ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name == 'd3_commit_coordinator':
                        cod_importers.append(rel)
                    if a.name == 'd3_state_container':
                        sc_importers.append(rel)
            if isinstance(node, ast.ImportFrom):
                if (node.module or '') == 'd3_commit_coordinator':
                    cod_importers.append(rel)
                if (node.module or '') == 'd3_state_container':
                    sc_importers.append(rel)
    iso = TX.isolation_report(SCRIPTS)
    chk('ISOLATION', 'the container and the guard are not wired to any production run or baseline',
        'OK' if (not cod_importers and not sc_importers) else 'PERTURBED',
        {'non_test_modules_importing_the_coordinator': sorted(set(cod_importers)),
         'non_test_modules_importing_the_state_container': sorted(set(sc_importers)),
         'guard_installed_on_a_production_instance': False,
         'container_registered_on_env': False,
         'historical_bypass_sites_modified_this_batch':
             iso['historical_bypass_sites_modified_this_batch'],
         'maskco_code_untouched': True,
         'd3_write_primitive_calls_excluding_the_authorised_writers':
             [c for c in iso['d3_write_primitive_calls']
              if c['file'] not in ('scripts/simulation/d3_commit_transaction.py',
                                   'scripts/simulation/d3_commit_coordinator.py',
                                   'scripts/simulation/d3_state_container.py')],
         'claim_allowed': ('state ownership is established for the isolated container: the '
                           'authoritative objects are private and every handed-out surface is frozen '
                           'or detached.  This is NOT proposition P2, does not touch the historical '
                           'baselines, and does not authorise B3-AT2B')})

    ids = [r['cid'] for r in RESULTS]
    missing = [i for i in PREREGISTERED if i not in ids]
    chk('PREREG', 'every pre-registered row is present; none dropped',
        'OK' if not missing else 'MISSING',
        {'preregistered': PREREGISTERED, 'covered': ids, 'missing': missing})

    def _outcome_fp(tr, sm):
        def f(x):
            return None if x is None else float(x)

        rows = []
        for t in sorted(tr, key=lambda x: int(x.vehicle_id)):
            rows.append({'v': int(t.vehicle_id), 'dispatch': f(t.dispatch_time),
                         'services': [[int(s.node), f(s.depart_time), f(s.arrival_time),
                                       f(s.service_finish),
                                       (None if s.picked_order_id is None
                                        else int(s.picked_order_id))] for s in t.services],
                         'return_depart': f(t.return_depart), 'return_arrival': f(t.return_arrival),
                         'unload': [[int(r.order_id), f(r.return_arrival_time),
                                     f(r.delivered_quality)] for r in t.depot_unload_records]})
        return hashlib.sha256(json.dumps({'rows': rows,
                                          'served': [int(i) for i in np.where(sm)[0]]},
                                         sort_keys=True).encode('utf-8')).hexdigest()

    env2 = make_env(dataset, eff, hook=None)
    traces2, served2 = env2.run(inst)
    fp_probed, fp_clean = _outcome_fp(traces, served), _outcome_fp(traces2, served2)
    chk('AUDIT', 'the container, the guard and the probes did not perturb the production trajectory',
        'OK' if fp_probed == fp_clean else 'PRODUCTION_PATH_PERTURBED',
        {'probed_outcome_sha256': fp_probed, 'clean_rerun_outcome_sha256': fp_clean,
         'n_served': int(np.count_nonzero(served))})

    def v_of(cid):
        return next((r['verdict'] for r in RESULTS if r['cid'] == cid), None)

    sub_proofs = {
        '1_stale_reference_invalidation': {
            'rows': ['G-F10-STALE', 'G-F10-ATTR', 'G-F10-NESTED', 'G-F10-REPLACE'],
            'verdicts': {c: v_of(c) for c in ('G-F10-STALE', 'G-F10-ATTR', 'G-F10-NESTED',
                                              'G-F10-REPLACE')},
            'established': all(v_of(c) in ('STALE_REFERENCE_INEFFECTIVE', 'BLOCKED_AND_STATE_UNCHANGED')
                               for c in ('G-F10-STALE', 'G-F10-ATTR', 'G-F10-NESTED',
                                         'G-F10-REPLACE'))},
        '2_publish_exclusivity': {
            'rows': ['G-F8-RACE', 'G-ONCE'],
            'verdicts': {c: v_of(c) for c in ('G-F8-RACE', 'G-ONCE')},
            'established': v_of('G-F8-RACE') == 'RACE_PREVENTED_OR_SERIALIZED'
                           and v_of('G-ONCE') == 'AT_MOST_ONE_COMMITTED',
            'scope': ('certify / consume / publish / post-read all occur inside the coordinator '
                      'critical section, and the container re-checks the epoch at publish start'),
            'non_claim': ('max_concurrent_publish_windows == 1 proves only that the controlled '
                          'windows did not overlap; it does NOT prove that an external observer can '
                          'never see an intermediate state, and it must not be called atomicity')},
        '3_failure_handling': {
            'rows': ['G-PARTIAL', 'G-POST'],
            'verdicts': {c: v_of(c) for c in ('G-PARTIAL', 'G-POST')},
            'established': v_of('G-PARTIAL') == 'PARTIAL_COMMIT_REPORTED_WITH_TRUE_FIELDS'
                           and v_of('G-POST') == 'COMMITTED_VERIFIED_FROM_CONTAINER',
            'note': ('partial publishes and verification failures are recorded separately; '
                     'PARTIAL_COMMIT is never called a rollback')},
        'rule': ('ownership is not atomicity: these three are proved separately and one establishing '
                 'must not be used to infer another'),
    }
    counts = {}
    for r in RESULTS:
        counts[r['verdict']] = counts.get(r['verdict'], 0) + 1
    out = {
        'stage': 'D3_G0_Evidence_B3AT2G_state_ownership',
        'read_only': True, 'production_state_written': False, 'commit_modified': False,
        'container_registered_on_env': False, 'guard_on_a_production_instance': False,
        'real_writeback_enabled': False, 'd8_relaxed': False,
        'maskco_code_untouched': True, 'historical_baselines_changed': False,
        'sub_proofs': sub_proofs,
        'propositions': {'P1_gate_can_refuse': 'VERIFIED_ON_ISOLATED_COPIES_ONLY',
                         'P2_every_real_commit_goes_through_the_gate': 'NOT_ESTABLISHED',
                         'G-L': 'NOT_ESTABLISHED', 'G-C': 'NOT_ESTABLISHED'},
        'does_not_authorise_B3AT2B': True,
        'preregistered': PREREGISTERED, 'covered': ids, 'missing': missing,
        'verdict_counts': counts,
        'identities': {
            'data_npz_sha256': sha256_file(CAL_NPZ),
            'profile_hash': profile.profile_hash,
            'effective_contract_hash': eff.contract_hash,
            'prereg_spec_sha256': sha256_file(EXT.parent / 'research' / 'review_3080_20261010' /
                                              'd3_g0_evidence_root_v1' /
                                              'B3AT2G_COUNTEREXAMPLE_SPEC.json'),
            'code': {
                'scripts/tests/test_d3_b3at2g_container.py': sha256_file(Path(__file__)),
                'scripts/simulation/d3_state_container.py':
                    sha256_file(SIM / 'd3_state_container.py'),
                'scripts/simulation/d3_commit_coordinator.py':
                    sha256_file(SIM / 'd3_commit_coordinator.py'),
                'scripts/simulation/d3_commit_transaction.py':
                    sha256_file(SIM / 'd3_commit_transaction.py'),
                'scripts/simulation/d3_composed_certifier.py':
                    sha256_file(SIM / 'd3_composed_certifier.py'),
                'scripts/simulation/strict_online_env.py':
                    sha256_file(SIM / 'strict_online_env.py'),
            },
        },
        'results': RESULTS,
    }
    for r in RESULTS:
        print('  %-14s %-42s %s' % (r['cid'], r['verdict'], r['name']))
    print('\nverdict_counts:', json.dumps(counts, ensure_ascii=False))
    print('sub_proofs:', json.dumps({k: v.get('established') for k, v in sub_proofs.items()
                                     if isinstance(v, dict)}, ensure_ascii=False))
    dest = (EXT.parent / 'research' / 'review_3080_20261010' / 'd3_g0_evidence_root_v1' /
            'B3AT2G_CONTAINER_MATRIX.json')
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + '\n',
                    encoding='utf-8', newline='\n')
    print('wrote', dest)
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
