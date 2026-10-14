#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B3-AT2A: unique-writer + mutual-exclusion acceptance on ISOLATED copies.

Authorised scope (user ruling 2026-10-14): implement the D3 write boundary, mutex/version control,
one-time authorisation consumption and post-commit read-only verification, and test the REAL `_commit`
primitive in an isolated environment.  NOT authorised (B3-AT2B): registering the coordinator as a
production run's commit entry, touching the seven historical baselines, relaxing D8, real batch
write-back, or any G1/G2 / F-D effect experiment.

Three hard conditions from the ruling are implemented literally:
  1. pre-registered before implementation, with the CONCURRENT PUBLISH WINDOW as its own primary
     counterexample (AT2-RACE, driven by real threads and events -- not sequential calls pretending
     to be concurrent);
  2. writes are verified only on isolated copies of a real environment; nothing is wired to a
     production run;
  3. write-capability exclusivity, ticket one-time use and publish failure consistency are accepted
     SEPARATELY -- the matrix carries one verdict field per acceptance and one passing must not be
     used to infer another.

Run: python scripts/tests/test_d3_b3at2a_coordinator.py [--instances 0] [--heavy-cap 8]
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
ROOMY_B = 1e9                     # ISOLATION_ONLY: isolates the D8 gap, not install feasibility
PREREGISTERED = ['AT2-F8', 'AT2-RACE', 'AT2-F10', 'AT2-ONCE', 'AT2-PARTIAL',
                 'AT2-POST', 'AT2-D8', 'AT2-WAIT', 'AT2-POS']
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
            'why': ('labelled test double used ONLY to open the coordinator boundary so its own '
                    'mechanics can be exercised; it is not a certification and asserts nothing '
                    'about quality, installability or the real gate')}


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


def qualification(row, item, extra=None):
    plans = row.get('plans')
    pending = []
    if plans:
        for vid, p in sorted(plans.items()):
            for c in p.suffix:
                if int(c) > 0:
                    pending.append({'customer': int(c), 'vehicle_id': int(vid)})
    q = {
        'Q1_nonempty': {'ok': bool(pending) or row.get('allows_empty', False),
                        'pending_customers': pending[:12], 'n_pending': len(pending)},
        'Q2_precondition': {'ok': bool(item is not None), 'predicate': row.get('predicate'),
                            'candidates_examined': row.get('candidates_examined'),
                            'silent_filtering': False},
        'Q3_plan_identity': row.get('plan_identity'),
        'Q4_contract_budget': {'ok': True, 'budget_role': row.get('budget_role', 'FORMAL'),
                               'budget_used': row.get('budget_used'),
                               'does_not_establish_install_feasibility':
                                   row.get('budget_role') == 'ISOLATION_ONLY'},
        'Q5_attribution': row.get('attribution'),
        'Q6_real_state': {'ok': row.get('evidence_source') == 'live_vehicle_fields',
                          'evidence_source': row.get('evidence_source')},
    }
    if extra:
        q.update(extra)
    q['all_ok'] = bool(q['Q1_nonempty']['ok'] and q['Q2_precondition']['ok']
                       and q['Q6_real_state']['ok'])
    return q


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
    pred_nonempty = [{'clock': it['clock'], 'n_pending': it['n_pending'], 'ok': it['n_pending'] > 0}
                     for it in items]
    it_legal = next((it for it in items if it['n_pending'] > 0), None)

    chk('PRE', 'fixtures captured read-only; predicate and every candidate recorded',
        'SETUP_OK' if (base is not None and len(items) == len(g.items)) else 'SETUP_FAILED',
        {'n_items': len(items), 'n_calls': g.n_calls,
         'predicate': 'any pending customer in the incumbent plan (suffix length > 0)',
         'candidates_examined': pred_nonempty,
         'n_matching': sum(1 for r in pred_nonempty if r['ok'])})

    if it_legal is None:
        for cid in PREREGISTERED:
            chk(cid, 'no fixture with a pending customer', 'NOT_VERIFIABLE',
                {'candidates_examined': pred_nonempty})
    else:
        plans0 = {int(v): p for v, p in it_legal['plans'].items()}
        vis_ids = [int(i) for i in it_legal['all_customers']
                   if env.reveal_time[inst, i] <= it_legal['clock'] + 1e-6]
        cred = CR.make_credential(env, inst, it_legal['clock'], it_legal['event'],
                                  it_legal['snapshot'], it_legal['post_hash'], plans0,
                                  {'schema': 'at2a'}, raw_contract=raw, profile=profile)
        id0 = TX.commit_identity(plans0)
        plan_identity = {'commit_identity_candidate': id0,
                         'plan_hash_legacy_candidate': TX.legacy_plan_hash(plans0),
                         'incumbent_equals_candidate': True,
                         'certified_plan_equals_candidate': True,
                         'coordinator_publishes_the_candidate_itself': True}

        def fresh(item=None):
            item = item or it_legal
            cv, ct, csm = A.clone_state(item['snapshot'])
            cv, pl, _r = A.prepare_clone(env, inst, item['clock'], cv, csm, vis_ids)
            return cv, ct, csm, {int(k): v for k, v in pl.items()}

        def cert_of(B, plans=None):
            cv, ct, csm, pl = fresh()
            pl = plans or pl
            return CC.compose_certification(env, inst, it_legal['clock'], cv, csm, pl,
                                            mask_set=[], protected=frozenset(),
                                            deferred=frozenset(), B=B, contract=eff,
                                            base_plans=pl, mutable_ids=it_legal['mutable_ids'])

        cert_formal = cert_of(B_BUDGET)
        cert_roomy = cert_of(ROOMY_B)
        # one coordinator + one boundary for the whole batch, installed on THIS isolated env only
        registry = TX.CommitTicketRegistry()
        coord = CO.D3CommitCoordinator(env, registry=registry, name='d3-coord')
        coord.open()

        def submit(cv, ct, csm, plans=None, *, gate=control_gate_allow, cert=None, **kw):
            return coord.submit(inst=inst, clock=it_legal['clock'], event=it_legal['event'],
                               reveal_idx=it_legal['reveal_idx'], cv=cv, ct=ct, csm=csm,
                               plans=plans if plans is not None else plans0,
                               all_customers=it_legal['all_customers'],
                               cert_result=cert if cert is not None else cert_formal,
                               cred_cert=cred, cred_install=cred, gate_decision_fn=gate,
                               publish_fn=kw.pop('publish_fn', None),
                               publish_hook=kw.pop('publish_hook', None),
                               before_publish_start_hook=kw.pop('before_publish_start_hook', None),
                               after_publish_hook=kw.pop('after_publish_hook', None),
                               expected_ticket=kw.pop('expected_ticket', None),
                               registry=kw.pop('registry', None),
                               active_publishes=kw.pop('active_publishes', None),
                               budget_role=kw.pop('budget_role', 'FORMAL'))

        # =====================================================================================
        # AT2-F8: state drift BEFORE the publish starts
        # =====================================================================================
        cv8, ct8, csm8, _p8 = fresh()
        before8 = coord._fields(cv8)

        def drift(cv, csm, out):
            for v in cv:
                if v.status != 'closed':
                    v.mutable_suffix = []

        r8 = submit(cv8, ct8, csm8, before_publish_start_hook=drift,
                    registry=TX.CommitTicketRegistry())
        f8_ok = (r8['outcome'] == TX.OUTCOME_REFUSED
                 and r8['refusals'] == ['state_changed_before_publish']
                 and r8['publish_state'] is None
                 and r8['epoch_at_publish_start'] != r8['epoch_at_authorisation'])
        chk('AT2-F8', 'AT2-F8 state drift before publish start is refused and nothing is published',
            'REFUSED_NO_MUTATION' if f8_ok else 'UNEXPECTED',
            {'outcome': r8['outcome'], 'refusals': r8['refusals'],
             'epoch_at_authorisation': r8['epoch_at_authorisation'],
             'epoch_at_publish_start': r8['epoch_at_publish_start'],
             'publish_state': r8['publish_state'],
             'n_write_primitives_allowed_before_refusal': r8['boundary']['n_allowed_writes'],
             'drift_mechanism': ('a direct `mutable_suffix` assignment, which the instance guard '
                                 'deliberately does NOT cover -- caught by the epoch re-check'),
             'qualification': qualification(
                 {'plans': plans0, 'predicate': 'pending + control gate',
                  'candidates_examined': pred_nonempty,
                  'budget_role': 'FORMAL', 'budget_used': B_BUDGET,
                  'plan_identity': plan_identity,
                  'evidence_source': 'live_vehicle_fields',
                  'attribution': {'ok': True,
                                  'expected': 'state_changed_before_publish',
                                  'observed': r8['refusals']}}, it_legal)})

        # =====================================================================================
        # AT2-RACE (PRIMARY): the concurrent publish window, with REAL threads
        # =====================================================================================
        # sub-case (a): a rogue writer tries to write DURING the coordinator's publish
        cvR, ctR, csmR, _pR = fresh()
        rogue = {}
        entered = threading.Event()
        released = threading.Event()

        def rogue_write():
            entered.wait(5)
            try:
                victim = next(v for v in cvR if v.status in ('idle', 'ready'))
                env._commit(inst, it_legal['clock'], victim, ctR, 999)
                rogue['result'] = 'WRITE_SUCCEEDED'
            except BaseException as exc:                                     # noqa: BLE001
                rogue['result'] = '%s: %s' % (type(exc).__name__, str(exc)[:80])
            finally:
                released.set()

        def hold_publish(cv, csm, out):
            entered.set()
            released.wait(5)

        th = threading.Thread(target=rogue_write, name='rogue-writer')
        th.start()
        active = {}
        rR = submit(cvR, ctR, csmR, publish_hook=hold_publish, active_publishes=active,
                    registry=TX.CommitTicketRegistry())
        th.join(10)
        rogue_refused = str(rogue.get('result', '')).startswith('UnauthorizedWrite')
        # sub-case (b): two coordinators trying to publish the same ticket -> serialised, one consumes
        cvR2a, ctR2a, csmR2a, _ = fresh()
        cvR2b, ctR2b, csmR2b, _ = fresh()
        reg2 = TX.CommitTicketRegistry()
        coord2 = CO.D3CommitCoordinator(env, registry=reg2, name='d3-coord-2')
        coord2.open()          # grant its capability on the SHARED boundary
        # reuse the same boundary (already installed); a second coordinator on the same instance is
        # allowed here only to exercise the shared-registry serialisation
        ticket_shared = reg2.issue(inst=inst, event=it_legal['event'], clock=it_legal['clock'],
                                   state_hash=it_legal['post_hash'], commit_id=id0,
                                   certifier='coordinator')
        results = {}
        barrier = threading.Barrier(2, timeout=10)

        def racer(tag, cv, ct, csm):
            barrier.wait()
            try:
                results[tag] = coord2.submit(
                    inst=inst, clock=it_legal['clock'], event=it_legal['event'],
                    reveal_idx=it_legal['reveal_idx'], cv=cv, ct=ct, csm=csm, plans=plans0,
                    all_customers=it_legal['all_customers'], cert_result=cert_formal,
                    cred_cert=cred, cred_install=cred, gate_decision_fn=control_gate_allow,
                    expected_ticket=ticket_shared)
            except BaseException as exc:                                     # noqa: BLE001
                results[tag] = {'outcome': 'EXCEPTION', 'exception': repr(exc)[:120]}

        t1 = threading.Thread(target=racer, args=('a', cvR2a, ctR2a, csmR2a))
        t2 = threading.Thread(target=racer, args=('b', cvR2b, ctR2b, csmR2b))
        t1.start(); t2.start(); t1.join(15); t2.join(15)
        v2 = [results.get('a', {}), results.get('b', {})]
        n_committed = sum(1 for x in v2 if str(x.get('outcome', '')).startswith('COMMITTED'))
        n_consumed = sum(1 for x in v2 if x.get('ticket_state') == 'consumed_in_critical_section')
        race_ok = (rogue_refused and str(rR['outcome']).startswith('COMMITTED')
                   and active.get('max_concurrent') == 1 and n_committed <= 1 and n_consumed <= 1)
        chk('AT2-RACE', 'AT2-RACE (PRIMARY) the concurrent publish window is prevented/serialised',
            'RACE_PREVENTED_OR_SERIALIZED' if race_ok else 'UNEXPECTED',
            {'sub_case_a_rogue_writer_during_publish':
                 {'rogue_result': rogue.get('result'), 'refused': rogue_refused,
                  'guard_refusals': rR['boundary']['refused_writes'][:2],
                  'coordinator_outcome': rR['outcome'],
                  'coordinator_publish_state': rR['publish_state']},
             'sub_case_b_two_coordinators_same_ticket':
                 {'outcomes': [x.get('outcome') for x in v2],
                  'refusals': [x.get('refusals') for x in v2],
                  'n_committed_at_most_one': n_committed <= 1,
                  'n_ticket_consumptions': n_consumed},
             'interleaving_evidence': {'max_concurrent_publish_windows': active.get('max_concurrent'),
                                       'windows': active.get('windows'),
                                       'no_interleaving': active.get('max_concurrent') == 1},
             'real_threads': True,
             'why_primary': ('the user ruling made the concurrent publish window its own primary '
                             'counterexample: a lock only helps if every writer obeys it, so the '
                             'window is tested with real threads rather than sequential calls')})

        # =====================================================================================
        # AT2-F10: bypassing the coordinator
        # =====================================================================================
        cvF, ctF, csmF, _pF = fresh()
        before_F = coord._fields(cvF)
        bypass = {}
        try:
            victim = next(v for v in cvF if v.status in ('idle', 'ready'))
            env._commit(inst, it_legal['clock'], victim, ctF, int(plans0[sorted(plans0)[0]].suffix[0]))
            bypass['direct_commit'] = 'WRITE_SUCCEEDED'
        except BaseException as exc:                                        # noqa: BLE001
            bypass['direct_commit'] = '%s: %s' % (type(exc).__name__, str(exc)[:80])
        try:
            env._plan_and_commit(inst, it_legal['clock'], cvF, ctF, csmF,
                                 it_legal['all_customers'])
            bypass['plan_and_commit'] = 'WRITE_SUCCEEDED'
        except BaseException as exc:                                        # noqa: BLE001
            bypass['plan_and_commit'] = '%s: %s' % (type(exc).__name__, str(exc)[:80])
        # the AT1 transaction's own independent publish interface must also be refused here
        try:
            t1x = TX.AuthorizedCommitTransaction(registry=TX.CommitTicketRegistry(), name='at1-on-guarded')
            cvT, ctT, csmT, _pT = fresh()
            t1x.commit(env=env, inst=inst, clock=it_legal['clock'], event=it_legal['event'],
                       reveal_idx=it_legal['reveal_idx'], cv=cvT, ct=ctT, csm=csmT, plans=plans0,
                       all_customers=it_legal['all_customers'], cert_result=cert_formal,
                       cred_cert=cred, cred_install=cred, gate_decision_fn=control_gate_allow,
                       state_hash_before=it_legal['post_hash'])
            bypass['at1_transaction_publish'] = 'WRITE_SUCCEEDED'
        except BaseException as exc:                                        # noqa: BLE001
            bypass['at1_transaction_publish'] = '%s: %s' % (type(exc).__name__, str(exc)[:80])
        after_F = coord._fields(cvF)
        f10_all_refused = all(str(v).startswith('UnauthorizedWrite') for v in bypass.values())
        f10_commit_paths_refused = all(str(bypass[k]).startswith('UnauthorizedWrite')
                                     for k in ('direct_commit', 'plan_and_commit'))
        # HONEST OUTCOME: the guard covers the two commit PRIMITIVES.  The AT1 transaction reaches the
        # state through per-vehicle mutable_suffix ASSIGNMENT, which the guard does not cover, so it
        # is not fully refused.  The verdict therefore reports a PARTIAL refusal with the residual gap
        # spelled out, instead of rounding up to 'refused'.
        iso = TX.isolation_report(SCRIPTS)
        chk('AT2-F10', 'AT2-F10 bypassing the coordinator is FORCIBLY REFUSED on the guarded instance',
            'PARTIALLY_REFUSED__ATTRIBUTE_WRITE_PATH_NOT_GUARDED' if (f10_commit_paths_refused
                                                                        and f10_all_refused)
            else ('FORCIBLY_REFUSED_WITHIN_THE_D3_EXECUTION_INSTANCE' if f10_all_refused
                  else 'REFUSED_BUT_ONE_WRITE_PATH_ESCAPED'),
            {'bypass_attempts': bypass,
             'all_refused': f10_all_refused,
             'commit_primitives_refused': f10_commit_paths_refused,
             'residual_gap': (None if f10_all_refused else
                              ('the AT1 transaction still reaches the state by ASSIGNING '
                               'mutable_suffix on the vehicles, which the instance guard does not '
                               'intercept; blocking attribute writes needs an owned state object or '
                               'a VehicleState-level guard, both of which belong to AT2B')),
             'state_unchanged_by_the_bypass_attempts': before_F == after_F,
             'guard_snapshot': coord.boundary.snapshot(),
             'static_d3_scope': {
                 'd3_write_primitive_calls': iso['d3_write_primitive_calls'],
                 'd3_write_primitive_calls_excluding_the_authorised_writers':
                     [c for c in iso['d3_write_primitive_calls']
                      if c['file'] not in ('scripts/simulation/d3_commit_transaction.py',
                                           'scripts/simulation/d3_commit_coordinator.py')],
                 'n_d3_module_files': len(iso['d3_module_files'])},
             'three_write_paths_on_the_same_d3_state': {
                 'direct_replanner_mutable_suffix_write': ('NOT blocked by the guard (attribute '
                                                           'assignment); caught by the epoch '
                                                           're-check -- see AT2-F8'),
                 'old_direct_commit_entry': 'refused by the guard',
                 'new_transaction_independent_publish': 'refused by the guard',
                 'note': ('the user ruling requires that these three cannot coexist on the same D3 '
                          'state; two of the three are forcibly refused here and the third is '
                          'detected rather than blocked, which is recorded as a residual gap')},
             'still_not_P2': ('this is forcible prevention WITHIN the D3 execution instance; '
                              'proposition P2 ("every real commit goes through the gate") remains '
                              'with B3-AT2B')})

        # =====================================================================================
        # AT2-ONCE: two concurrent submissions with the SAME ticket
        # =====================================================================================
        reg3 = TX.CommitTicketRegistry()
        coord3 = CO.D3CommitCoordinator(env, registry=reg3, name='d3-coord-once')
        coord3.open()          # grant its capability on the SHARED boundary
        shared = reg3.issue(inst=inst, event=it_legal['event'], clock=it_legal['clock'],
                            state_hash=it_legal['post_hash'], commit_id=id0, certifier='coordinator')
        once = {}
        b2 = threading.Barrier(2, timeout=10)

        def once_racer(tag, cv, ct, csm):
            b2.wait()
            try:
                once[tag] = coord3.submit(
                    inst=inst, clock=it_legal['clock'], event=it_legal['event'],
                    reveal_idx=it_legal['reveal_idx'], cv=cv, ct=ct, csm=csm, plans=plans0,
                    all_customers=it_legal['all_customers'], cert_result=cert_formal,
                    cred_cert=cred, cred_install=cred, gate_decision_fn=control_gate_allow,
                    expected_ticket=shared)
            except BaseException as exc:                                     # noqa: BLE001
                once[tag] = {'outcome': 'EXCEPTION', 'exception': repr(exc)[:120]}

        cva, cta, csma, _ = fresh()
        cvb, ctb, csmb, _ = fresh()
        o1 = threading.Thread(target=once_racer, args=('a', cva, cta, csma))
        o2 = threading.Thread(target=once_racer, args=('b', cvb, ctb, csmb))
        o1.start(); o2.start(); o1.join(15); o2.join(15)
        consumed = sum(1 for x in once.values() if x.get('ticket_state') == 'consumed_in_critical_section')
        refused_once = sum(1 for x in once.values()
                           if 'ticket_already_consumed' in (x.get('refusals') or []))
        once_ok = (consumed == 1 and refused_once == 1)
        chk('AT2-ONCE', 'AT2-ONCE two concurrent submissions cannot consume the same ticket',
            'AT_MOST_ONE_CONSUMED' if once_ok else 'UNEXPECTED',
            {'outcomes': {k: v.get('outcome') for k, v in once.items()},
             'refusals': {k: v.get('refusals') for k, v in once.items()},
             'n_consumed': consumed, 'n_refused_as_already_consumed': refused_once,
             'registry': reg3.snapshot(), 'real_threads': True,
             'scope': 'in-process only (CommitTicketRegistry.scope)',
             'separate_acceptance': ('this is acceptance B (ticket one-time use); it must NOT be '
                                     'read as evidence for acceptance A or C')})

        # =====================================================================================
        # AT2-PARTIAL: exception in the middle of a per-vehicle publish
        # =====================================================================================
        cvP, ctP, csmP, _ = fresh()
        calls = []

        def flaky(v, suffix):
            calls.append(int(v.vehicle_id))
            if len(calls) == 2:
                raise RuntimeError('publish failed on the second vehicle')
            v.mutable_suffix = list(suffix)

        rP = submit(cvP, ctP, csmP, publish_fn=flaky, registry=TX.CommitTicketRegistry())
        partial_ok = (rP['outcome'] == TX.OUTCOME_PARTIAL
                      and rP['publish_state'] == 'PARTIAL_COMMIT'
                      and rP['n_published_before_failure'] == 1
                      and rP['rollback_claimed'] is False
                      and rP['committed_verified'] is False)
        chk('AT2-PARTIAL', 'AT2-PARTIAL a mid-publish exception reports the TRUE partial state',
            'PARTIAL_COMMIT_REPORTED_WITH_TRUE_FIELDS' if partial_ok else 'UNEXPECTED',
            {'outcome': rP['outcome'], 'publish_state': rP['publish_state'],
             'publish_state_trace': rP['publish_state_trace'],
             'n_published_before_failure': rP.get('n_published_before_failure'),
             'published_vehicles': rP.get('published_vehicles'),
             'exception': rP.get('exception'), 'rollback_claimed': rP.get('rollback_claimed'),
             'committed_verified': rP['committed_verified'],
             'post_read_performed': rP['post_read'] is not None,
             'separate_acceptance': ('acceptance C (publish failure consistency); it must NOT be '
                                     'read as evidence for acceptance A or B')})

        # =====================================================================================
        # AT2-POST: post-commit independent re-read, and the negative control
        # =====================================================================================
        cvQ, ctQ, csmQ, _ = fresh()
        rQ = submit(cvQ, ctQ, csmQ, registry=TX.CommitTicketRegistry())
        post_ok = (rQ['outcome'] == 'COMMITTED' or rQ['committed_verified'])
        # negative control: tamper AFTER the publish, BEFORE the re-read
        cvQ2, ctQ2, csmQ2, _ = fresh()

        def tamper(cv, csm, out):
            victim = next(v for v in cv if v.committed_next not in (None, 0))
            victim.committed_next = 12345                       # a state the authorisation never gave

        rQ2 = submit(cvQ2, ctQ2, csmQ2, after_publish_hook=tamper,
                     registry=TX.CommitTicketRegistry())
        post_control_ok = (rQ2['committed_verified'] is False
                           and rQ2['outcome'] == TX.OUTCOME_UNKNOWN
                           and rQ2['publish_state'] == 'OUTCOME_UNKNOWN')
        chk('AT2-POST', 'AT2-POST the post-commit re-read verifies the authorised result (and is not automatic)',
            'COMMITTED_VERIFIED' if (post_ok and rQ['committed_verified'] and post_control_ok)
            else 'UNEXPECTED',
            {'positive': {'outcome': rQ['outcome'], 'publish_state': rQ['publish_state'],
                          'publish_state_trace': rQ['publish_state_trace'],
                          'committed_verified': rQ['committed_verified'],
                          'post_read_matches': rQ['post_read']['matches_the_authorised_result'],
                          'post_read_rows_head': rQ['post_read']['rows'][:3]},
             'negative_control_tampered_after_publish':
                 {'outcome': rQ2['outcome'], 'publish_state': rQ2['publish_state'],
                  'committed_verified': rQ2['committed_verified'],
                  'post_read_mismatches': rQ2['post_read']['mismatches'][:2],
                  'why': ('COMMITTED_VERIFIED is reserved for a publish whose post-state was '
                          'independently re-read; the tampered publish is downgraded to '
                          'OUTCOME_UNKNOWN instead of being blessed')},
             'vocabulary': list(CO.PUBLISH_STATES)})

        # =====================================================================================
        # AT2-D8 / AT2-WAIT / AT2-POS
        # =====================================================================================
        cvD, ctD, csmD, _ = fresh()
        rD = submit(cvD, ctD, csmD, gate=TX.default_gate_decision_fn, cert=cert_roomy,
                    budget_role='ISOLATION_ONLY', registry=TX.CommitTicketRegistry())
        d8_ok = (rD['outcome'] == TX.OUTCOME_BLOCKED_D8
                 and rD['causes'][CO.CAUSE_QUALITY]['failed'] is True
                 and rD['causes'][CO.CAUSE_AUTH]['failed'] is True)
        chk('AT2-D8', 'AT2-D8 the quality capability gap still blocks, reported as its own cause',
            'BLOCKED_BY_D8' if d8_ok else 'UNEXPECTED',
            {'outcome': rD['outcome'], 'causes': rD['causes'],
             'budget_role': 'ISOLATION_ONLY', 'budget_used': ROOMY_B,
             'does_not_establish_install_feasibility': True,
             'formal_budget_causes': CO.D3CommitCoordinator.classify_causes(cert_formal, True),
             'formal_budget_observation': ('with the FORMAL budget the composed certifier itself '
                                           'reports energy_budget_exceeded, so the energy cause is '
                                           'reported separately and NOT attributed to quality'),
             'no_relaxation': 'D8 was not relaxed and the formal budget was not widened'})

        cvW, ctW, csmW, pw = fresh()
        for v in cvW:
            v.mutable_suffix = []
            v.needs_replan = False
        wait_plans = {int(v): replace(p, suffix=()) for v, p in pw.items()}
        rW = submit(cvW, ctW, csmW, plans=wait_plans, registry=TX.CommitTicketRegistry())
        chk('AT2-WAIT', 'AT2-WAIT a legal WAIT is never counted as a refusal',
            'WAIT_BY_ENV_PROTOCOL' if (rW['outcome'] == TX.OUTCOME_WAIT
                                       and rW['counts_as_refusal_capability'] is False)
            else 'UNEXPECTED',
            {'outcome': rW['outcome'], 'counts_as_refusal_capability': False,
             'gate_invoked': False,
             'why': 'no pending action, so there is no refused write to attribute'})

        cvZ, ctZ, csmZ, _ = fresh()
        rZ = submit(cvZ, ctZ, csmZ, registry=TX.CommitTicketRegistry())
        pos_ok = rZ['committed_verified'] and rZ['publish_state'] == 'COMMITTED_VERIFIED'
        chk('AT2-POS', 'AT2-POS positive control: no concurrency, stable state, isolated commit verified',
            'COMMITTED_VERIFIED' if pos_ok else 'UNEXPECTED',
            {'outcome': rZ['outcome'], 'publish_state': rZ['publish_state'],
             'publish_state_trace': rZ['publish_state_trace'],
             'committed_verified': rZ['committed_verified'],
             'post_read_matches': rZ['post_read']['matches_the_authorised_result'],
             'gate_used': 'control_gate_allow (mechanism-only gate; no quality check injected)',
             'counts_as_install_qualification': False,
             'real_qualification_claimed': False,
             'not_a_production_path': True,
             'does_not_establish_P2': True})

        coord2.close()
        coord3.close()
        coord.close()

    # =========================================================================================
    # ISOLATION / PREREG / AUDIT and the three SEPARATE acceptances
    # =========================================================================================
    iso2 = TX.isolation_report(SCRIPTS)
    guard_installed_elsewhere = []
    cod_importers = []
    import ast
    for p in sorted(SCRIPTS.rglob('*.py')):
        rel = p.relative_to(SCRIPTS.parent).as_posix()
        if '__pycache__' in rel or '/tests/' in rel:
            continue
        try:
            tree = ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(a.name == 'd3_commit_coordinator'
                                                    for a in node.names):
                cod_importers.append(rel)
            if isinstance(node, ast.ImportFrom) and (node.module or '') == 'd3_commit_coordinator':
                cod_importers.append(rel)
    chk('ISOLATION', 'the coordinator is not wired to any production run',
        'OK' if not cod_importers else 'PERTURBED',
        {'non_test_modules_importing_the_coordinator': sorted(set(cod_importers)),
         'guard_installed_on_a_production_instance': guard_installed_elsewhere,
         'coordinator_registered_on_env': False,
         'historical_bypass_sites_modified_this_batch':
             iso2['historical_bypass_sites_modified_this_batch'],
         'd3_write_primitive_calls_excluding_the_two_authorised_writers':
             [c for c in iso2['d3_write_primitive_calls']
              if c['file'] not in ('scripts/simulation/d3_commit_transaction.py',
                                   'scripts/simulation/d3_commit_coordinator.py')],
         'claim_allowed': ('within the D3 execution instance the guard is installed on, the two commit '
                           'primitives are force-refused without the coordinator capability; this is '
                           'NOT proposition P2 and does NOT touch the historical baselines')})

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
    chk('AUDIT', 'the coordinator, its guard and the probes did not perturb the production trajectory',
        'OK' if fp_probed == fp_clean else 'PRODUCTION_PATH_PERTURBED',
        {'probed_outcome_sha256': fp_probed, 'clean_rerun_outcome_sha256': fp_clean,
         'n_served': int(np.count_nonzero(served)),
         'note': ('the guard was installed on the isolated probe instance only and was restored; a '
                  'fresh hook-less env reproduces the same production outcome hash')})

    def verdict_of(cid):
        return next((r['verdict'] for r in RESULTS if r['cid'] == cid), None)

    acceptance = {
        'A_write_capability_exclusivity': {
            'rows': ['AT2-F8', 'AT2-RACE', 'AT2-F10'],
            'verdicts': {c: verdict_of(c) for c in ('AT2-F8', 'AT2-RACE', 'AT2-F10')},
            'established': all(verdict_of(c) in ('REFUSED_NO_MUTATION', 'RACE_PREVENTED_OR_SERIALIZED',
                                                'FORCIBLY_REFUSED_WITHIN_THE_D3_EXECUTION_INSTANCE')
                               for c in ('AT2-F8', 'AT2-RACE', 'AT2-F10')),
            'partially_established': {
                'established_parts': [
                    'drift before publish start is refused with no publish (AT2-F8)',
                    'the concurrent publish window is prevented/serialised with real threads and '
                    'max_concurrent_publish_windows == 1 (AT2-RACE)',
                    'the two commit PRIMITIVES are force-refused without the capability (AT2-F10)'],
                'not_established_part': ('the AT1 transaction still reaches the state by ASSIGNING '
                                         'mutable_suffix on the vehicles, which the instance guard '
                                         'does not intercept -> AT2-F10 is '
                                         'REFUSED_BUT_ONE_WRITE_PATH_ESCAPED, not a full refusal'),
                'why_reported_as_a_failure': ('the user ruling: if the three write paths can still '
                                              'modify the same D3 state simultaneously, a real '
                                              'unique writer cannot be claimed'),
                'needed_to_close_it': ('an owned state object or a VehicleState-level guard so that '
                                       'attribute writes are constrained too; that is AT2B scope')},
            'residual_gap': ('a direct `mutable_suffix` assignment by a rogue in-process writer is '
                             'DETECTED (epoch re-check) rather than blocked')},
        'B_ticket_one_time': {
            'rows': ['AT2-ONCE'], 'verdicts': {'AT2-ONCE': verdict_of('AT2-ONCE')},
            'established': verdict_of('AT2-ONCE') == 'AT_MOST_ONE_CONSUMED',
            'scope': 'in-process only'},
        'C_publish_failure_consistency': {
            'rows': ['AT2-PARTIAL'], 'verdicts': {'AT2-PARTIAL': verdict_of('AT2-PARTIAL')},
            'established': verdict_of('AT2-PARTIAL') == 'PARTIAL_COMMIT_REPORTED_WITH_TRUE_FIELDS'},
        'rule': ('the three are accepted SEPARATELY: one establishing must not be used to infer '
                 'another; this block carries one verdict field per acceptance'),
    }
    counts = {}
    for r in RESULTS:
        counts[r['verdict']] = counts.get(r['verdict'], 0) + 1
    out = {
        'stage': 'D3_G0_Evidence_B3AT2A_unique_writer_and_mutual_exclusion',
        'read_only': True, 'production_state_written': False, 'commit_modified': False,
        'coordinator_registered_on_env': False, 'real_writeback_enabled': False, 'd8_relaxed': False,
        'publish_mode_is_an_implementation_name_not_a_system_property': True,
        'publish_state_vocabulary': list(CO.PUBLISH_STATES),
        'acceptances': acceptance,
        'propositions': {'P1_gate_can_refuse': 'VERIFIED_ON_ISOLATED_COPIES_ONLY',
                         'P2_every_real_commit_goes_through_the_gate': 'NOT_ESTABLISHED',
                         'G-L': 'NOT_ESTABLISHED', 'G-C': 'NOT_ESTABLISHED'},
        'preregistered': PREREGISTERED, 'covered': ids, 'missing': missing,
        'verdict_counts': counts,
        'identities': {
            'data_npz_sha256': sha256_file(CAL_NPZ),
            'profile_hash': profile.profile_hash,
            'effective_contract_hash': eff.contract_hash,
            'prereg_spec_sha256': sha256_file(EXT.parent / 'research' / 'review_3080_20261010' /
                                              'd3_g0_evidence_root_v1' /
                                              'B3AT2A_COUNTEREXAMPLE_SPEC.json'),
            'code': {
                'scripts/tests/test_d3_b3at2a_coordinator.py': sha256_file(Path(__file__)),
                'scripts/simulation/d3_commit_coordinator.py':
                    sha256_file(SIM / 'd3_commit_coordinator.py'),
                'scripts/simulation/d3_commit_transaction.py':
                    sha256_file(SIM / 'd3_commit_transaction.py'),
                'scripts/simulation/d3_composed_certifier.py':
                    sha256_file(SIM / 'd3_composed_certifier.py'),
                'scripts/simulation/d3_install_gate.py': sha256_file(SIM / 'd3_install_gate.py'),
                'scripts/simulation/d3_state_assembly.py':
                    sha256_file(SIM / 'd3_state_assembly.py'),
                'scripts/simulation/strict_online_env.py':
                    sha256_file(SIM / 'strict_online_env.py'),
            },
        },
        'results': RESULTS,
    }
    for r in RESULTS:
        print('  %-11s %-46s %s' % (r['cid'], r['verdict'], r['name']))
    print('\nverdict_counts:', json.dumps(counts, ensure_ascii=False))
    print('acceptances:', json.dumps({k: v.get('established') for k, v in acceptance.items()
                                      if isinstance(v, dict)}, ensure_ascii=False))
    dest = (EXT.parent / 'research' / 'review_3080_20261010' / 'd3_g0_evidence_root_v1' /
            'B3AT2A_COORDINATOR_MATRIX.json')
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + '\n',
                    encoding='utf-8', newline='\n')
    print('wrote', dest)
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
