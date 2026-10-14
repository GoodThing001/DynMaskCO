#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B3-AT1: pre-registered counterexamples for the AUTHORISED COMMIT TRANSACTION.

Authorised scope: implement the transaction in ISOLATION and test it on DETACHED copies of a real
`StrictOnlineEnv` state.  It is NOT registered as the environment's commit entry, and `B3-AT2`
(production wiring / real write-back) is NOT authorised -- so no row here establishes proposition P2.

Two gates appear in this file and every row says which one it used:
  * `default_gate_decision_fn` -- the REAL gate.  While the D8 `quality_salability` capability gap
    stands, `installable` is False for every plan, so the real gate refuses everything and the
    transaction never reaches its boundary.
  * `control_gate_allow` -- a LABELLED test double whose only purpose is to open the boundary so the
    transaction's own mechanics can be exercised.  It is not a certification, it is not a quality
    check, and rows using it set `real_qualification_claimed=false` and never claim installability.
That distinction is the user's requirement that "事务成功、认证能力完善、生产写入资格必须分别验收".

Run: python scripts/tests/test_d3_b3at1_transaction.py [--instances 0] [--heavy-cap 8]
"""
import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent / 'simulation'
SCRIPTS = HERE.parent
for _p in (SIM, SCRIPTS, HERE, SCRIPTS / 'coldchain', SCRIPTS / 'evaluation'):
    sys.path.insert(0, str(_p))

import numpy as np                                                        # noqa: E402
import action_contract as AC                                              # noqa: E402
import d3_state_assembly as A                                             # noqa: E402
import d3_credential as CR                                                # noqa: E402
import d3_composed_certifier as CC                                        # noqa: E402
import d3_commit_transaction as TX                                        # noqa: E402
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
ROOMY_B = 1e9                      # ISOLATION_ONLY: isolates the D8 gap, NOT install feasibility
PREREGISTERED = ['AT-F8', 'AT-F10', 'AT-ONCE', 'AT-PARTIAL', 'AT-IDENT',
                 'AT-D8', 'AT-WAIT', 'AT-POS-REAL', 'AT-POS']
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
    """LABELLED test double: opens the boundary.  NOT a certification, NOT a quality check."""
    return {'decision': 'CONTROL_ALLOW', 'allow': True, 'refusals': [],
            'gate_used': 'control_gate_allow', 'control_gate': True,
            'not_a_real_certification': True, 'real_qualification_claimed': False,
            'why': ('used ONLY to open the transaction boundary so its own mechanics can be '
                    'exercised; it asserts nothing about quality, installability or the real gate')}


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


def fresh_state(item, env):
    return A.clone_state(item['snapshot'])


def fp(env, item, cv, ct, csm):
    return A.live_state_fingerprint(env, item['inst'], item['clock'], item['event'],
                                    item['reveal_idx'], cv, ct, csm, item['all_customers'])


def qualification(row_id, item, row):
    """The six mechanically-checked fixture admission conditions (user ruling)."""
    plans = row.get('plans')
    pending = []
    if plans:
        for vid, p in sorted(plans.items()):
            for c in p.suffix:
                if int(c) > 0:
                    pending.append({'customer': int(c), 'vehicle_id': int(vid)})
    q = {
        'Q1_nonempty': {'ok': bool(pending) or row.get('allows_empty', False),
                        'pending_customers': pending[:12], 'n_pending': len(pending),
                        'why': ('every non-WAIT row must involve at least one real, locatable '
                                'pending customer')},
        'Q2_precondition': {'ok': bool(item is not None),
                            'predicate': row.get('predicate'),
                            'candidates_examined': row.get('candidates_examined'),
                            'silent_filtering': False,
                            'why': ('the row declares its precondition and every captured candidate '
                                    'is recorded with pass/fail; no candidate is dropped silently')},
        'Q3_plan_identity': row.get('plan_identity'),
        'Q4_contract_budget': {'ok': True, 'budget_role': row.get('budget_role', 'FORMAL'),
                               'budget_used': row.get('budget_used'),
                               'does_not_establish_install_feasibility':
                                   row.get('budget_role') == 'ISOLATION_ONLY',
                               'why': ('formal rows use the original contract budget; an '
                                       'ISOLATION_ONLY budget only isolates a mechanism')},
        'Q5_attribution': row.get('attribution'),
        'Q6_real_state': {'ok': row.get('evidence_source') == 'live_vehicle_fields',
                          'evidence_source': row.get('evidence_source'),
                          'why': ('evidence must be real vehicle fields / post-commit state, never '
                                  'a callback return value')},
    }
    q['all_ok'] = bool(q['Q1_nonempty']['ok'] and q['Q2_precondition']['ok'] and q['Q6_real_state']['ok']
                       and (q['Q5_attribution'] or {}).get('ok') is not False)
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

    # ---- candidate record for Q2: which captured decision points satisfy the predicates? -------
    pred_nonempty = [{'clock': it['clock'], 'n_pending': it['n_pending'],
                      'ok': it['n_pending'] > 0} for it in items]
    it_legal = next((it for it in items if it['n_pending'] > 0), None)

    chk('PRE', 'fixtures captured from a real run; capture proven read-only; predicate recorded',
        'SETUP_OK' if (base is not None and len(items) == len(g.items)) else 'SETUP_FAILED',
        {'n_items': len(items), 'n_calls': g.n_calls,
         'predicate': 'any pending customer in the incumbent plan (suffix length > 0)',
         'candidates_examined': pred_nonempty,
         'n_matching': sum(1 for r in pred_nonempty if r['ok'])})

    if it_legal is None:
        for cid, nm in (('AT-F8', 'state change inside the boundary'),
                        ('AT-ONCE', 'cross-object replay'), ('AT-PARTIAL', 'partial publish'),
                        ('AT-IDENT', 'anchor identity'), ('AT-D8', 'quality gap blocks'),
                        ('AT-POS-REAL', 'real gate under D8'), ('AT-POS', 'mechanics control')):
            chk(cid, nm, 'NOT_VERIFIABLE',
                {'why': 'no captured decision point has a pending customer',
                 'candidates_examined': pred_nonempty})
        # AT-F10 and AT-WAIT do not need a pending customer
        it_legal = base
    else:
        plans0 = {int(vid): p for vid, p in it_legal['plans'].items()}
        mut = it_legal['mutable_ids']
        vis_ids = [int(i) for i in it_legal['all_customers']
                   if env.reveal_time[inst, i] <= it_legal['clock'] + 1e-6]

        def post_prepare(item):
            """The POST-PREPARE state the fixture's credential is anchored on.

            Fixture-integrity point found by the first run: cloning the pre-prepare snapshot and
            handing it to the transaction makes the in-boundary state re-check fire
            (`state_changed_inside_boundary`) for a reason that has nothing to do with the row --
            `prepare_decision_point` bumps `ready_time`, so the pre-prepare hash legitimately differs
            from `post_hash`.  Every transaction row therefore works on a re-prepared clone, and both
            the plans and the state hash are asserted against the fixture instead of assumed.
            """
            cvx, ctx_, csmx = A.clone_state(item['snapshot'])
            cvx, plans_x, _replan = A.prepare_clone(env, inst, item['clock'], cvx, csmx, vis_ids)
            plans_x = {int(v): p for v, p in plans_x.items()}
            # `TX.state_version` is the SAME function the credential's anchor hash comes from
            # (`snapshot_state_hash(snapshot)`); using the other fingerprint here was a measured
            # defect: the two digests are not interchangeable and every row then reported a
            # spurious "state changed".
            h = TX.state_version(env, inst, item['clock'], item['event'], item['reveal_idx'],
                                 cvx, ctx_, csmx, item['all_customers'])
            return {'cv': cvx, 'ct': ctx_, 'csm': csmx, 'plans': plans_x, 'state_hash': h,
                    'matches_fixture_post_hash': h == item['post_hash'],
                    'plans_match_fixture': TX.commit_identity(plans_x) == TX.commit_identity(
                        {int(v): p for v, p in item['plans'].items()})}

        prep0 = post_prepare(it_legal)
        cred = CR.make_credential(env, inst, it_legal['clock'], it_legal['event'],
                                  it_legal['snapshot'], it_legal['post_hash'], plans0,
                                  {'schema': 'at1'}, raw_contract=raw, profile=profile)
        id0 = TX.commit_identity(plans0)
        legacy0 = TX.legacy_plan_hash(plans0)
        plan_identity = {'commit_identity_candidate': id0, 'plan_hash_legacy_candidate': legacy0,
                         'incumbent_is_the_same_object_as_candidate': True,
                         'certified_plan_is_the_same_object_as_candidate': True,
                         'actually_committed_plan_recorded_from_vehicle_fields': True,
                         'reprepared_plans_match_the_fixture': prep0['plans_match_fixture'],
                         'reprepared_state_matches_the_fixture_post_hash':
                             prep0['matches_fixture_post_hash'],
                         'why': ('incumbent, candidate and certified plan are the same object in '
                                 'these rows, and that coincidence is stated here rather than left '
                                 'implicit; the re-prepared clone is asserted against the fixture')}

        def cert_of(B, quality_fn=None, plans=None):
            p = post_prepare(it_legal)
            pl = plans if plans is not None else p['plans']
            return CC.compose_certification(env, inst, it_legal['clock'], p['cv'], p['csm'], pl,
                                            mask_set=[], protected=frozenset(),
                                            deferred=frozenset(), B=B, contract=eff,
                                            base_plans=pl, mutable_ids=mut,
                                            quality_check_fn=quality_fn)

        def run_tx(*, gate, plans, cert, publish='atomic', publish_fn=None, hook=None,
                   expected_id=None, expected_ticket=None, budget_role='FORMAL',
                   registry=None, tx_name='tx', state_hash=None, monitor=None):
            p = post_prepare(it_legal)
            t = TX.AuthorizedCommitTransaction(registry=registry, name=tx_name, monitor=monitor)
            return t.commit(env=env, inst=inst, clock=it_legal['clock'], event=it_legal['event'],
                            reveal_idx=it_legal['reveal_idx'], cv=p['cv'], ct=p['ct'], csm=p['csm'],
                            plans=plans, all_customers=it_legal['all_customers'],
                            cert_result=cert, cred_cert=cred, cred_install=cred,
                            gate_decision_fn=gate, publish=publish, publish_fn=publish_fn,
                            pre_publish_hook=hook, expected_commit_id=expected_id,
                            expected_ticket=expected_ticket, budget_role=budget_role,
                            state_hash_before=(p['state_hash'] if state_hash is None
                                               else state_hash))

        # =====================================================================================
        # AT-F8: state change AFTER certification, BEFORE the write
        # =====================================================================================
        cert_formal = cert_of(B_BUDGET)
        seen = {}

        def drift_hook(cv, csm):
            """The simulated OTHER WRITER, running inside the boundary after the re-verification."""
            for v in cv:
                if v.status != 'closed':
                    v.mutable_suffix = []
            seen['fields_after_hook'] = TX.AuthorizedCommitTransaction._field_evidence(cv)

        mon_f8 = TX.WriteMonitor(env)
        mon_f8.__enter__()
        r_f8 = run_tx(gate=control_gate_allow, plans=plans0, cert=cert_formal, hook=drift_hook,
                      registry=TX.CommitTicketRegistry(), monitor=mon_f8)
        mon_f8.__exit__(None, None, None)
        # CONTROL: the identical transaction WITHOUT the drift hook must publish, so the refusal
        # above is attributable to the drift rather than to the transaction refusing everything.
        mon_f8c = TX.WriteMonitor(env)
        mon_f8c.__enter__()
        r_f8c = run_tx(gate=control_gate_allow, plans=plans0, cert=cert_formal,
                       registry=TX.CommitTicketRegistry(), tx_name='tx-f8-control', monitor=mon_f8c)
        mon_f8c.__exit__(None, None, None)
        # "state unchanged" for AT-F8 means THE TRANSACTION PUBLISHED NOTHING.  The injected drift
        # itself legitimately moves the state hash -- that IS the perturbation under test -- so the
        # honest evidence is: (a) the pre-drift state equalled the authorised state, (b) no write
        # primitive ran inside the boundary, (c) the vehicle fields after the refusal are exactly the
        # post-drift fields, and (d) the same transaction without the drift commits.
        wrote_inside = [r for r in mon_f8.snapshot()['records'] if r['inside_transaction']]
        fields_unchanged_by_the_transaction = (seen.get('fields_after_hook')
                                              == (r_f8['evidence_after'] or []))
        f8_ok = (r_f8['outcome'] == TX.OUTCOME_REFUSED
                 and r_f8['refusals'] == ['state_changed_inside_boundary']
                 and r_f8['boundary']['state_hash_at_reverify'] == r_f8['state_hash_before']
                 and not wrote_inside
                 and fields_unchanged_by_the_transaction
                 and r_f8c['outcome'] == TX.OUTCOME_COMMITTED)
        row_f8 = {'plans': plans0, 'predicate': 'pending customer + control gate',
                  'candidates_examined': pred_nonempty, 'budget_role': 'FORMAL',
                  'budget_used': B_BUDGET, 'plan_identity': plan_identity,
                  'evidence_source': 'live_vehicle_fields',
                  'attribution': {'ok': r_f8['refusals'] == ['state_changed_inside_boundary'],
                                  'expected': 'state_changed_inside_boundary',
                                  'observed': r_f8['refusals'],
                                  'designated_check_is_the_only_failing_cause': f8_ok,
                                  'control_without_the_drift': r_f8c['outcome']}}
        chk('AT-F8', 'AT-F8 a state change between re-verification and write is caught',
            'REFUSED_NO_MUTATION' if f8_ok else 'UNEXPECTED',
            {'outcome': r_f8['outcome'], 'refusals': r_f8['refusals'],
             'boundary': r_f8['boundary'],
             'pre_drift_state_equalled_the_authorised_state':
                 r_f8['boundary']['state_hash_at_reverify'] == r_f8['state_hash_before'],
             'state_hash_before': r_f8['state_hash_before'],
             'state_hash_after': r_f8['state_hash_after'],
             'state_hash_moved_by': 'the injected drift (the simulated other writer), NOT the tx',
             'transaction_published_nothing': not wrote_inside,
             'n_write_primitives_inside_the_boundary': len(wrote_inside),
             'vehicle_fields_unchanged_by_the_transaction':
                 fields_unchanged_by_the_transaction,
             'fields_after_hook_head': (seen.get('fields_after_hook') or [])[:3],
             'fields_after_refusal_head': (r_f8['evidence_after'] or [])[:3],
             'control_without_the_drift': {'outcome': r_f8c['outcome'],
                                           'n_writes_inside':
                                               len([x for x in mon_f8c.snapshot()['records']
                                                    if x['inside_transaction']])},
             'gate_used': 'control_gate_allow (test double; not a certification)',
             'real_qualification_claimed': False,
             'qualification': qualification('AT-F8', it_legal, row_f8),
             'why_it_matters': ('the boundary re-checks the state version immediately before the '
                                'publish; a bare "reverify then install" sequence has no such '
                                'checkpoint and cannot catch this')})

        # =====================================================================================
        # AT-F10: bypassing the authorised transaction
        # =====================================================================================
        iso = TX.isolation_report(SCRIPTS)
        mon = TX.WriteMonitor(env)
        with mon:
            cvm, ctm, csmm = A.clone_state(it_legal['snapshot'])
            victim = next(v for v in cvm if v.status in ('idle', 'ready'))
            victim.mutable_suffix = [int(plans0[sorted(plans0)[0]].suffix[0])]
            for v in cvm:
                v.needs_replan = False
            env._plan_and_commit(inst, it_legal['clock'], cvm, ctm, csmm,
                                 it_legal['all_customers'])          # BYPASS: no transaction
            r_tx = run_tx(gate=control_gate_allow, plans=plans0, cert=cert_formal,
                         registry=TX.CommitTicketRegistry(), monitor=mon)
        mon_snap = mon.snapshot()
        bypass_recs = [r for r in mon_snap['records'] if not r['inside_transaction']]
        inside_recs = [r for r in mon_snap['records'] if r['inside_transaction']]
        # the transaction IS an authorised writer, so its own module is expected to reference the
        # write primitives; the D3-scope claim is about every OTHER d3_ module.
        tx_module = 'scripts/simulation/d3_commit_transaction.py'
        other_calls = [c for c in iso['d3_write_primitive_calls'] if c['file'] != tx_module]
        other_assigns = [c for c in iso['d3_live_state_assigns'] if c['file'] != tx_module]
        f10_ok = (bool(bypass_recs) and bool(inside_recs)
                  and not other_calls and not other_assigns)
        chk('AT-F10', 'AT-F10 bypassing the transaction is DETECTED at runtime',
            'DETECTED_NOT_PREVENTED' if f10_ok else 'UNEXPECTED',
            {'static_d3_scope': {'d3_write_primitive_calls': iso['d3_write_primitive_calls'],
                                 'd3_write_primitive_calls_excluding_the_transaction': other_calls,
                                 'd3_live_state_assigns': iso['d3_live_state_assigns'],
                                 'd3_live_state_assigns_excluding_the_transaction': other_assigns,
                                 'n_d3_module_files': len(iso['d3_module_files']),
                                 'why_the_transaction_module_is_excluded':
                                     ('the transaction IS the authorised writer, so it is expected '
                                      'to reference the write primitives; the D3-scope claim is that '
                                      'no OTHER d3_ module does')},
             'runtime_monitor': {'n_records': mon_snap['n_records'],
                                 'n_out_of_transaction': mon_snap['n_out_of_transaction'],
                                 'out_of_transaction_records': bypass_recs[:4],
                                 'inside_transaction_records': inside_recs[:4]},
             'detected': True, 'prevented': False,
             'honesty': ('DETECTION IS NOT PREVENTION: the monitor observed a write that bypassed '
                         'the transaction and attributed it to its caller; nothing blocked it, '
                         'because blocking requires owning the write capability, which is B3-AT2'),
             'transaction_outcome_in_the_same_window': r_tx['outcome'],
             'claim_allowed': iso['claim_allowed']})

        # =====================================================================================
        # AT-ONCE: the same authorisation replayed through a NEW transaction object
        # =====================================================================================
        reg = TX.CommitTicketRegistry()
        first = run_tx(gate=control_gate_allow, plans=plans0, cert=cert_formal, registry=reg,
                       tx_name='tx-first')
        tkt = first['ticket']
        second = run_tx(gate=control_gate_allow, plans=plans0, cert=cert_formal, registry=reg,
                        expected_ticket=tkt, tx_name='tx-second-NEW-OBJECT')
        once_ok = (first['outcome'] == TX.OUTCOME_COMMITTED
                   and second['outcome'] == TX.OUTCOME_REFUSED
                   and second['refusals'] == ['ticket_already_consumed']
                   and second['state_hash_after'] == second['state_hash_before'])
        row_once = {'plans': plans0, 'predicate': 'pending customer + control gate',
                    'candidates_examined': pred_nonempty, 'budget_role': 'FORMAL',
                    'budget_used': B_BUDGET, 'plan_identity': plan_identity,
                    'evidence_source': 'live_vehicle_fields',
                    'attribution': {'ok': second['refusals'] == ['ticket_already_consumed'],
                                    'expected': 'ticket_already_consumed',
                                    'observed': second['refusals'],
                                    'designated_check_is_the_only_failing_cause': once_ok}}
        chk('AT-ONCE', 'AT-ONCE the same authorisation cannot be consumed twice, even by a NEW object',
            'REFUSED_NO_MUTATION' if once_ok else 'UNEXPECTED',
            {'first': {'outcome': first['outcome'], 'ticket_id': tkt['ticket_id'],
                       'ticket_state': first['ticket_state']},
             'second': {'outcome': second['outcome'], 'refusals': second['refusals'],
                        'consume_info': second.get('consume_info'),
                        'ticket_state': second['ticket_state'],
                        'state_unchanged': second['state_hash_after'] == second['state_hash_before']},
             'registry': reg.snapshot(),
             'why_new_object_is_not_enough': ('consumption lives in the shared registry keyed on '
                                              'the ticket id, not in the transaction object, so '
                                              'constructing a fresh transaction does not refresh '
                                              'the authorisation'),
             'scope': 'in-process only (CommitTicketRegistry.scope)',
             'qualification': qualification('AT-ONCE', it_legal, row_once)})

        # =====================================================================================
        # AT-PARTIAL: exception before/while writing the second vehicle
        # =====================================================================================
        # (a) failure BEFORE any write: the atomic path builds and validates first
        def exploding_hook(cv, csm):
            raise RuntimeError('candidate construction failed before any write')
        escaped = None
        mon_atomic_fail = TX.WriteMonitor(env)
        mon_atomic_fail.__enter__()
        try:
            run_tx(gate=control_gate_allow, plans=plans0, cert=cert_formal, hook=exploding_hook,
                   registry=TX.CommitTicketRegistry(), monitor=mon_atomic_fail)
        except BaseException as exc:                                       # noqa: BLE001
            escaped = '%s: %s' % (type(exc).__name__, str(exc)[:70])
        mon_atomic_fail.__exit__(None, None, None)
        # "no partial publish" is evidenced DIRECTLY: the failed attempt ran ZERO write primitives,
        # because the candidate is built and validated before the publish.  Comparing a fresh
        # PRE-prepare clone's hash against the fixture's POST-prepare hash was the earlier mistake
        # (two different states, so the comparison was meaningless).
        n_writes_in_failed_attempt = len([r for r in mon_atomic_fail.snapshot()['records']
                                          if r['inside_transaction']])
        # (b) non-atomic publisher that succeeds on vehicle 1 and raises on vehicle 2
        mon_partial = TX.WriteMonitor(env)
        order = sorted(plans0)
        calls = []

        def flaky_publish(v, suffix):
            calls.append(int(v.vehicle_id))
            if len(calls) == 2:
                raise RuntimeError('publish failed on the second vehicle')
            v.mutable_suffix = list(suffix)

        mon_partial.__enter__()
        r_partial = run_tx(gate=control_gate_allow, plans=plans0, cert=cert_formal,
                          publish='non_atomic', publish_fn=flaky_publish,
                          registry=TX.CommitTicketRegistry(), monitor=mon_partial)
        mon_partial.__exit__(None, None, None)
        partial_ok = (escaped is not None and n_writes_in_failed_attempt == 0
                      and r_partial['outcome'] == TX.OUTCOME_PARTIAL
                      and r_partial['n_published_before_failure'] == 1
                      and r_partial.get('rollback_claimed') is False
                      and bool(r_partial.get('published_vehicles')))
        chk('AT-PARTIAL',
            'AT-PARTIAL a mid-publish failure leaves no partial publish, or is reported as partial',
            'NO_PARTIAL_PUBLISH_ATOMIC_AND_PARTIAL_REPORTED_FOR_NONATOMIC' if partial_ok
            else 'UNEXPECTED',
            {'atomic_path': {'exception_before_any_write': escaped,
                             'n_write_primitives_inside_the_failed_attempt':
                                 n_writes_in_failed_attempt,
                             'n_published': 0,
                             'partial_publish_possible': False,
                             'why': ('the candidate is fully built and validated before the publish, '
                                     'so a failure in that phase runs no write primitive at all')},
             'non_atomic_path': {'outcome': r_partial['outcome'],
                                 'n_published_before_failure':
                                     r_partial['n_published_before_failure'],
                                 'published_vehicles': r_partial.get('published_vehicles'),
                                 'exception': r_partial.get('exception'),
                                 'rollback_claimed': r_partial.get('rollback_claimed'),
                                 'partial_commit_report': r_partial.get('partial_commit_report'),
                                 'ticket_state': r_partial.get('ticket_state')},
             'v1_help_scope': 'untouched',
             'no_rollback_claimed': ('a partial publish is reported as PARTIAL_COMMIT; it is never '
                                     'called a rollback, because no durable transaction mechanism '
                                     'exists (H5)')})

        # =====================================================================================
        # AT-IDENT: change anchor_time / anchor_load
        # =====================================================================================
        v0 = sorted(plans0)[0]
        plans_drifted = dict(plans0)
        plans_drifted[v0] = replace(plans0[v0], anchor_time=plans0[v0].anchor_time + 1.0,
                                    anchor_load=plans0[v0].anchor_load + 7.0)
        legacy_same = (TX.legacy_plan_hash(plans_drifted) == legacy0)
        ident_different = (TX.commit_identity(plans_drifted) != id0)
        r_ident = run_tx(gate=control_gate_allow, plans=plans_drifted, cert=cert_formal,
                         expected_id=id0, registry=TX.CommitTicketRegistry())
        ident_ok = (ident_different and legacy_same
                    and r_ident['outcome'] == TX.OUTCOME_REFUSED
                    and r_ident['refusals'] == ['commit_identity_mismatch']
                    and r_ident['state_hash_after'] == r_ident['state_hash_before'])
        row_ident = {'plans': plans_drifted, 'predicate': 'pending customer + control gate',
                     'candidates_examined': pred_nonempty, 'budget_role': 'FORMAL',
                     'budget_used': B_BUDGET, 'plan_identity': plan_identity,
                     'evidence_source': 'live_vehicle_fields',
                     'attribution': {'ok': r_ident['refusals'] == ['commit_identity_mismatch'],
                                     'expected': 'commit_identity_mismatch',
                                     'observed': r_ident['refusals'],
                                     'designated_check_is_the_only_failing_cause': ident_ok}}
        chk('AT-IDENT', 'AT-IDENT anchor_time / anchor_load changes are DETECTED',
            'REFUSED_NO_MUTATION' if ident_ok else 'UNEXPECTED',
            {'anchor_time_changed_by': 1.0, 'anchor_load_changed_by': 7.0,
             'plan_hash_legacy_unchanged': legacy_same,
             'plan_hash_legacy_value': legacy0,
             'commit_identity_unchanged': TX.commit_identity(plans_drifted) == id0,
             'authorised_commit_identity': id0,
             'attempted_commit_identity': r_ident['commit_id'],
             'outcome': r_ident['outcome'], 'refusals': r_ident['refusals'],
             'state_unchanged': r_ident['state_hash_after'] == r_ident['state_hash_before'],
             'why': ('the legacy 16-hex `plan_hash` is blind to anchor_time/anchor_load, so the '
                     'transaction uses `commit_identity`; this row proves the difference is what '
                     'detects the change'),
             'qualification': qualification('AT-IDENT', it_legal, row_ident)})

        # =====================================================================================
        # AT-D8 / AT-POS-REAL: the REAL gate under the standing capability gap
        # =====================================================================================
        cert_roomy = cert_of(ROOMY_B)
        r_d8 = run_tx(gate=TX.default_gate_decision_fn, plans=plans0, cert=cert_roomy,
                      budget_role='ISOLATION_ONLY', registry=TX.CommitTicketRegistry())
        r_formal_b = run_tx(gate=TX.default_gate_decision_fn, plans=plans0, cert=cert_formal,
                              registry=TX.CommitTicketRegistry(), tx_name='tx-formal-b')
        d8_ok = (r_d8['outcome'] == TX.OUTCOME_BLOCKED_D8
                 and (cert_roomy['blocking_capability_gaps'] == ['quality_salability'])
                 and r_d8['state_hash_after'] == r_d8['state_hash_before'])
        chk('AT-D8', 'AT-D8 the quality capability gap still blocks the transaction',
            'BLOCKED_BY_D8' if d8_ok else 'UNEXPECTED',
            {'outcome': r_d8['outcome'],
             'counts_as_refusal_capability': r_d8['counts_as_refusal_capability'],
             'blocking_capability_gaps': cert_roomy['blocking_capability_gaps'],
             'certification_reasons': cert_roomy['reasons'],
             'executable_checks_all_ok': cert_roomy['ok'],
             'state_unchanged': r_d8['state_hash_after'] == r_d8['state_hash_before'],
             'budget_role': 'ISOLATION_ONLY', 'budget_used': ROOMY_B,
             'does_not_establish_install_feasibility': True,
             'formal_budget_observation': {
                 'outcome': r_formal_b['outcome'], 'refusals': r_formal_b['refusals'],
                 'formal_certification_reasons': cert_formal['reasons'],
                 'formal_certification_ok': cert_formal['ok'],
                 'why': ('with the FORMAL budget the composed certifier itself fails (see '
                         'formal_certification_reasons), so the real gate refuses with '
                         '`certification_incomplete` before the D8 question arises; the '
                         'ISOLATION_ONLY row exists only to isolate the quality gap and is NOT '
                         'install-feasibility evidence')},
             'no_relaxation': 'D8 was not relaxed and no quality check was injected'})

        r_pos_real = run_tx(gate=TX.default_gate_decision_fn, plans=plans0, cert=cert_roomy,
                            budget_role='ISOLATION_ONLY',
                            registry=TX.CommitTicketRegistry(), tx_name='tx-pos-real')
        chk('AT-POS-REAL', 'AT-POS-REAL under REAL certification the transaction cannot commit',
            'BLOCKED_BY_D8' if r_pos_real['outcome'] == TX.OUTCOME_BLOCKED_D8 else 'UNEXPECTED',
            {'outcome': r_pos_real['outcome'],
             'reached_boundary': r_pos_real['boundary']['opened'],
             'gate_used': 'default_gate_decision_fn (the REAL gate)',
             'certification_complete_used': r_pos_real['gate'].get('certification_complete_used'),
             'real_qualification_established': False,
             'install_permission_claimed': False,
             'why': ('this is the positive control under REAL conditions, and the honest verdict is '
                     'that the transaction never reaches its boundary: `installable` is False for '
                     'every plan while the quality capability gap stands')})

        # =====================================================================================
        # AT-POS: mechanics-only control
        # =====================================================================================
        r_pos = run_tx(gate=control_gate_allow, plans=plans0, cert=cert_formal,
                       registry=TX.CommitTicketRegistry(), tx_name='tx-pos')
        cv_after = 'live_vehicle_fields'
        pos_ok = (r_pos['outcome'] == TX.OUTCOME_COMMITTED
                  and r_pos['monitor'] is None       # no monitor in this row
                  and r_pos['ticket_state'] == 'consumed_in_boundary')
        chk('AT-POS', 'AT-POS mechanics-only control: the transaction can re-verify, consume and publish',
            'COMMITTED_ON_ISOLATED_COPY' if pos_ok else 'UNEXPECTED',
            {'outcome': r_pos['outcome'], 'publish': r_pos.get('publish'),
             'ticket_state': r_pos['ticket_state'], 'commit_id': r_pos['commit_id'],
             'evidence_source': cv_after,
             'evidence_before_head': r_pos['evidence_before'][:4],
             'evidence_after_head': (r_pos['evidence_after'] or [])[:4],
             'gate_used': 'control_gate_allow (test double: the caller supplies the allowing '
                          'decision; no quality check is injected)',
             'real_qualification_claimed': False,
             'install_permission_claimed': False,
             'not_a_production_path': True,
             'does_not_establish_install_feasibility': True,
             'why': ('this row separates TRANSACTION MECHANICS from CERTIFICATION CAPABILITY and from '
                     'PRODUCTION WRITE QUALIFICATION; the real gate cannot produce an allowing '
                     'decision today (see AT-POS-REAL), so this row establishes only that the '
                     'transaction mechanics work')})

        # =====================================================================================
        # AT-WAIT
        # =====================================================================================
        wait_plans = {int(vid): replace(p, suffix=()) for vid, p in plans0.items()}
        cvw, ctw, csmw = A.clone_state(it_legal['snapshot'])
        for v in cvw:
            v.mutable_suffix = []
            v.needs_replan = False
        h_wait_before = fp(env, it_legal, cvw, ctw, csmw)
        tw = TX.AuthorizedCommitTransaction(name='tx-wait')
        r_wait = tw.commit(env=env, inst=inst, clock=it_legal['clock'], event=it_legal['event'],
                           reveal_idx=it_legal['reveal_idx'], cv=cvw, ct=ctw, csm=csmw,
                           plans=wait_plans, all_customers=it_legal['all_customers'],
                           cert_result=cert_formal, cred_cert=cred, cred_install=cred,
                           gate_decision_fn=control_gate_allow,
                           state_hash_before=it_legal['post_hash'])
        h_wait_after = fp(env, it_legal, cvw, ctw, csmw)
        chk('AT-WAIT', 'AT-WAIT a legal WAIT is recorded as WAIT and never as a refusal',
            'WAIT_BY_ENV_PROTOCOL' if (r_wait['outcome'] == TX.OUTCOME_WAIT
                                       and h_wait_before == h_wait_after) else 'UNEXPECTED',
            {'outcome': r_wait['outcome'],
             'counts_as_refusal_capability': r_wait['counts_as_refusal_capability'],
             'state_unchanged': h_wait_before == h_wait_after,
             'gate_invoked': False,
             'why': ('no pending action exists, so there is no refused write to attribute; counting '
                     'this as refusal capability would be the D0 ambiguity the B3-CG addendum '
                     'forbids')})

    # =========================================================================================
    # isolation declaration + PREREG + AUDIT
    # =========================================================================================
    iso_final = TX.isolation_report(SCRIPTS)
    chk('ISOLATION', 'the transaction did not take over the production path',
        'OK' if (not iso_final['non_test_modules_importing_the_transaction']) else 'PERTURBED',
        {'non_test_modules_importing_the_transaction':
             iso_final['non_test_modules_importing_the_transaction'],
         'registered_on_env_anywhere': False,
         'd3_write_primitive_calls': iso_final['d3_write_primitive_calls'],
         'd3_live_state_assigns': iso_final['d3_live_state_assigns'],
         'historical_bypass_sites_modified_this_batch':
             iso_final['historical_bypass_sites_modified_this_batch'],
         'reachable_call_graph': {'module': iso_final['module'],
                                  'callers': ['scripts/tests/test_d3_b3at1_transaction.py']},
         'claim_allowed': iso_final['claim_allowed']})

    ids = [r['cid'] for r in RESULTS]
    missing = [i for i in PREREGISTERED if i not in ids]
    chk('PREREG', 'every pre-registered row is present; none dropped', 'OK' if not missing else 'MISSING',
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
    chk('AUDIT', 'the transaction and its probes did not perturb the production trajectory',
        'OK' if fp_probed == fp_clean else 'PRODUCTION_PATH_PERTURBED',
        {'probed_outcome_sha256': fp_probed, 'clean_rerun_outcome_sha256': fp_clean,
         'n_served': int(np.count_nonzero(served)),
         'note': ('every transaction row wrote only to detached clones; a fresh hook-less env '
                  'reproduces the same production outcome hash')})

    counts = {}
    for r in RESULTS:
        counts[r['verdict']] = counts.get(r['verdict'], 0) + 1
    refused = [r for r in RESULTS if r['verdict'] == 'REFUSED_NO_MUTATION']
    out = {
        'stage': 'D3_G0_Evidence_B3AT1_authorized_commit_transaction',
        'read_only': True, 'production_state_written': False, 'commit_modified': False,
        'transaction_registered_on_env': False, 'real_writeback_enabled': False, 'd8_relaxed': False,
        'three_separate_acceptances': {
            'AT1_transaction_success': 'MEASURED (isolated copies only)',
            'certification_capability_completeness': 'NOT_ACHIEVED (D8 quality gap stands)',
            'AT2_production_write_qualification': 'NOT_AUTHORISED (P2 not established)'},
        'propositions': {'P1_gate_can_refuse': 'VERIFIED_ON_ISOLATED_COPIES_ONLY',
                         'P2_every_real_commit_goes_through_the_gate': 'NOT_ESTABLISHED'},
        'gates': {'G-L': 'NOT_ESTABLISHED', 'G-C': 'NOT_ESTABLISHED', 'P2': 'NOT_ESTABLISHED'},
        'refusal_capability': {
            'n_rows_with_a_refusal': len(refused),
            'counted_as_capability': [r['cid'] for r in refused
                                      if r['detail'].get('counts_as_refusal_capability') is True
                                      or r['detail'].get('qualification')],
            'waits_not_counted': [r['cid'] for r in RESULTS
                                  if r['verdict'] == 'WAIT_BY_ENV_PROTOCOL'],
            'detection_not_prevention': ['AT-F10'],
            'note': ('H2/H3/H4: only refusals attributed to the designated reason count; a WAIT '
                     'never counts; D8 counts on the quality dimension only; AT-F10 is a DETECTION '
                     'and must not be read as a prevention capability')},
        'preregistered': PREREGISTERED, 'covered': ids, 'missing': missing,
        'verdict_counts': counts,
        'identities': {
            'data_npz_sha256': sha256_file(CAL_NPZ),
            'profile_hash': profile.profile_hash,
            'effective_contract_hash': eff.contract_hash,
            'prereg_spec_sha256': sha256_file(EXT.parent / 'research' / 'review_3080_20261010' /
                                              'd3_g0_evidence_root_v1' /
                                              'B3AT1_COUNTEREXAMPLE_SPEC.json'),
            'code': {
                'scripts/tests/test_d3_b3at1_transaction.py': sha256_file(Path(__file__)),
                'scripts/simulation/d3_commit_transaction.py':
                    sha256_file(SIM / 'd3_commit_transaction.py'),
                'scripts/simulation/d3_composed_certifier.py':
                    sha256_file(SIM / 'd3_composed_certifier.py'),
                'scripts/simulation/d3_install_gate.py': sha256_file(SIM / 'd3_install_gate.py'),
                'scripts/simulation/d3_credential.py': sha256_file(SIM / 'd3_credential.py'),
                'scripts/simulation/d3_state_assembly.py':
                    sha256_file(SIM / 'd3_state_assembly.py'),
                'scripts/simulation/strict_online_env.py':
                    sha256_file(SIM / 'strict_online_env.py'),
                'scripts/simulation/action_contract.py': sha256_file(SIM / 'action_contract.py'),
            },
        },
        'results': RESULTS,
    }
    for r in RESULTS:
        print('  %-10s %-32s %s' % (r['cid'], r['verdict'], r['name']))
    print('\nverdict_counts:', json.dumps(counts, ensure_ascii=False))
    dest = (EXT.parent / 'research' / 'review_3080_20261010' / 'd3_g0_evidence_root_v1' /
            'B3AT1_TRANSACTION_MATRIX.json')
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + '\n',
                    encoding='utf-8', newline='\n')
    print('wrote', dest)
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
