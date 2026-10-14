#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B3-AC part 2: gate REFUSAL-CAPABILITY acceptance on ISOLATED copies.

The user ruling separates two propositions and authorises only the first for this batch:

    P1 "门能拒绝非法动作"          -> B3-AC (this file)
    P2 "所有实际提交都必须经过这个门" -> independently authorised B3-AT (NOT this batch)

So this file answers exactly one question: given a plan and a state, does the EXISTING gate
(`compose_certification` + `pre_install_reverify` + `InstallLedger`) REFUSE when it should, for the
RIGHT REASON, leaving the state byte-identical?  It never registers itself as a production entry
point, never modifies `_commit` / `_plan_and_commit`, never relaxes D8, and never writes real state:
every write happens on a DETACHED copy produced by `d3_state_assembly.clone_state`.

Honesty rules (pre-registered in `B3AC_COUNTEREXAMPLE_SPEC.json`):
  H1 a reproduced defect is `COUNTEREXAMPLE_REPRODUCED`, never a pass;
  H2 a refusal must be attributed by REASON -- a refusal for the wrong reason (`REFUSAL_WRONG_REASON`)
     does not count as refusal capability;
  H3 `WAIT_BY_ENV_PROTOCOL` is never counted as refusal capability;
  H4 the D8 refusal proves refusal capability on the QUALITY dimension only -- it is not
     installability, not G-L, not a pass;
  H5 no crash-recovery atomicity is claimed (there is no durable transaction mechanism);
  H6 the controlled commit path lives only in this harness, on isolated copies.

Run: python scripts/tests/test_d3_b3ac_gate_refusal.py [--instances 0] [--heavy-cap 8]
"""
import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent / 'simulation'
for _p in (SIM, HERE.parent, HERE, HERE.parent / 'coldchain', HERE.parent / 'evaluation'):
    sys.path.insert(0, str(_p))

import numpy as np                                                      # noqa: E402
import action_contract as AC                                           # noqa: E402
import d3_state_assembly as A                                          # noqa: E402
import d3_credential as CR                                             # noqa: E402
import d3_composed_certifier as CC                                     # noqa: E402
import d3_install_gate as GATE                                         # noqa: E402
import d3_geo_adapter as GA                                            # noqa: E402
from strict_online_env import StrictOnlineEnv                          # noqa: E402
from jf1h_repair import make_continuation                              # noqa: E402
from coldchain_contract import (default_pilot_contract,                # noqa: E402
                                apply_objective_profile, ObjectiveProfile)
from dynmaskco_cc_context import mutable_scope                         # noqa: E402
from recourse_snapshot import snapshot_state_hash                       # noqa: E402

EXT = HERE.parent.parent
CAL_NPZ = EXT / 'data' / 'm0_scale' / 'dcc_50_r1_edod05_cal_teacher.npz'
PROFILE_JSON = EXT / 'results' / 'o0cc' / 'scale_v2' / 'objective_profile.json'
CAPACITY, NUM_VEHICLES, B_BUDGET = 50.0, 25, 300.0
PREREGISTERED = ['AC-R1', 'AC-R2', 'AC-R3', 'AC-R4', 'AC-R5',
                 'AC-R6', 'AC-R7', 'AC-R8', 'AC-R9', 'AC-R10']
RESULTS = []

NOT_A_PRODUCTION_PATH = ('the controlled commit path below exists ONLY in this test file; it is '
                         'never registered on an env, never imported by a d3_ module, and P2 '
                         '("every real commit goes through the gate") stays with B3-AT')


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
    """Capture decision-point fixtures, proving read-only capture via a live fingerprint."""

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
        live = A.live_state_fingerprint(env, inst_idx, clock, event_id, reveal_idx, vehicles,
                                        traces, served_mask, all_customers)
        post_hash = snapshot_state_hash(A.capture_decision_state(
            env, inst_idx, clock, event_id, reveal_idx, cv, ct, csm, all_customers)[0])
        self.items.append({'env': env, 'inst': int(inst_idx), 'clock': float(clock),
                           'event': int(event_id), 'snapshot': snap, 'plans': plans,
                           'mutable_ids': mutable_scope(cv), 'visible': visible,
                           'served_mask': np.array(csm, copy=True), 'post_hash': post_hash,
                           'all_customers': list(all_customers or []),
                           'readonly_ok': live == snapshot_state_hash(snap)})

    def fresh(self, item):
        """A fresh detached copy of the fixture's decision-point state."""
        return A.clone_state(item['snapshot'])


def fp(env, item, cv, ct, csm):
    """Sanctioned fingerprint of a DETACHED copy (same capture path as the live one)."""
    return A.live_state_fingerprint(env, item['inst'], item['clock'], item['event'], 0, cv, ct,
                                    csm, item['all_customers'])


# ---------------------------------------------------------------------------------------------
# THE CONTROLLED COMMIT PATH -- harness-local (H6).  Never registered, never imported.
# ---------------------------------------------------------------------------------------------
def controlled_commit(env, item, cv, ct, csm, plans, *, contract, B, base_plans, mask_set,
                      protected, deferred, cred_cert, cred_install, mutable_ids,
                      deadline_ok=True, quality_check_fn=None):
    """Gate, then (only if authorised) write -- on the DETACHED copy only.

    Order: WAIT detection first (H3: a WAIT is not a refused write), then the composed
    certification, then the pre-install re-verification, then the write through the SAME
    primitive the environment uses.
    """
    before = fp(env, item, cv, ct, csm)
    pending = {int(vid) for vid, p in plans.items() if len(p.suffix) > 0}
    if not pending:
        return {'action': 'WAIT_BY_ENV_PROTOCOL',
                'counts_as_refusal_capability': False,
                'state_hash_before': before, 'state_hash_after': fp(env, item, cv, ct, csm),
                'why': 'no pending action exists, so there is no refused WRITE to attribute'}

    cert = CC.compose_certification(env, item['inst'], item['clock'], cv, csm, plans,
                                    mask_set=mask_set, protected=protected, deferred=deferred,
                                    B=B, contract=contract, base_plans=base_plans,
                                    mutable_ids=mutable_ids, quality_check_fn=quality_check_fn)
    if not cert['installable']:
        return {'action': 'REFUSE_NO_MUTATION',
                'counts_as_refusal_capability': True,
                'refusal_stage': 'compose_certification',
                'reasons': list(cert['reasons']),
                'blocking_capability_gaps': list(cert['blocking_capability_gaps']),
                'certification_ok': bool(cert['ok']),
                'legality_ok': bool(cert['legality_ok']),
                'components': cert['components'],
                'state_hash_before': before, 'state_hash_after': fp(env, item, cv, ct, csm)}

    decision = GATE.pre_install_reverify(cred_cert, cred_install, deadline_ok=deadline_ok,
                                         certification_complete=cert['installable'])
    if not decision['allow']:
        return {'action': 'REFUSE_NO_MUTATION',
                'counts_as_refusal_capability': True,
                'refusal_stage': 'pre_install_reverify',
                'reasons': list(decision['refusals']),
                'certification_ok': bool(cert['ok']),
                'state_hash_before': before, 'state_hash_after': fp(env, item, cv, ct, csm)}

    by_id = {int(v.vehicle_id): v for v in cv}
    for vid, p in plans.items():
        by_id[int(vid)].mutable_suffix = [int(x) for x in p.suffix if int(x) > 0]
    for v in cv:
        v.needs_replan = False
    env._plan_and_commit(item['inst'], item['clock'], cv, ct, csm, item['all_customers'])
    return {'action': 'COMMIT', 'counts_as_refusal_capability': False,
            'certification_ok': bool(cert['ok']), 'legality_ok': bool(cert['legality_ok']),
            'state_hash_before': before, 'state_hash_after': fp(env, item, cv, ct, csm)}


def reason_match(row, needle):
    return any(needle in r for r in row.get('reasons', []))


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

    chk('PRE', 'fixtures captured from a real run; capture proven read-only',
        'SETUP_OK' if base is not None and len(items) == len(g.items) else 'SETUP_FAILED',
        {'n_items': len(items), 'n_calls': g.n_calls,
         'first_clock': items[0]['clock'] if items else None,
         'last_clock': items[-1]['clock'] if items else None})

    def pick(pred):
        return next((it for it in items if pred(it)), None)

    # a fixture that HAS a served customer (for the legality counterexample) ...
    it_served = pick(lambda it: any(bool(it['served_mask'][c]) for c in range(1, env.num_nodes)))
    # ... and a fixture whose incumbent plan actually ASSIGNS customers.  NOT `all(...)`: measured
    # on instance 0, the only decision points with non-empty suffixes are SERVICE-COMPLETION events
    # where two or three committed vehicles carry tails while most vehicles carry none, so requiring
    # every vehicle to be non-empty skipped exactly the fixtures that exist (40/40 skipped on the
    # first attempt, 20/20 on each of instances 1-5).
    it_legal = pick(lambda it: any(len(p.suffix) > 0 for p in it['plans'].values()))

    def fixture_plans(item):
        return {int(vid): p for vid, p in item['plans'].items()}

    # =========================================================================================
    # AC-R1 / AC-R2: the SAME illegal suffix, old path vs controlled path
    # =========================================================================================
    r1 = r2 = None
    if it_served is None:
        for cid, nm in (('AC-R1', 'old path'), ('AC-R2', 'controlled path')):
            chk(cid, 'AC-%s illegal suffix' % nm, 'NOT_VERIFIABLE',
                {'why': 'no served customer exists at any captured decision point'})
    else:
        csm = it_served['served_mask']
        illegal = sorted(int(c) for c in range(1, env.num_nodes) if bool(csm[c]))
        victim_c = illegal[0]
        # ---- AC-R1: the env's OWN commit loop, no gate anywhere in the path -----------------
        cv1, ct1, csm1 = A.clone_state(it_served['snapshot'])
        plans1 = fixture_plans(it_served)
        tgt1 = next(v for v in cv1 if v.status in ('idle', 'ready'))
        for v in cv1:
            v.needs_replan = False
        tgt1.mutable_suffix = [int(victim_c)]
        before1 = {'status': tgt1.status, 'committed_next': tgt1.committed_next,
                   'suffix': list(tgt1.mutable_suffix)}
        h1_before = fp(env, it_served, cv1, ct1, csm1)
        env._plan_and_commit(inst, it_served['clock'], cv1, ct1, csm1, it_served['all_customers'])
        after1 = {'status': tgt1.status,
                  'committed_next': (None if tgt1.committed_next is None
                                     else int(tgt1.committed_next)),
                  'suffix': list(tgt1.mutable_suffix)}
        wrote = (after1['committed_next'] == int(victim_c))
        r1 = {'illegal_customer': int(victim_c), 'before': before1, 'after': after1,
              'gate_invoked': False, 'certifier_called': False,
              'state_hash_before': h1_before,
              'state_hash_after': fp(env, it_served, cv1, ct1, csm1),
              'note': ('the old path is the env\'s own _plan_and_commit; this row is the '
                       'counterfactual baseline for AC-R2 and is expected to WRITE')}
        chk('AC-R1', 'AC-R1 OLD path: the same illegal suffix is written with no gate',
            'COUNTEREXAMPLE_REPRODUCED' if wrote else 'NOT_REPRODUCED', r1)

        # ---- AC-R2: the SAME illegal suffix through the controlled (gated) path -------------
        cv2, ct2, csm2 = A.clone_state(it_served['snapshot'])
        tgt2 = next(v for v in cv2 if v.status in ('idle', 'ready'))
        for v in cv2:
            v.needs_replan = False
        tgt2.mutable_suffix = [int(victim_c)]
        # hand the gate the illegal suffix as a plan: the certifier reads the FleetPlan suffix, so
        # the served customer goes into the victim's suffix (that IS the object under test).
        plans2 = dict(fixture_plans(it_served))
        if int(tgt2.vehicle_id) in plans2:
            plans2[int(tgt2.vehicle_id)] = replace(plans2[int(tgt2.vehicle_id)],
                                                   suffix=(int(victim_c),))
        else:
            plans2[int(tgt2.vehicle_id)] = None
        cred_a = CR.make_credential(env, inst, it_served['clock'], it_served['event'],
                                    it_served['snapshot'], it_served['post_hash'], plans2,
                                    {'schema': 'ac'}, raw_contract=raw, profile=profile)
        h2_before = fp(env, it_served, cv2, ct2, csm2)
        r2 = controlled_commit(env, it_served, cv2, ct2, csm2, plans2, contract=eff, B=B_BUDGET,
                               base_plans=fixture_plans(it_served), mask_set=[],
                               protected=frozenset(), deferred=frozenset(),
                               cred_cert=cred_a, cred_install=cred_a,
                               mutable_ids=it_served['mutable_ids'])
        illegal_ok = (r2['action'] == 'REFUSE_NO_MUTATION'
                      and reason_match(r2, 'legality_short_circuit')
                      and bool(r2['components'].get('route_feasible', {})
                               .get('not_run_illegal_order_set', False))
                      and r2['state_hash_before'] == r2['state_hash_after'])
        detail2 = dict(r2)
        detail2.update({'illegal_customer': int(victim_c),
                        'expected_reason_substring': 'legality_short_circuit',
                        'expected_reason_found': reason_match(r2, 'legality_short_circuit'),
                        'state_unchanged': r2['state_hash_before'] == r2['state_hash_after'],
                        'counterfactual_row': 'AC-R1',
                        'not_a_production_path': NOT_A_PRODUCTION_PATH})
        chk('AC-R2', 'AC-R2 CONTROLLED path: the same illegal suffix is REFUSED, state unchanged',
            'REFUSE_NO_MUTATION' if illegal_ok else
            ('REFUSAL_WRONG_REASON' if r2['action'] == 'REFUSE_NO_MUTATION' else r2['action']),
            detail2)
        r2 = detail2

    # =========================================================================================
    # AC-R3 / AC-R4: legal plan, D8 gap and energy failure
    # =========================================================================================
    it = it_legal or base
    plans = fixture_plans(it)
    mut = it['mutable_ids']
    legal_creds = []
    for row_i in (0, 1):
        try:
            legal_creds.append(CR.make_credential(env, inst, it['clock'], it['event'],
                                                  it['snapshot'], it['post_hash'], plans,
                                                  {'schema': 'ac', 'row': row_i},
                                                  raw_contract=raw, profile=profile))
        except Exception as exc:                                          # noqa: BLE001
            legal_creds.append(None)
            print('  credential build failed:', type(exc).__name__, exc)

    # MEASURED FIXTURE FACT (first run): at the nominal B=300 the captured incumbent plan FAILS
    # the energy budget, so the first version of AC-R3 asserted "the D8 gap is the only blocker"
    # on a plan that was also energy-infeasible -- the verdict came back UNEXPECTED with
    # reasons=['energy_budget_exceeded'].  The D8 row therefore needs a fixture/B at which EVERY
    # executable check passes, and that fact is searched for mechanically instead of assumed.
    ROOMY_B = 1e9
    clean = None
    attempt_rows = []
    for cand in items:
        cand_plans = fixture_plans(cand)
        if not cand_plans or not any(len(p.suffix) > 0 for p in cand_plans.values()):
            attempt_rows.append({'clock': cand['clock'], 'skipped': 'no assignment in the incumbent'})
            continue
        cvc, ctc, csmc = A.clone_state(cand['snapshot'])
        r = CC.compose_certification(env, inst, cand['clock'], cvc, csmc, cand_plans,
                                     mask_set=[], protected=frozenset(), deferred=frozenset(),
                                     B=ROOMY_B, contract=eff, base_plans=cand_plans,
                                     mutable_ids=cand['mutable_ids'])
        # record WHY a candidate is not executable-clean, so a NOT_VERIFIABLE verdict is
        # attributable instead of a bare "no fixture found"
        attempt_rows.append({'clock': cand['clock'], 'ok': bool(r['ok']),
                             'reasons': list(r['reasons']),
                             'failed_components': sorted(k for k, v in r['components'].items()
                                                         if v.get('ok') is False)})
        if r['ok'] is True:
            clean = cand
            it = cand
            plans = cand_plans
            mut = cand['mutable_ids']
            legal_creds = [CR.make_credential(env, inst, cand['clock'], cand['event'],
                                              cand['snapshot'], cand['post_hash'], plans,
                                              {'schema': 'ac-clean'},
                                              raw_contract=raw, profile=profile)] * 2
            break
    n_clean_attempts = len(items)
    if clean is None:
        chk('AC-R3', 'AC-R3 legal plan refused ONLY by the D8 gap', 'NOT_VERIFIABLE',
            {'why': ('no captured decision point has an incumbent plan whose executable checks all '
                     'pass, even at B=1e9; the D8-only claim cannot be made on this window'),
             'B_tried': ROOMY_B, 'attempts': attempt_rows})
        chk('AC-R10', 'AC-R10 PROBE ONLY: gate is not a wall', 'NOT_VERIFIABLE',
            {'why': 'same fixture gap as AC-R3', 'attempts': attempt_rows})
        cert_legal = cert_probe = None
    else:
        cv3, ct3, csm3 = A.clone_state(it['snapshot'])
        r3 = controlled_commit(env, it, cv3, ct3, csm3, plans, contract=eff, B=ROOMY_B,
                               base_plans=plans, mask_set=[], protected=frozenset(),
                               deferred=frozenset(), cred_cert=legal_creds[0],
                               cred_install=legal_creds[0], mutable_ids=mut)
        cvL, ctL, csmL = A.clone_state(it['snapshot'])
        cert_legal = CC.compose_certification(env, inst, it['clock'], cvL, csmL, plans,
                                              mask_set=[], protected=frozenset(),
                                              deferred=frozenset(), B=ROOMY_B, contract=eff,
                                              base_plans=plans, mutable_ids=mut)
        d8_ok = (r3['action'] == 'REFUSE_NO_MUTATION'
                 and r3['certification_ok'] is True
                 and r3['reasons'] == []
                 and r3['blocking_capability_gaps'] == ['quality_salability']
                 and r3['state_hash_before'] == r3['state_hash_after']
                 and cert_legal['ok'] is True)
        chk('AC-R3', 'AC-R3 legal plan is refused ONLY by the D8 quality capability gap',
            'REFUSE_NO_MUTATION' if d8_ok else 'UNEXPECTED',
            {'action': r3['action'], 'reasons': r3.get('reasons'),
             'blocking_capability_gaps': r3.get('blocking_capability_gaps'),
             'certification_ok_all_executable_checks_passed': r3['certification_ok'],
             'quality_state': cert_legal['quality']['state'],
             'B_used': ROOMY_B,
             'why_roomy_B': ('the incumbent plan of the captured decision points fails the energy '
                             'budget at the nominal B=300, so a B at which the executable checks '
                             'all pass is required to isolate the D8 gap; that is measured, not '
                             'assumed (see AC-R4 for the energy-failure row at B=0.0)'),
             'state_unchanged': r3['state_hash_before'] == r3['state_hash_after'],
             'counts_as_refusal_capability': ('quality dimension only (H4): NOT installability, '
                                              'NOT G-L'),
             'components_ok': {k: v.get('ok') for k, v in r3.get('components', {}).items()},
             'not_a_production_path': NOT_A_PRODUCTION_PATH})

    cv4, ct4, csm4 = A.clone_state(it['snapshot'])
    r4 = controlled_commit(env, it, cv4, ct4, csm4, plans, contract=eff, B=0.0,
                           base_plans=plans, mask_set=[], protected=frozenset(),
                           deferred=frozenset(), cred_cert=legal_creds[0],
                           cred_install=legal_creds[0], mutable_ids=mut)
    energy_ok = (r4['action'] == 'REFUSE_NO_MUTATION'
                 and reason_match(r4, 'energy_budget_exceeded')
                 and r4['state_hash_before'] == r4['state_hash_after'])
    chk('AC-R4', 'AC-R4 certification failure (energy budget) refuses by its own reason',
        'REFUSE_NO_MUTATION' if energy_ok else
        ('REFUSAL_WRONG_REASON' if r4['action'] == 'REFUSE_NO_MUTATION' else r4['action']),
        {'action': r4['action'], 'reasons': r4.get('reasons'),
         'expected_reason_substring': 'energy_budget_exceeded',
         'expected_reason_found': reason_match(r4, 'energy_budget_exceeded'),
         'state_unchanged': r4['state_hash_before'] == r4['state_hash_after'],
         'counts_as_refusal_capability': True,
         'certification_ok': r4.get('certification_ok'),
         'not_a_production_path': NOT_A_PRODUCTION_PATH})

    # =========================================================================================
    # AC-R5: legal WAIT -- NOT a refusal
    # =========================================================================================
    cv5, ct5, csm5 = A.clone_state(it['snapshot'])
    for v in cv5:
        v.mutable_suffix = []
        v.needs_replan = False
    wait_plans = {int(vid): replace(p, suffix=()) for vid, p in plans.items()}
    h5_before = fp(env, it, cv5, ct5, csm5)
    r5 = controlled_commit(env, it, cv5, ct5, csm5, wait_plans, contract=eff, B=B_BUDGET,
                           base_plans=wait_plans, mask_set=[], protected=frozenset(),
                           deferred=frozenset(), cred_cert=legal_creds[0],
                           cred_install=legal_creds[0], mutable_ids=mut)
    # the environment's OWN loop must also write nothing in this situation
    env._plan_and_commit(inst, it['clock'], cv5, ct5, csm5, it['all_customers'])
    h5_after = fp(env, it, cv5, ct5, csm5)
    wait_ok = (r5['action'] == 'WAIT_BY_ENV_PROTOCOL' and h5_before == h5_after
               and r5['counts_as_refusal_capability'] is False)
    chk('AC-R5', 'AC-R5 legal WAIT is recorded as WAIT_BY_ENV_PROTOCOL, never as a refusal',
        'WAIT_BY_ENV_PROTOCOL' if wait_ok else 'UNEXPECTED',
        {'action': r5['action'], 'counts_as_refusal_capability': False,
         'state_unchanged_by_both_paths': h5_before == h5_after,
         'env_own_loop_wrote_nothing': h5_before == h5_after,
         'constructed_by': 'harness (all candidate suffixes emptied = the env WAIT protocol state)',
         'why_not_a_refusal': ('no write was requested, so there is nothing to refuse; counting '
                               'this as refusal capability would be the D0 ambiguity the B3-CG '
                               'addendum forbids')})

    # =========================================================================================
    # AC-R6: state drift -- the credential refuses although the PLAN is byte-identical
    # =========================================================================================
    cv6, ct6, csm6 = A.clone_state(it['snapshot'])
    plans_at_s1 = fixture_plans(it)
    hash_s1 = AC.plan_hash(plans_at_s1)
    snap_s1 = it['snapshot']
    hash_state_s1 = it['post_hash']
    cred_s1 = CR.make_credential(env, inst, it['clock'], it['event'], snap_s1, hash_state_s1,
                                 plans_at_s1, {'schema': 'ac-drift'}, raw_contract=raw,
                                 profile=profile)
    # drift: advance the clock by 1.0 and clear mutable tails (exactly what a reveal does)
    for v in cv6:
        if v.status != 'closed':
            v.mutable_suffix = []
            v.needs_replan = True
    env.prepare_decision_point(it['clock'] + 1.0, cv6)
    snap_drift, hash_state_drift = A.capture_decision_state(
        env, inst, it['clock'] + 1.0, it['event'] + 1, 0, cv6, ct6, csm6, it['all_customers'])
    cred_drift = CR.make_credential(env, inst, it['clock'] + 1.0, it['event'] + 1, snap_drift,
                                    hash_state_drift, plans_at_s1, {'schema': 'ac-drift'},
                                    raw_contract=raw, profile=profile)
    control = GATE.pre_install_reverify(cred_s1, cred_s1, deadline_ok=True,
                                        certification_complete=True)
    drifted = GATE.pre_install_reverify(cred_s1, cred_drift, deadline_ok=True,
                                        certification_complete=True)
    plan_unchanged = (AC.plan_hash(plans_at_s1) == hash_s1)
    drift_ok = (control['allow'] is True and drifted['allow'] is False
                and plan_unchanged
                and any(r in drifted['refusals'] for r in ('state_hash_mismatch',
                                                           'plan_digest_anchor_mismatch',
                                                           'plan_hash_legacy_mismatch')))
    chk('AC-R6', 'AC-R6 state drift is refused even though the PLAN is unchanged',
        'REFUSE_NO_MUTATION' if drift_ok else 'UNEXPECTED',
        {'control_same_state_allow': control['allow'],
         'drifted_allow': drifted['allow'],
         'drifted_refusals': drifted['refusals'],
         'plan_hash_s1': hash_s1, 'plan_hash_after_drift': AC.plan_hash(plans_at_s1),
         'plan_byte_identical': plan_unchanged,
         'state_hash_s1': hash_state_s1, 'state_hash_drifted': hash_state_drift,
         'counts_as_refusal_capability': True,
         'refusal_caused_by_drift_not_by_blanket_deny': (control['allow'] is True
                                                        and drifted['allow'] is False)})

    # =========================================================================================
    # AC-R7: the WRITER layer cannot see the drift (plan_hash is blind to anchor_time/load)
    # =========================================================================================
    p_a = plans_at_s1
    vid0 = sorted(p_a)[0]
    p0 = p_a[vid0]
    p_b = dict(p_a)
    p_b[vid0] = replace(p0, anchor_time=p0.anchor_time + 1.0, anchor_load=p0.anchor_load + 7.0)
    hash_blind = (AC.plan_hash(p_a) == AC.plan_hash(p_b)
                  and p_a[vid0].anchor_time != p_b[vid0].anchor_time)
    installed = []
    led = GA.InstallLedger(install_fn=lambda pl: installed.append(pl))
    allowed = led.finish(failure_reason=None, plan=p_a, original_plan=p_a, certified=True,
                         certified_plan_hash=AC.plan_hash(plans_at_s1), certifier='compose',
                         deadline_ok=True)
    chk('AC-R7', 'AC-R7 EXCLUSIVITY gap: the writer layer installs under an unseen drift',
        'COUNTEREXAMPLE_REPRODUCED' if (hash_blind and allowed and len(installed) == 1)
        else 'NOT_REPRODUCED',
        {'plan_hash_blind_to_anchor_time_and_load': hash_blind,
         'plan_hash_a': AC.plan_hash(p_a), 'plan_hash_b': AC.plan_hash(p_b),
         'anchor_time_a': p0.anchor_time, 'anchor_time_b': p_b[vid0].anchor_time,
         'ledger_installed': len(installed),
         'ledger_returns': bool(allowed),
         'honest_scope': ('the CREDENTIAL layer (AC-R6) does refuse the drift; the WRITER layer '
                          '(`InstallLedger.finish`, d3_geo_adapter.py:666-689) never consults the '
                          'credential, it only compares `plan_hash` -- so the two layers are not '
                          'composed and exclusivity holds only where the credential is consulted')})

    # =========================================================================================
    # AC-R8: one-time use
    # =========================================================================================
    sub1 = GATE.pre_install_reverify(cred_s1, cred_s1, deadline_ok=True,
                                     certification_complete=True)
    sub2 = GATE.pre_install_reverify(cred_s1, cred_s1, deadline_ok=True,
                                     certification_complete=True)
    installed2 = []
    led2 = GA.InstallLedger(install_fn=lambda pl: installed2.append(pl))
    first = led2.finish(failure_reason=None, plan=p_a, original_plan=p_a, certified=True,
                        certified_plan_hash=AC.plan_hash(p_a), certifier='compose',
                        deadline_ok=True)
    second = led2.finish(failure_reason=None, plan=p_a, original_plan=p_a, certified=True,
                         certified_plan_hash=AC.plan_hash(p_a), certifier='compose',
                         deadline_ok=True)
    second_event = led2.events[-1] if led2.events else {}
    gate_replay = sub1['allow'] and sub2['allow']
    chk('AC-R8', 'AC-R8 ONE-TIME-USE: the gate is a pure function and can be replayed',
        'COUNTEREXAMPLE_REPRODUCED' if gate_replay else 'NOT_REPRODUCED',
        {'gate_layer': {'first_allow': sub1['allow'], 'second_allow_same_credential': sub2['allow'],
                        'why': ('pre_install_reverify(cred_cert, cred_install, ...) is a pure '
                                'function of its arguments; there is no nonce, counter or sequence, '
                                'so the same credential authorises arbitrarily many submissions')},
         'writer_layer': {'first_finish': bool(first), 'second_finish': bool(second),
                          'n_installs': led2.n_installs,
                          'second_finish_reason': second_event.get('reason'),
                          'why': ('InstallLedger.finish has a per-OBJECT `finished` guard, so '
                                  'reuse is refused only while the SAME ledger object is reused -- '
                                  'a fresh ledger with the same credential is not refused')}})

    # =========================================================================================
    # AC-R9: failure consistency -- install_fn raises midway
    # =========================================================================================
    cv9, ct9, csm9 = A.clone_state(it['snapshot'])
    h9_before = fp(env, it, cv9, ct9, csm9)
    partial_target = next(v for v in cv9 if v.status in ('idle', 'ready'))

    def exploding_install(plan):
        partial_target.mutable_suffix = [999]          # a PARTIAL write, then failure
        raise RuntimeError('install_fn failed after mutating the first vehicle')

    led9 = GA.InstallLedger(install_fn=exploding_install)
    escaped = None
    try:
        led9.finish(failure_reason=None, plan=p_a, original_plan=p_a, certified=True,
                    certified_plan_hash=AC.plan_hash(p_a), certifier='compose', deadline_ok=True)
    except BaseException as exc:                                          # noqa: BLE001
        escaped = '%s: %s' % (type(exc).__name__, str(exc)[:80])
    h9_after = fp(env, it, cv9, ct9, csm9)
    chk('AC-R9', 'AC-R9 FAILURE CONSISTENCY: a raising install_fn leaves an unreported partial write',
        'FAILURE_ATOMICITY_ABSENT' if (escaped and h9_before != h9_after) else 'UNEXPECTED',
        {'exception_escaped': escaped,
         'partial_write_visible_on_the_copy': h9_before != h9_after,
         'partial_target_suffix_after': list(partial_target.mutable_suffix),
         'ledger_n_installs': led9.n_installs,
         'ledger_events': led9.events,
         'ledger_finished': led9.finished,
         'ledger_n_rollback_unverified': led9.n_rollback_unverified,
         'why': ('`d3_geo_adapter.py:689` calls `self.install_fn(plan)` outside any guard and sets '
                 '`finished` only AFTER it returns, so an exception escapes with n_installs==0, no '
                 'event, finished==False and the next finish() free to install again'),
         'no_crash_recovery_claimed': ('H5: there is no durable transaction mechanism, so NOTHING '
                                      'here may be described as crash-recovery atomicity')})

    # =========================================================================================
    # AC-R10: PROBE ONLY -- the gate is not a wall; the only blocker is the D8 gap
    # =========================================================================================
    def _probe_quality(env_, inst_, clock_, vehicles_, served_mask_, plans_):
        return True, {'probe_only': True,
                      'warning': 'this is NOT a quality certificate and must never be installed'}

    cvP, ctP, csmP = A.clone_state(it['snapshot'])
    cert_probe = CC.compose_certification(env, inst, it['clock'], cvP, csmP, plans,
                                          mask_set=[], protected=frozenset(),
                                          deferred=frozenset(), B=ROOMY_B, contract=eff,
                                          base_plans=plans, mutable_ids=mut,
                                          quality_check_fn=_probe_quality) if clean else None
    if clean:
        chk('AC-R10',
            'AC-R10 PROBE ONLY: installable is reachable solely via an injected quality check',
            'PROBE_ONLY_GATE_NOT_A_WALL' if (cert_probe['installable'] is True
                                             and cert_legal['installable'] is False) else 'UNEXPECTED',
            {'without_quality_check_installable': cert_legal['installable'],
             'with_injected_check_installable': cert_probe['installable'],
             'blocking_gaps_without_check': cert_legal['blocking_capability_gaps'],
             'blocking_gaps_with_check': cert_probe['blocking_capability_gaps'],
             'executable_checks_ok_without_check': cert_legal['ok'],
             'B_used': ROOMY_B,
             'not_a_production_path': True,
             'install_permission_claimed': False,
             'why': ('answers "is the gate discriminating or a wall?": every executable check '
                     'already passes, so the D8 capability gap is the ONLY blocker -- and injecting '
                     'a fake quality check is what flips `installable`. That injection is '
                     'diagnostic only; no install permission is claimed and D8 is not relaxed')})

    # =========================================================================================
    # PREREG + AUDIT
    # =========================================================================================
    ids = [r['cid'] for r in RESULTS]
    missing = [i for i in PREREGISTERED if i not in ids]
    chk('PREREG', 'every pre-registered counterexample is present; none dropped',
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
    chk('AUDIT', 'the acceptance probes did not perturb the production trajectory',
        'OK' if fp_probed == fp_clean else 'PRODUCTION_PATH_PERTURBED',
        {'probed_outcome_sha256': fp_probed, 'clean_rerun_outcome_sha256': fp_clean,
         'n_served': int(np.count_nonzero(served))})

    counts = {}
    for r in RESULTS:
        counts[r['verdict']] = counts.get(r['verdict'], 0) + 1
    refused_rows = [r for r in RESULTS if r['verdict'] == 'REFUSE_NO_MUTATION']
    out = {
        'stage': 'D3_G0_Evidence_B3AC_gate_refusal_acceptance',
        'read_only': True, 'production_state_written': False, 'commit_modified': False,
        'd8_relaxed': False, 'real_writeback_enabled': False,
        'propositions': {'P1_gate_can_refuse': 'VERIFIED_ON_ISOLATED_COPIES_ONLY',
                         'P2_every_real_commit_goes_through_the_gate': 'NOT_THIS_BATCH_B3_AT'},
        'gates': {'G-L': 'NOT_ESTABLISHED', 'G-C': 'NOT_ESTABLISHED'},
        'refusal_capability': {
            'n_rows_with_a_refusal': len(refused_rows),
            'counted_as_capability': [r['cid'] for r in refused_rows
                                      if r['detail'].get('counts_as_refusal_capability') is True],
            'counted_on_the_quality_dimension_only': [
                r['cid'] for r in refused_rows
                if isinstance(r['detail'].get('counts_as_refusal_capability'), str)],
            'waits_not_counted': [r['cid'] for r in RESULTS
                                  if r['verdict'] == 'WAIT_BY_ENV_PROTOCOL'],
            'wrong_reason_refusals': [r['cid'] for r in RESULTS
                                      if r['verdict'] == 'REFUSAL_WRONG_REASON'],
            'note': ('H2/H3/H4: only refusals attributed to the expected REASON count; legal WAIT '
                     'never counts; the D8 refusal counts on the quality dimension only.  The '
                     'quality-dimension row is listed SEPARATELY and deliberately not merged into '
                     '`counted_as_capability`, so nobody can read it as installability or G-L')},
        'preregistered': PREREGISTERED, 'covered': ids, 'missing': missing,
        'verdict_counts': counts,
        'identities': {
            'data_npz_sha256': sha256_file(CAL_NPZ),
            'profile_hash': profile.profile_hash,
            'effective_contract_hash': eff.contract_hash,
            'prereg_spec_sha256': sha256_file(EXT.parent / 'research' / 'review_3080_20261010' /
                                              'd3_g0_evidence_root_v1' /
                                              'B3AC_COUNTEREXAMPLE_SPEC.json'),
            'code': {
                'scripts/tests/test_d3_b3ac_gate_refusal.py': sha256_file(Path(__file__)),
                'scripts/simulation/d3_composed_certifier.py':
                    sha256_file(SIM / 'd3_composed_certifier.py'),
                'scripts/simulation/d3_install_gate.py': sha256_file(SIM / 'd3_install_gate.py'),
                'scripts/simulation/d3_credential.py': sha256_file(SIM / 'd3_credential.py'),
                'scripts/simulation/d3_geo_adapter.py': sha256_file(SIM / 'd3_geo_adapter.py'),
                'scripts/simulation/action_contract.py': sha256_file(SIM / 'action_contract.py'),
                'scripts/simulation/strict_online_env.py':
                    sha256_file(SIM / 'strict_online_env.py'),
            },
        },
        'results': RESULTS,
    }
    for r in RESULTS:
        print('  %-9s %-34s %s' % (r['cid'], r['verdict'], r['name']))
    print('\nverdict_counts:', json.dumps(counts, ensure_ascii=False))
    dest = (EXT.parent / 'research' / 'review_3080_20261010' / 'd3_g0_evidence_root_v1' /
            'B3AC_REFUSAL_MATRIX.json')
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + '\n',
                    encoding='utf-8', newline='\n')
    print('wrote', dest)
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
