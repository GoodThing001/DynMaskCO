#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B2: pre-registered acceptance tests D1-D8 + SCOPE.

The id set and the expected outcome of every row are fixed in
`B2_COUNTEREXAMPLE_SPEC.json` BEFORE any result existed; this file asserts that exact id set so a
row cannot be silently dropped or re-scoped.

Authorised scope (user ruling 2026-10-14): build a computable / refusable / independently testable
certification credential -- NOT taking over the real commit loop.  Nothing here installs, nothing
modifies `_commit` / `_plan_and_commit`, no state-level rollback is claimed, and no hash is
presented as cross-process verifiable.  Counterexamples run on DETACHED copies of a real
`StrictOnlineEnv`.

Run: python scripts/tests/test_d3_b2_certification.py [--instances 0] [--heavy-cap 6]
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
import action_contract as AC                                          # noqa: E402
import d3_state_assembly as A                                          # noqa: E402
import d3_credential as CR                                             # noqa: E402
import d3_composed_certifier as CC                                     # noqa: E402
import d3_install_gate as GATE                                         # noqa: E402
from strict_online_env import StrictOnlineEnv                          # noqa: E402
from jf1h_repair import make_continuation                              # noqa: E402
from coldchain_contract import (default_pilot_contract,                # noqa: E402
                                apply_objective_profile, ObjectiveProfile,
                                ColdChainContractV2, QualityConfigV2)
from dynmaskco_cc_context import mutable_scope                         # noqa: E402
from recourse_snapshot import snapshot_state_hash                      # noqa: E402

EXT = HERE.parent.parent
CAL_NPZ = EXT / 'data' / 'm0_scale' / 'dcc_50_r1_edod05_cal_teacher.npz'
PROFILE_JSON = EXT / 'results' / 'o0cc' / 'scale_v2' / 'objective_profile.json'
CAPACITY, NUM_VEHICLES, B_BUDGET = 50.0, 25, 300.0

PREREGISTERED_IDS = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'SCOPE']
RESULTS = []


def chk(cid, name, ok, detail=None):
    RESULTS.append({'cid': cid, 'name': name, 'ok': bool(ok), 'detail': detail})


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
    """Capture the first N decision points as (snapshot, post-prepare clone) pairs, read-only."""

    def __init__(self, cap=6):
        self.cap = cap
        self.items = []
        self.n_calls = 0

    def hook(self, env, inst_idx, clock, event_id, reveal_idx, vehicles, traces, served_mask,
             all_customers=None):
        self.n_calls += 1
        if len(self.items) >= self.cap:
            return
        pre = A.live_state_fingerprint(env, inst_idx, clock, event_id, reveal_idx, vehicles,
                                       traces, served_mask, all_customers)
        snap, _ = A.capture_decision_state(env, inst_idx, clock, event_id, reveal_idx, vehicles,
                                          traces, served_mask, all_customers)
        cv, ct, csm = A.clone_state(snap)
        visible_ids = [int(i) for i in all_customers
                       if env.reveal_time[inst_idx, i] <= clock + 1e-6]
        cv, plans, replan_ids = A.prepare_clone(env, inst_idx, clock, cv, csm, visible_ids)
        post = snapshot_state_hash(A.capture_decision_state(
            env, inst_idx, clock, event_id, reveal_idx, cv, ct, csm, all_customers)[0])
        after = A.live_state_fingerprint(env, inst_idx, clock, event_id, reveal_idx, vehicles,
                                        traces, served_mask, all_customers)
        mut = mutable_scope(cv)
        with_suffix = [(len(plans[vid].suffix), int(vid)) for vid in mut
                       if vid in plans and len(plans[vid].suffix) > 0]
        self.items.append({
            'env': env, 'inst': int(inst_idx), 'clock': float(clock), 'event': int(event_id),
            'vehicles': cv, 'traces': ct, 'served_mask': csm, 'plans': plans,
            'mutable_ids': mut, 'visible_ids': visible_ids,
            'protected': (frozenset({max(with_suffix)[1]}) if with_suffix else frozenset()),
            'snapshot': snap, 'pre_hash': pre, 'post_hash': post,
            'readonly_ok': pre == after,
        })


def covered_ids():
    return [r['cid'] for r in RESULTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instances', type=int, nargs='+', default=[0])
    ap.add_argument('--heavy-cap', type=int, default=6)
    args = ap.parse_args()

    profile = load_profile()
    dataset = dict(np.load(CAL_NPZ))
    raw_contract = default_pilot_contract()
    eff_contract = apply_objective_profile(raw_contract, profile)

    g = Grab(cap=args.heavy_cap)
    env = make_env(dataset, eff_contract, hook=g.hook)
    env.run(args.instances[0])
    items = [it for it in g.items if it['readonly_ok']]
    chk('PRE', 'state captured read-only on a real env (fingerprint unchanged)',
        bool(items) and len(items) == len(g.items),
        {'n_items': len(items), 'n_calls': g.n_calls})

    base = items[0]
    env0, inst, clock = base['env'], base['inst'], base['clock']
    plans, vehicles, sm = base['plans'], base['vehicles'], base['served_mask']
    mut, prot = base['mutable_ids'], base['protected']

    def compose(**over):
        kw = dict(mask_set=[], protected=prot, deferred=frozenset(), B=B_BUDGET,
                  contract=eff_contract, mutable_ids=mut, base_plans=plans)
        kw.update(over)
        return CC.compose_certification(env0, over.pop('inst', inst), over.pop('clock', clock),
                                        over.pop('vehicles', vehicles),
                                        over.pop('served_mask', sm), over.pop('plans', plans),
                                        **kw)

    def make_cred(snapshot=None, state_hash=None, plans_=None, anchor='post_prepare',
                  raw=None, profile_=None):
        snapshot = snapshot or base['snapshot']
        return CR.make_credential(env0, inst, clock, base['event'], snapshot,
                                  state_hash or base['post_hash'], plans_ or plans,
                                  {'schema': 'test'}, raw_contract=raw, profile=profile_,
                                  state_capture_anchor=anchor)

    cert_result = compose()
    cred_cert = make_cred(raw=raw_contract, profile_=profile)

    # ================= D1: state drift after certification =================
    later = items[1] if len(items) > 1 else None
    if later is None:
        chk('D1', 'D1 state drift after certification must be refused', False,
            {'status': 'NOT_VERIFIABLE', 'why': 'only one decision point captured'})
    else:
        cred_install = CR.make_credential(later['env'], later['inst'], later['clock'],
                                          later['event'], later['snapshot'], later['post_hash'],
                                          later['plans'], {'schema': 'test'},
                                          raw_contract=raw_contract, profile=profile)
        res = GATE.pre_install_reverify(cred_cert, cred_install, deadline_ok=True,
                                       certification_complete=True)
        chk('D1', 'D1 state drift after certification must be refused',
            res['allow'] is False and 'state_hash_mismatch' in res['refusals']
            and cred_cert['state_hash'] != cred_install['state_hash'],
            {'allow': res['allow'], 'refusals': res['refusals'],
             'state_hashes_differ': cred_cert['state_hash'] != cred_install['state_hash']})

    # ================= D2: contract / profile drift =================
    v2 = ColdChainContractV2(
        schema_version=ColdChainContractV2.__dataclass_fields__['schema_version'].default,
        operational=raw_contract.operational, units=raw_contract.units,
        thermal=raw_contract.thermal, quality=QualityConfigV2(),
        objective=raw_contract.objective, parameter_provenance=raw_contract.parameter_provenance)
    eff_v2 = apply_objective_profile(v2, profile)
    # The EFFECTIVE contract is read from the env, so varying it requires a SECOND env: using the
    # same env would hand both credentials the same effective hash and the test would only be
    # exercising the raw-contract field. (This was my construction bug, not a code bug.)
    env_v2 = make_env(dataset, eff_v2)
    cred_v2 = CR.make_credential(env_v2, inst, clock, base['event'], base['snapshot'],
                                 base['post_hash'], plans, {'schema': 'test'},
                                 raw_contract=v2, profile=profile)
    res2 = GATE.pre_install_reverify(cred_cert, cred_v2, deadline_ok=True,
                                    certification_complete=True)
    # a third variant: same physics, different OBJECTIVE PROFILE (changes the effective contract)
    prof_alt = ObjectiveProfile(name='b2-alt', distance_scale=1.0, quality_scale=1.0,
                                energy_scale=1.0, lambda_quality=0.0, lambda_energy=0.0,
                                scale_source='pilot')
    env_alt = make_env(dataset, apply_objective_profile(raw_contract, prof_alt))
    cred_p = CR.make_credential(env_alt, inst, clock, base['event'], base['snapshot'],
                                base['post_hash'], plans, {'schema': 'test'},
                                raw_contract=raw_contract, profile=prof_alt)
    res2b = GATE.pre_install_reverify(cred_cert, cred_p, deadline_ok=True,
                                     certification_complete=True)
    eff_differs = cred_v2['effective_contract_hash'] != cred_cert['effective_contract_hash']
    eff_profile_differs = cred_p['effective_contract_hash'] != cred_cert['effective_contract_hash']
    d2_ok = (res2['allow'] is False and 'contract_hash_mismatch' in res2['refusals']
             and 'effective_contract_hash_mismatch' in res2['refusals']
             and res2b['allow'] is False and 'profile_hash_mismatch' in res2b['refusals']
             and eff_differs and eff_profile_differs)
    chk('D2', 'D2 contract / profile drift after certification must be refused', d2_ok,
        {'contract_variant': {'allow': res2['allow'], 'refusals': res2['refusals']},
         'profile_variant': {'allow': res2b['allow'], 'refusals': res2b['refusals']},
         'v1_raw': raw_contract.contract_hash, 'v2_raw': v2.contract_hash,
         'cert_effective': cred_cert['effective_contract_hash'],
         'v2_effective': cred_v2['effective_contract_hash'],
         'alt_profile_effective': cred_p['effective_contract_hash'],
         'effective_differs_for_v2': eff_differs,
         'effective_differs_for_alt_profile': eff_profile_differs})

    # ================= D3: anchor confusion =================
    cred_pre = make_cred(anchor='pre_prepare')
    res3 = GATE.pre_install_reverify(cred_cert, cred_pre, deadline_ok=True,
                                    certification_complete=True)
    ok3 = (res3['allow'] is False and 'anchor_mismatch' in res3['refusals']
           and cred_pre['state_capture_anchor'] == 'pre_prepare'
           and cred_cert['state_capture_anchor'] == 'post_prepare'
           and cred_cert['state_hash'] != base['pre_hash'])
    chk('D3', 'D3 pre/post-prepare anchor mismatch is explicitly recorded and refused', ok3,
        {'allow': res3['allow'], 'refusals': res3['refusals'],
         'anchors': [cred_cert['state_capture_anchor'], cred_pre['state_capture_anchor']],
         'post_vs_pre_hash_differ': cred_cert['state_hash'] != base['pre_hash'],
         'anchor_field_present': 'state_capture_anchor' in cred_cert})

    # ================= D4: anchor_time blind spot =================
    shifted = {vid: AC.VehiclePlan(p.vehicle_id, p.anchor_node, p.anchor_time + 1.0,
                                  p.anchor_load, p.suffix) for vid, p in plans.items()}
    legacy_same = AC.plan_hash(plans) == AC.plan_hash(shifted)
    strong_differs = CR.plan_digest_anchor_v1(plans) != CR.plan_digest_anchor_v1(shifted)
    cred_shift = make_cred(plans_=shifted)
    res4 = GATE.pre_install_reverify(cred_cert, cred_shift, deadline_ok=True,
                                    certification_complete=True)
    chk('D4', 'D4 the strong plan identity detects an anchor_time change the legacy hash cannot',
        legacy_same and strong_differs and res4['allow'] is False
        and 'plan_digest_anchor_mismatch' in res4['refusals'],
        {'plan_hash_legacy_equal': legacy_same, 'plan_digest_anchor_differs': strong_differs,
         'allow': res4['allow'], 'refusals': res4['refusals'],
         'legacy_len': len(AC.plan_hash(plans))})

    # ================= D5: TW / capacity =================
    victim = next((vid for vid, p in plans.items() if p.suffix), None)
    tw_violation, cap_violation = None, None
    if victim is not None:
        p = plans[victim]
        # (a) TW: force a plan whose arrival cannot meet tw_end by anchoring very late
        late = {vid: (AC.VehiclePlan(pp.vehicle_id, pp.anchor_node, 1e4, pp.anchor_load, pp.suffix)
                      if vid == victim else pp) for vid, pp in plans.items()}
        rt = CC.compose_certification(env0, inst, clock, vehicles, sm, late, mask_set=[],
                                     protected=prot, B=B_BUDGET, contract=eff_contract, base_plans=plans)
        tw_violation = {'ok': rt['ok'], 'route_ok': rt['components']['route_feasible']['ok'],
                        'reason': rt['components']['route_feasible']['per_route'][int(victim)]['reason']}
        # (b) capacity: inflate the anchor load beyond the shared capacity
        heavy = {vid: (AC.VehiclePlan(pp.vehicle_id, pp.anchor_node, pp.anchor_time,
                                      CAPACITY + 10.0, pp.suffix) if vid == victim else pp)
                 for vid, pp in plans.items()}
        rc = CC.compose_certification(env0, inst, clock, vehicles, sm, heavy, mask_set=[],
                                     protected=prot, B=B_BUDGET, contract=eff_contract, base_plans=plans)
        cap_violation = {'ok': rc['ok'], 'route_ok': rc['components']['route_feasible']['ok'],
                         'reason': rc['components']['route_feasible']['per_route'][int(victim)]['reason']}
    chk('D5', 'D5 TW and capacity violations are refused by the composed certification',
        bool(tw_violation and cap_violation and not tw_violation['ok'] and not cap_violation['ok']
             and tw_violation['reason'] == 'tw' and cap_violation['reason'] == 'capacity'),
        {'tw': tw_violation, 'capacity': cap_violation})

    # ================= D6: C0 energy budget =================
    tight = CC.compose_certification(env0, inst, clock, vehicles, sm, plans, mask_set=[],
                                    protected=prot, B=0.0, contract=eff_contract, base_plans=plans)
    roomy = CC.compose_certification(env0, inst, clock, vehicles, sm, plans, mask_set=[],
                                    protected=prot, B=1e9, contract=eff_contract, base_plans=plans)
    chk('D6', 'D6 an energy-budget violation is refused on the energy component',
        tight['components']['energy_budget']['ok'] is False and 'energy_budget_exceeded' in tight['reasons']
        and roomy['components']['energy_budget']['ok'] is True
        and tight['components']['route_feasible']['ok'] is True,
        {'tight': tight['components']['energy_budget'], 'roomy_ok':
         roomy['components']['energy_budget']['ok'],
         'route_ok_while_energy_fails': tight['components']['route_feasible']['ok']})

    # ================= D7: commitment / ownership / protected =================
    d7 = {}
    # (a) re-plan a committed customer
    committed_now = sorted({int(v.committed_next) for v in vehicles
                            if v.committed_next not in (None, 0)})
    if committed_now and victim is not None:
        c = committed_now[0]
        p = plans[victim]
        inject = dict(plans)
        inject[victim] = AC.VehiclePlan(p.vehicle_id, p.anchor_node, p.anchor_time,
                                        p.anchor_load, tuple(list(p.suffix) + [c]))
        r = CC.compose_certification(env0, inst, clock, vehicles, sm, inject, mask_set=[],
                                    protected=prot, B=B_BUDGET, contract=eff_contract, base_plans=plans)
        d7['committed_replanned'] = {'ok': r['ok'],
                                     'full_partition_ok': r['components']['full_partition']['ok'],
                                     'committed_in_suffix':
                                     r['components']['full_partition']['detail']['committed_in_suffix']}
    # (b) partition gap: drop an order
    if victim is not None and plans[victim].suffix:
        p = plans[victim]
        drop = dict(plans)
        drop[victim] = AC.VehiclePlan(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load,
                                      tuple(list(p.suffix)[:-1]))
        r = CC.compose_certification(env0, inst, clock, vehicles, sm, drop, mask_set=[],
                                    protected=prot, B=B_BUDGET, contract=eff_contract, base_plans=plans)
        d7['partition_missing'] = {'ok': r['ok'],
                                   'missing': r['components']['full_partition']['detail']['missing']}
    # (c) protected vehicle changed -- constructed as a MOVE between mutable vehicles so that
    #     exact-once is preserved (no missing/extra) and the responsible component is attributable.
    if prot and plans:
        vid = sorted(prot)[0]
        others = [v for v in sorted(mut) if v != vid and v in plans]
        pv = plans[vid]
        prot_change, moved, other = None, None, None
        if pv.suffix and others:
            other = others[0]
            moved = int(pv.suffix[0])
            prot_change = dict(plans)
            prot_change[vid] = AC.VehiclePlan(pv.vehicle_id, pv.anchor_node, pv.anchor_time,
                                              pv.anchor_load, tuple(list(pv.suffix)[1:]))
            op = plans[other]
            prot_change[other] = AC.VehiclePlan(op.vehicle_id, op.anchor_node, op.anchor_time,
                                                op.anchor_load,
                                                tuple(list(op.suffix) + [moved]))
        if prot_change is not None:
            r = CC.compose_certification(env0, inst, clock, vehicles, sm, prot_change, mask_set=[],
                                        protected=prot, B=B_BUDGET, contract=eff_contract, base_plans=plans)
            det = r['components']['repair_plan'].get('detail', {})
            fp = r['components']['full_partition'].get('detail', {})
            d7['protected_changed'] = {
                'ok': r['ok'],
                'construction': 'moved customer %s from protected vehicle %s to mutable vehicle %s'
                                % (moved, vid, other),
                'repair_plan_ok': r['components']['repair_plan']['ok'],
                'protected_vehicle_changed': det.get('protected_vehicle_changed'),
                'full_partition_ok': r['components']['full_partition']['ok'],
                'exact_once_preserved': fp.get('missing') == [] and fp.get('extra') == []}
        else:
            d7['protected_changed'] = {'ok': False,
                                       'skipped': 'protected vehicle had no suffix to move'}
    d7_ok = bool(d7) and all(v['ok'] is False for v in d7.values())
    chk('D7', 'D7 commitment / ownership / protected violations are each refused', d7_ok, d7)

    # ================= D8: quality / salability capability gap =================
    gap = compose()
    def failing_quality(env, inst_idx, clock, vehicles, served_mask, plans):
        return False, {'reason': 'injected_decision_time_quality_failure'}
    with_check = CC.compose_certification(env0, inst, clock, vehicles, sm, plans, mask_set=[],
                                         protected=prot, B=B_BUDGET, contract=eff_contract, base_plans=plans,
                                         quality_check_fn=failing_quality)
    d8_ok = (gap['quality']['state'] == CR.QUALITY_CAPABILITY_GAP
             and gap['blocking_capability_gaps'] == ['quality_salability']
             and gap['installable'] is False
             and with_check['ok'] is False and 'quality_failed' in with_check['reasons'])
    chk('D8', 'D8 quality/salability: capability gap marked AND installability refused (never a '
        'default pass)', d8_ok,
        {'no_check': {'quality': gap['quality'], 'blocking_gaps':
                      gap['blocking_capability_gaps'], 'installable': gap['installable'],
                      'ok': gap['ok']},
         'injected_failing_check': {'ok': with_check['ok'], 'reasons': with_check['reasons']},
         'evidence': {'projected_quality_loss': 'None (action_contract.certify_route:172)',
                      'salability_check': 'terminal only (coldchain_evaluator_a1:115-117)'}})

    # ================= SCOPE: no future information in the state identity =================
    sc = base['snapshot'] and cred_cert['state_identity_information_scope']
    scope_ok = (sc['visible_mask_is_causal'] is True
                and sc['unrevealed_customers_contributing'] == []
                and sc['future_only_keys_present_in_snapshot'] == []
                and sc['visible_mask_is_bound_into_hash'] is True
                and sc['base_state_hash_matches_snapshot'] is True
                and sc['declared_scope'] == 'revealed_and_committable_state_only')
    chk('SCOPE', 'SCOPE state identity depends only on decision-time / revealed / contract state',
        scope_ok,
        {'visible_mask_is_causal': sc['visible_mask_is_causal'],
         'unrevealed_customers_contributing': sc['unrevealed_customers_contributing'],
         'future_only_keys_present': sc['future_only_keys_present_in_snapshot'],
         'visible_mask_bound_into_hash': sc['visible_mask_is_bound_into_hash'],
         'probe_customer_flipped': sc['probe_customer_flipped'],
         'n_keys_classified': sc['n_keys']})

    # ================= identity matching is NOT sufficient =================
    res_ns = GATE.pre_install_reverify(cred_cert, cred_cert, deadline_ok=False,
                                      certification_complete=False)
    chk('SUF', 'identical credentials still refuse on a missed deadline / incomplete certification',
        res_ns['allow'] is False and 'deadline_exceeded' in res_ns['refusals']
        and 'certification_incomplete' in res_ns['refusals'],
        {'allow': res_ns['allow'], 'refusals': res_ns['refusals']})

    # ================= pre-registration integrity =================
    ids = [r['cid'] for r in RESULTS]
    missing = [i for i in PREREGISTERED_IDS if i not in ids]
    chk('PREREG', 'every pre-registered id (D1-D8 + SCOPE) is present; none dropped',
        not missing, {'preregistered': PREREGISTERED_IDS, 'covered': ids, 'missing': missing})

    out = {
        'stage': 'D3_G0_Evidence_B2_certification',
        'scope': 'credential_and_composed_certification_only',
        'installs_anything': False,
        'production_commit_semantics_modified': False,
        'real_writeback_enabled': False,
        'state_level_rollback_claimed': False,
        'cross_process_hash_claimed': False,
        'preregistered_ids': PREREGISTERED_IDS,
        'covered_ids': ids,
        'identities': {
            'data_npz_sha256': sha256_file(CAL_NPZ),
            'profile_hash': profile.profile_hash,
            'raw_contract_hash': raw_contract.contract_hash,
            'effective_contract_hash': eff_contract.contract_hash,
            'code': {
                'scripts/simulation/d3_credential.py': sha256_file(SIM / 'd3_credential.py'),
                'scripts/simulation/d3_composed_certifier.py':
                    sha256_file(SIM / 'd3_composed_certifier.py'),
                'scripts/simulation/d3_install_gate.py': sha256_file(SIM / 'd3_install_gate.py'),
                'scripts/simulation/d3_state_assembly.py':
                    sha256_file(SIM / 'd3_state_assembly.py'),
                'scripts/tests/test_d3_b2_certification.py': sha256_file(Path(__file__)),
            },
        },
        'results': RESULTS,
        'all_pass': all(r['ok'] for r in RESULTS),
    }
    for r in RESULTS:
        print('  %s %-7s %s' % ('PASS' if r['ok'] else 'FAIL', r['cid'], r['name']))
        if not r['ok']:
            print('        <- %s' % json.dumps(r['detail'], ensure_ascii=False, default=str)[:500])
    print('\nall_pass:', out['all_pass'])
    dest = (EXT.parent / 'research' / 'review_3080_20261010' / 'd3_g0_evidence_root_v1' /
            'B2_CERTIFICATION_MATRIX.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + '\n',
                    encoding='utf-8', newline='\n')
    print('wrote', dest)
    return 0 if out['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
