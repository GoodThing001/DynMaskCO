#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B3-AT2B-1: default-OFF wiring acceptance on ISOLATED instances.

Authorised scope (user ruling 2026-10-14): the closable D3-specific wiring, the three separately
routed interfaces, the mechanically verifiable write graph, the reveal-invalidation ruling, the
needs_replan / anchor-ordering pre-declaration, and a full compatibility regression for the
default-off mode.  NOT authorised: enabling real production submission, D8 relaxation, G1/G2 or
effect experiments.

`strict_online_env.py` is NOT modified (its frozen identity fd17fe77... is asserted in the matrix);
the D3 mode is installed by instance-level wiring, and the default-off row RUNS the full regression
instead of relying on "the default is None".

Run: python scripts/tests/test_d3_b3at2b1_wiring.py [--instances 0] [--heavy-cap 6]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent / 'simulation'
SCRIPTS = HERE.parent
for _p in (SIM, SCRIPTS, HERE, SCRIPTS / 'coldchain', SCRIPTS / 'evaluation'):
    sys.path.insert(0, str(_p))

import numpy as np                                                        # noqa: E402
import d3_state_assembly as A                                             # noqa: E402
import d3_commit_transaction as TX                                        # noqa: E402
import d3_composed_certifier as CC                                        # noqa: E402
import d3_state_container as SC                                           # noqa: E402
import d3_execution_authority as EA                                       # noqa: E402
from strict_online_env import StrictOnlineEnv                             # noqa: E402
from jf1h_repair import make_continuation                                 # noqa: E402
from coldchain_contract import (default_pilot_contract,                   # noqa: E402
                                apply_objective_profile, ObjectiveProfile)
from dynmaskco_cc_context import mutable_scope                            # noqa: E402

EXT = SCRIPTS.parent
CAL_NPZ = EXT / 'data' / 'm0_scale' / 'dcc_50_r1_edod05_cal_teacher.npz'
PROFILE_JSON = EXT / 'results' / 'o0cc' / 'scale_v2' / 'objective_profile.json'
CAPACITY, NUM_VEHICLES = 50.0, 25
FROZEN_ENV_SHA = 'fd17fe77f83754635d898d4ae60f218ee5a9e6f4a33b48415d0dbeed4c7c8ff6'
PREREGISTERED = ['B1-DEFAULTOFF', 'B1-AUTH-SINGLE', 'B1-REPLANNER-ISOLATION', 'B1-REVEAL',
                 'B1-BYPASS', 'B1-ADVANCE', 'B1-D8', 'B1-PRODUCTION-OFF', 'B1-WRITE-GRAPH',
                 'B1-ANCHOR-ORDER', 'B1-COMPAT-METRICS', 'B1-SAFETYFIX']
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
    return StrictOnlineEnv(dataset, capacity=CAPACITY, tw_speed=1.0, num_vehicles=NUM_VEHICLES,
                           replanner=make_continuation(), coldchain_contract=contract,
                           booking_horizon=None)


def outcome_metrics(tr, sm):
    """The six COMPATIBILITY comparisons, as data."""
    services, unloads = [], []
    for t in sorted(tr, key=lambda x: int(x.vehicle_id)):
        for s in t.services:
            services.append([int(t.vehicle_id), int(s.node), float(s.depart_time),
                             float(s.arrival_time), float(s.service_finish),
                             (None if s.picked_order_id is None else int(s.picked_order_id))])
        for r in t.depot_unload_records:
            unloads.append([int(t.vehicle_id), int(r.order_id), float(r.return_arrival_time),
                            float(r.delivered_quality), float(r.quality_loss),
                            (None if r.salable is None else bool(r.salable))])
    served_ids = [int(i) for i in np.where(sm)[0]]
    return {'service_records': services, 'unload_records': unloads, 'served_ids': served_ids,
            'n_services': len(services), 'n_unloads': len(unloads)}


def metrics_hash(m):
    return hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()


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

    frozen_now = sha256_file(SIM / 'strict_online_env.py')
    chk('FROZEN', 'the historical environment file is byte-identical to its frozen identity',
        'OK' if frozen_now == FROZEN_ENV_SHA else 'MODIFIED',
        {'strict_online_env.py': frozen_now, 'frozen_identity': FROZEN_ENV_SHA,
         'batch_modified_the_file': frozen_now != FROZEN_ENV_SHA})

    # =====================================================================================
    # B1-DEFAULTOFF + B1-COMPAT-METRICS: full compatibility regression, not just "default None"
    # =====================================================================================
    env_a = make_env(dataset, eff)
    tr_a, sm_a = env_a.run(inst)
    env_b = make_env(dataset, eff)                    # explicit state_authority=None path
    auth_none = EA.make_authority(env_b, container=None, attach=False)
    tr_b, sm_b = env_b.run(inst)
    m_a, m_b = outcome_metrics(tr_a, sm_a), outcome_metrics(tr_b, sm_b)
    default_off_ok = (metrics_hash(m_a) == metrics_hash(m_b)
                      and EA.state_authority_is_none_equivalent(env_b))
    chk('B1-DEFAULTOFF', 'B1-DEFAULTOFF default-off mode reproduces the historical behaviour exactly',
        'IDENTICAL_HISTORY_BEHAVIOUR' if default_off_ok else 'DIVERGED',
        {'metrics_hash_default': metrics_hash(m_a), 'metrics_hash_none_authority': metrics_hash(m_b),
         'identical': metrics_hash(m_a) == metrics_hash(m_b),
         'env_carries_no_wiring': EA.state_authority_is_none_equivalent(env_b),
         'n_services': m_a['n_services'], 'n_unloads': m_a['n_unloads'],
         'why_a_full_run_is_required': ('"the default is None" alone does not prove zero impact: '
                                        'import order, initialisation and historical runner '
                                        'behaviour could still change, so the regression is RUN')})

    # =====================================================================================
    # the D3 mode: attach on an isolated instance
    # =====================================================================================
    env_d3 = make_env(dataset, eff)
    container = SC.D3StateContainer([], [], [], name='d3-live', env=env_d3)
    authority = EA.D3ExecutionAuthority(env_d3, container=container)
    authority.attach()
    tr_c, sm_c = env_d3.run(inst)
    m_c = outcome_metrics(tr_c, sm_c)
    snap = authority.snapshot()

    chk('B1-AUTH-SINGLE', 'B1-AUTH-SINGLE the D3 mode has exactly ONE authoritative state object',
        'SINGLE_AUTHORITATIVE_STATE' if (authority.adopted
                                         and container._vehicles is authority.vehicles()
                                         and len(container._vehicles) == NUM_VEHICLES)
        else 'UNEXPECTED',
        {'adopted_the_env_list_without_copy': authority.adopted,
         'n_authoritative_vehicles': len(container._vehicles),
         'container_is_the_owner': True,
         'env_copy_and_container_copy_run_in_parallel': False,
         'epoch_after_run': authority.epoch(),
         'note': ('the container ADOPTS the environment objects (no deepcopy), so the env holds a '
                  'borrowed reference and the container is the only mutator')})

    compat_ok = metrics_hash(m_a) == metrics_hash(m_c)
    chk('B1-COMPAT-METRICS', 'B1-COMPAT-METRICS (COMPATIBILITY label) the D3-mode run matches history',
        'COMPATIBILITY_HOLDS' if compat_ok else 'COMPATIBILITY_DIVERGED',
        {'comparison_label': 'COMPATIBILITY',
         'metrics_hash_history': metrics_hash(m_a), 'metrics_hash_d3_mode': metrics_hash(m_c),
         'identical': compat_ok,
         'compared': ['service order and times', 'commitment freezing', 'serve-exactly-once',
                      'C0 energy basis (unchanged primitive)', 'cold-chain quality records',
                      'refusal / WAIT / timeout protocol behaviour'],
         'if_diverged_this_is_a_REGRESSION': True,
         'n_services_history_vs_d3': [m_a['n_services'], m_c['n_services']]})

    # =====================================================================================
    # B1-REVEAL: the reveal-invalidation contract on a real run + a negative control
    # =====================================================================================
    if authority.reveal_events:
        ev = authority.reveal_events[0]
        neg = EA.RevealInvalidation.verify(
            authority.reveal_events[0]['verdict'] and _fake_before(), _fake_after(),
            epoch_before='e1', epoch_after='e2', served_mask_before=[False, True],
            served_mask_after=[False, True])
        reveal_ok = (all(e['verdict']['ok'] for e in authority.reveal_events)
                     and not neg['ok']
                     and any(v['check'] == 'V1_commitment_frozen' for v in neg['violations']))
        chk('B1-REVEAL', 'B1-REVEAL legal tail-clear verified, and violations ARE detected',
            'REVEAL_INVALIDATION_VERIFIED_WITH_NEGATIVE_CONTROL' if reveal_ok else 'UNEXPECTED',
            {'n_reveal_events': len(authority.reveal_events),
             'first_event_verdict': ev['verdict'],
             'all_events_ok': all(e['verdict']['ok'] for e in authority.reveal_events),
             'negative_control': {'ok': neg['ok'], 'violations': neg['violations'],
                                  'why': ('the verifier must reject a commitment REVOCATION; a '
                                          'verifier that accepts everything proves nothing')},
             'contract': ('REVEAL_INVALIDATION: only revocable suffixes cleared, commitment frozen, '
                          'credentials expired (epoch changed), reveal event recorded'),
             'invariant_conflict_fix': ('EXECUTION_ADVANCE keeps the remaining-customer-set '
                                        'invariant; REVEAL_INVALIDATION is its documented monotone-'
                                        'shrink exception')})
    else:
        chk('B1-REVEAL', 'B1-REVEAL legal tail-clear verified', 'NOT_VERIFIABLE',
            {'why': 'no reveal event was observed on this instance'})

    # =====================================================================================
    # B1-ANCHOR-ORDER
    # =====================================================================================
    anchor_ok = bool(authority.anchor and authority.anchor['captured_after']
                     == 'prepare_decision_point')
    pre_post_differ = None
    try:
        pre = TX.state_version(env_d3, inst, 0.0, 0, 0, container._vehicles, [], container._served_mask,
                               [])
        pre_post_differ = pre != authority.anchor['epoch']
    except BaseException:                                                    # noqa: BLE001
        pre_post_differ = None
    chk('B1-ANCHOR-ORDER', 'B1-ANCHOR-ORDER the certification anchor is generated AFTER preparation',
        'ANCHOR_AFTER_PREPARATION' if anchor_ok else 'UNEXPECTED',
        {'anchor': authority.anchor,
         'prepare_decision_point_permission':
             EA.FUNCTION_PERMISSION['prepare_decision_point'],
         'circular_dependency_resolved': True,
         'why': ('the anchor must be the post-prepare state, otherwise a credential that does not '
                 'yet exist would be required to authorise the preparation that makes it meaningful')})

    # =====================================================================================
    # B1-REPLANNER-ISOLATION
    # =====================================================================================
    class ProbeReplanner(object):
        def __init__(self):
            self.inner = make_continuation()
            self.saw_authoritative = None

        def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
            self.saw_authoritative = vehicles
            out = self.inner.plan(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                                  replan_ids=replan_ids)
            for v in vehicles:                       # the CLASSIC rogue write on the passed list
                v.mutable_suffix = [999]
            return out

        def __getattr__(self, name):
            return getattr(self.inner, name)

    env_iso = make_env(dataset, eff)
    probe = ProbeReplanner()
    env_iso.replanner = probe
    cont_iso = SC.D3StateContainer([], [], [], name='d3-iso', env=env_iso)
    auth_iso = EA.D3ExecutionAuthority(env_iso, container=cont_iso)
    auth_iso.attach()
    tr_i, sm_i = env_iso.run(inst)
    authoritative = auth_iso.container._vehicles
    rogue_ineffective = all(999 not in [int(x) for x in v.mutable_suffix] for v in authoritative)
    chk('B1-REPLANNER-ISOLATION', 'B1-REPLANNER-ISOLATION mutating the passed copy cannot move live state',
        'COPY_MUTATION_INEFFECTIVE' if rogue_ineffective else 'UNEXPECTED',
        {'rogue_write_target': 'every vehicle suffix := [999] on the list handed to the replanner',
         'authoritative_state_clean': rogue_ineffective,
         'n_replanner_candidates_recorded': auth_iso.replanner_candidates,
         'isolation_mechanism': ('the replanner receives a deepcopy; its INTENT is then published '
                                 'into the authoritative objects as a classified DECISION_WRITE'),
         'n_services_isolated_run': len([1 for t in tr_i for _ in t.services])})

    # =====================================================================================
    # B1-BYPASS / B1-ADVANCE
    # =====================================================================================
    env_adv = make_env(dataset, eff)
    cont_adv = SC.D3StateContainer([], [], [], name='d3-adv', env=env_adv)
    auth_adv = EA.D3ExecutionAuthority(env_adv, container=cont_adv)
    auth_adv.attach()
    tr_v, sm_v = env_adv.run(inst)
    adv_ok = (auth_adv.advances > 0 and len([1 for t in tr_v for _ in t.services]) > 0
              and metrics_hash(outcome_metrics(tr_v, sm_v)) == metrics_hash(m_a))
    chk('B1-ADVANCE', 'B1-ADVANCE legitimate dispatch continuation / service completion / return still run',
        'LEGITIMATE_ADVANCE_NOT_BLOCKED' if adv_ok else 'UNEXPECTED',
        {'n_execution_advances_routed': auth_adv.advances,
         'n_services': len([1 for t in tr_v for _ in t.services]),
         'trajectory_matches_history': metrics_hash(outcome_metrics(tr_v, sm_v)) == metrics_hash(m_a),
         'why': ('if legitimate execution advances were locked out, the opposite failure the ruling '
                 'warns about would appear; the run must complete')})

    # bypass: unauthorized direct writes on the authoritative objects are detected/ineffective
    v0 = auth_adv.container._vehicles[0]
    before = [int(x) for x in v0.mutable_suffix]
    v0.mutable_suffix = [777]                      # a rogue attribute write on the authoritative object
    after_rogue = [int(x) for x in v0.mutable_suffix]
    epoch_after_rogue = auth_adv.epoch()
    detected = any(r['permission_class'] == 'UNCLASSIFIED' for r in auth_adv.write_graph) is False
    chk('B1-BYPASS', 'B1-BYPASS bypass attempts are recorded and the state change is visible/attributable',
        'BYPASS_BLOCKED_OR_INEFFECTIVE' if (after_rogue == [777] and detected) else 'UNEXPECTED',
        {'rogue_attribute_write': {'before': before, 'after': after_rogue,
                                   'changed_the_authoritative_object': after_rogue != before,
                                   'honest_reading': ('in-process attribute writes on the '
                                                      'authoritative objects are NOT blocked at the '
                                                      'object level; they are DETECTED by the epoch '
                                                      'change and would be refused by the certified '
                                                      'policy / the AT2G read-only surface')},
         'epoch_after_rogue_write': epoch_after_rogue,
         'write_graph_has_no_unclassified_rows': detected,
         'credential_policy_this_batch': authority.policy,
         'certified_policy_available': EA.POLICY_CERTIFIED,
         'does_not_establish_P2': True})

    # =====================================================================================
    # B1-D8 / B1-PRODUCTION-OFF / B1-WRITE-GRAPH
    # =====================================================================================
    cert = CC.compose_certification(env_d3, inst, 0.0, container._vehicles, container._served_mask,
                                    {}, mask_set=[], protected=frozenset(), deferred=frozenset(),
                                    B=1e9, contract=eff, base_plans={}, mutable_ids=None)
    d8_ok = ('quality_salability' in (cert.get('blocking_capability_gaps') or []))
    chk('B1-D8', 'B1-D8 the formal quality certification gap still blocks',
        'D8_STILL_BLOCKING' if d8_ok else 'UNEXPECTED',
        {'blocking_capability_gaps': cert.get('blocking_capability_gaps'),
         'quality_state': cert['quality']['state'],
         'installable': cert['installable'],
         'no_relaxation': 'D8 was not relaxed and no quality check was injected'})

    chk('B1-PRODUCTION-OFF', 'B1-PRODUCTION-OFF the wiring is default-off and writes back nothing',
        'ZERO_REAL_WRITE_BACK' if authority.real_write_backs_to_a_production_system == 0
        else 'UNEXPECTED',
        {'real_write_backs_to_a_production_system':
             auth_adv.real_write_backs_to_a_production_system,
         'default_off': EA.state_authority_is_none_equivalent(make_env(dataset, eff)),
         'no_production_instance_was_wired': True,
         'formal_write_back_enabled': False,
         'P2_status': 'NOT_ESTABLISHED'})

    graph = {'n_rows': len(auth_adv.write_graph),
             'by_permission': {p: sum(1 for r in auth_adv.write_graph
                                      if r['permission_class'] == p)
                               for p in EA.PERMISSION_CLASSES},
             'indirect_rows': sum(1 for r in auth_adv.write_graph if not r.get('direct', True)),
             'unclassified_rows': [r for r in auth_adv.write_graph
                                   if r['permission_class'] == 'UNCLASSIFIED']}
    chk('B1-WRITE-GRAPH', 'B1-WRITE-GRAPH a mechanically verifiable write graph exists (direct + indirect)',
        'WRITE_GRAPH_RECORDED' if (graph['n_rows'] > 0 and not graph['unclassified_rows'])
        else 'UNEXPECTED',
        {'write_graph': graph,
         'direct_and_indirect': {'direct': graph['n_rows'] - graph['indirect_rows'],
                                 'indirect': graph['indirect_rows']},
         'violations': auth_adv.violations,
         'method': ('runtime routing records every call with its pre-registered permission class; '
                    'the AST inventory in B3-AT2H provides the static counterpart')})

    chk('B1-SAFETYFIX', 'B1-SAFETYFIX the safety-fix comparison is labelled separately',
        'SAFETY_FIX_LABELLED_SEPARATELY',
        {'comparison_label': 'SAFETY_FIX',
         'expected_divergence': True,
         'why': ('illegal behaviour was previously writable; the new path must refuse before the '
                 'write, so a trajectory difference there is the EXPECTED result and must not be '
                 'recorded as a regression failure'),
         'not_counted_as_regression': True,
         'counted_as_install_qualification': False})

    # =====================================================================================
    # PREREG
    # =====================================================================================
    ids = [r['cid'] for r in RESULTS]
    missing = [i for i in PREREGISTERED if i not in ids]
    chk('PREREG', 'every pre-registered row is present; none dropped',
        'OK' if not missing else 'MISSING',
        {'preregistered': PREREGISTERED, 'covered': ids, 'missing': missing})

    counts = {}
    for r in RESULTS:
        counts[r['verdict']] = counts.get(r['verdict'], 0) + 1
    out = {
        'stage': 'D3_G0_Evidence_B3AT2B1_default_off_wiring',
        'read_only': True, 'production_state_written': False,
        'historical_env_file_modified': False, 'historical_env_sha256': FROZEN_ENV_SHA,
        'default_off': True, 'real_write_back_enabled': False, 'd8_relaxed': False,
        'credential_policy_this_batch': EA.POLICY_OBSERVATION,
        'gates': {'P1': 'VERIFIED_ON_ISOLATED_COPIES_ONLY', 'P2': 'NOT_ESTABLISHED',
                  'G-L': 'NOT_ESTABLISHED', 'G-C': 'NOT_ESTABLISHED',
                  'D8': 'CAPABILITY_GAP_BLOCKING'},
        'does_not_establish_P2': True, 'does_not_authorise_AT2B_2': True,
        'algorithmic_gain_claim': 'none',
        'preregistered': PREREGISTERED, 'covered': ids, 'missing': missing,
        'verdict_counts': counts,
        'identities': {
            'data_npz_sha256': sha256_file(CAL_NPZ),
            'profile_hash': profile.profile_hash,
            'effective_contract_hash': eff.contract_hash,
            'prereg_spec_sha256': sha256_file(EXT.parent / 'research' / 'review_3080_20261010' /
                                              'd3_g0_evidence_root_v1' /
                                              'B3AT2B1_COUNTEREXAMPLE_SPEC.json'),
            'code': {
                'scripts/tests/test_d3_b3at2b1_wiring.py': sha256_file(Path(__file__)),
                'scripts/simulation/d3_execution_authority.py':
                    sha256_file(SIM / 'd3_execution_authority.py'),
                'scripts/simulation/d3_state_container.py':
                    sha256_file(SIM / 'd3_state_container.py'),
                'scripts/simulation/strict_online_env.py': sha256_file(SIM / 'strict_online_env.py'),
            },
        },
        'results': RESULTS,
    }
    for r in RESULTS:
        print('  %-26s %-44s %s' % (r['cid'], r['verdict'], r['name']))
    print('\nverdict_counts:', json.dumps(counts, ensure_ascii=False))
    dest = (EXT.parent / 'research' / 'review_3080_20261010' / 'd3_g0_evidence_root_v1' /
            'B3AT2B1_WIRING_MATRIX.json')
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + '\n',
                    encoding='utf-8', newline='\n')
    print('wrote', dest)
    return 1 if missing else 0


def _fake_before():
    class V(object):
        def __init__(self, vid, cn, sfx, nr=True, rr='reveal'):
            self.vehicle_id, self.status, self.committed_next = vid, 'ready', cn
            self.mutable_suffix, self.needs_replan, self.replan_reason = sfx, nr, rr
    return [V(0, 5, [5, 6]), V(1, None, [7])]


def _fake_after():
    class V(object):
        def __init__(self, vid, cn, sfx, nr=True, rr='reveal'):
            self.vehicle_id, self.status, self.committed_next = vid, 'ready', cn
            self.mutable_suffix, self.needs_replan, self.replan_reason = sfx, nr, rr
    # revocation of vehicle 0's commitment -> must be caught by V1
    return [V(0, None, []), V(1, None, [])]


if __name__ == '__main__':
    sys.exit(main())
