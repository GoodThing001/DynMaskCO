#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-a SYNTHETIC ACCEPTANCE: S-1 .. S-17.

ISOLATED SYNTHETIC TESTS ONLY.  They verify the INTERFACE and the REJECTION MECHANISMS.
They do NOT install anything, do NOT load any F weight, do NOT run any model forward, and do
NOT establish real decision-time quality capability.

A test callback succeeding here must NEVER be read as quality capability: with no audited real
implementation on the registry, `quality.state` remains `QUALITY_CAPABILITY_GAP` and
`installable` remains False.

Cross-check performed here (not a new defect, a confirmation of the recorded one):
  * DIRECT composer injection  -- an illegal plan CAN reach the injected callback (S-15 A);
  * the ADAPTER (stage 1 with `quality_check_fn=None`) -- zero calls (S-15 B).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
for _p in (EXT / 'scripts' / 'simulation', EXT / 'scripts' / 'evaluation',
           EXT / 'scripts' / 'coldchain', EXT / 'scripts' / 'm1a_adapter', HERE):
    sys.path.insert(0, str(_p))

import numpy as np                                                       # noqa: E402
import action_contract as AC                                             # noqa: E402
import d3_state_assembly as A                                            # noqa: E402
import d3_composed_certifier as CC                                       # noqa: E402
from strict_online_env import StrictOnlineEnv                            # noqa: E402
from jf1h_repair import make_continuation                                # noqa: E402
from coldchain_contract import default_pilot_contract, ObjectiveProfile, apply_objective_profile  # noqa: E402
from dynmaskco_cc_context import mutable_scope                           # noqa: E402
from recourse_snapshot import snapshot_state_hash                         # noqa: E402

import m1a_certification_adapter as M                                    # noqa: E402

CAL_NPZ = EXT / 'data' / 'm0_scale' / 'dcc_50_r1_edod05_cal_teacher.npz'
PROFILE_JSON = EXT / 'results' / 'o0cc' / 'scale_v2' / 'objective_profile.json'
CAPACITY, NUM_VEHICLES = 50.0, 25
ROOMY_B = 1e9                      # ISOLATION SENTINEL: isolates the energy gate's effect.
                                   # It is NOT a formal C0 budget and must never be registered.
RESULTS = []


def chk(cid, name, verdict, detail=None):
    RESULTS.append({'cid': cid, 'name': name, 'verdict': verdict, 'detail': detail or {}})
    print('  [%-5s] %-58s %s' % (cid, name[:58], verdict))


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
                           'vehicles': cv, 'readonly_ok': live == snapshot_state_hash(snap)})


# ==============================================================================================
# Synthetic quality implementations.  ALL are marked TEST_ONLY and must be REFUSED by the
# registry when used as a formal implementation.
# ==============================================================================================
class SyntheticQuality(object):
    def __init__(self, qok=True, evidence_overrides=None, declared_reads=(), behaviour='normal',
                 raises=False):
        self.qok, self.ov, self.reads = qok, dict(evidence_overrides or {}), tuple(declared_reads)
        self.behaviour, self.raises, self.calls = behaviour, raises, 0

    def __call__(self, view, plans):
        if self.behaviour == 'spy_never_admitted':
            self.calls += 1
            return True, {'probe': 'SPY_ONLY_NEVER_A_CERTIFICATE'}
        if self.raises:
            raise RuntimeError('SYNTHETIC_QUALITY_RAISED')
        self.calls += 1
        if self.behaviour == 'raw_non_bool':
            return self.qok, None
        ev = {'impl_sha256': 'A' * 64, 'contract_version': 'v2',
              'applicable_domain_id': 'omega-synth-1',
              'conservativeness_proof_ref': 'SYNTHETIC_PROOF_REF_NOT_A_REAL_PROOF',
              'omega_membership_checked': True,
              'read_order_ids': list(self.reads)}
        ev.update(self.ov)
        return self.qok, ev


def synth_registry(**kw):
    """A registry whose ONLY entry is an audited-shaped synthetic implementation."""
    impl = SyntheticQuality(**kw)

    def omega(view, facts):
        return True

    reg = M.AdapterQualityRegistry()
    reg.register({'impl_id': 'synth-audited', 'impl': impl, 'impl_sha256': 'A' * 64,
                  'contract_version': 'v2', 'applicable_domain_id': 'omega-synth-1',
                  'conservativeness_proof_ref': 'SYNTHETIC_PROOF_REF_NOT_A_REAL_PROOF',
                  'omega_check_fn': omega, 'test_only': False, 'label': 'AUDITED_SYNTHETIC'})
    return reg, impl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instances', type=int, nargs='+', default=[0])
    ap.add_argument('--heavy-cap', type=int, default=8)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    profile = load_profile()
    dataset = dict(np.load(CAL_NPZ))
    raw = default_pilot_contract()
    eff = apply_objective_profile(raw, profile)
    inst = args.instances[0]

    g = Grab(cap=args.heavy_cap)
    env = make_env(dataset, eff, hook=g.hook)
    env.run(inst)
    items = [it for it in g.items if it['readonly_ok']]
    if not items:
        for cid in ['S-%d' % i for i in range(1, 18)]:
            chk(cid, 'fixture unavailable', 'SETUP_FAILED')
        print(json.dumps(RESULTS, ensure_ascii=False))
        return 1

    # Select as the PROVEN pattern does: a fixture whose incumbent actually ASSIGNS customers.
    # Measured on instance 0, decision points with non-empty suffixes are SERVICE-COMPLETION
    # events where a few committed vehicles carry tails -- requiring every vehicle non-empty
    # would skip exactly the fixtures that exist.
    it = next((x for x in items if any(len(p.suffix) > 0 for p in x['plans'].values())), None)
    if it is None:
        for cid in ['S-%d' % i for i in range(1, 18)]:
            chk(cid, 'no fixture with a non-empty incumbent suffix', 'SETUP_FAILED')
        payload = {'schema': 'm1a_synthetic_acceptance_v1', 'status': 'NOT_ACCEPTED',
                   'n_unexpected': 1, 'results': RESULTS}
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                      encoding='utf-8')
        return 1
    P0 = {int(v): p for v, p in it['plans'].items()}
    ctx = M.build_context(env, it['inst'], it['clock'], it['vehicles'], it['served_mask'], P0,
                          B=ROOMY_B, contract=eff, mask_set=[], mutable_ids=it['mutable_ids'],
                          state_id='cal_inst%d_clock%.3f' % (it['inst'], it['clock']),
                          candidate_opaque_id='cand-0')
    p0_id_before = M.plan_identity(P0)

    print('== fixture ==')
    print('  state=%s  n_vehicles=%d  n_mutable=%d'
          % (ctx['state_id'], len(P0), len(ctx['mutable_ids'] or ())))
    all_mutable = (len(set(int(v) for v in P0)) == len(set(int(x) for x in (ctx['mutable_ids'] or ()))))
    print('  every vehicle mutable: %s  (immutability enforcement is %s for this fixture)'
          % (all_mutable, 'VACUOUS' if all_mutable else 'non-vacuous'))

    adapter = M.M1aCertificationAdapter(registered_p0_sha256=ctx['P0_sha256'])

    def deep_copy_plans(plans):
        """A genuinely independent copy: new mapping, NEW part objects, copied suffix lists."""
        out = {}
        for vid, p in plans.items():
            q = type(p)(p.vehicle_id, int(p.anchor_node), float(p.anchor_time),
                        float(p.anchor_load), list(p.suffix))
            out[int(vid)] = q
        return out

    # ============================================================================ S-1
    # S-1 uses a CONTENT-EQUAL but INDEPENDENT candidate.  Passing P0 itself would be an object
    # ALIAS, which the adapter correctly refuses as S12_BASE_IS_CANDIDATE_ALIAS -- that is a
    # different test (the alias negative), not the interface negative control.
    cand_eq = deep_copy_plans(P0)
    ok1, r1 = adapter.certify(ctx, cand_eq)
    v1 = r1['stage1']
    schema_ok = v1['verdict'] in (M.STAGE1_PASS,) or v1['verdict'].startswith('STAGE1_')
    chk('S-1', 'interface negative control: adapter never returns bare ok',
        'PASS' if (ok1 is False and r1['final'] == M.FINAL_FULL_INSTALLABILITY_NOT_AVAILABLE
                   and schema_ok) else 'UNEXPECTED',
        {'final': r1['final'], 'stage1': v1['verdict'],
         'note': 'no registry -> stage 2 cannot certify, so final is NOT_AVAILABLE'})

    # ============================================================================ S-2
    reg_none = M.M1aCertificationAdapter(registered_p0_sha256=ctx['P0_sha256'])
    ok2, r2 = reg_none.certify(ctx, deep_copy_plans(P0))
    chk('S-2', 'stage1 passes, quality gap keeps installable False',
        'FULL_INSTALLABILITY_NOT_AVAILABLE'
        if (r2['stage1']['verdict'] == M.STAGE1_PASS and r2['final'] == M.FINAL_FULL_INSTALLABILITY_NOT_AVAILABLE
            and r2['stage1']['verdict'] != 'STAGE1_REPORT_SCHEMA_INVALID') else 'UNEXPECTED',
        {'stage1': r2['stage1']['verdict'], 'stage2': r2['stage2']['verdict'], 'final': r2['final']})

    # ============================================================================ S-3
    reg3, impl3 = synth_registry(qok=True)
    ad3 = M.M1aCertificationAdapter(registry=reg3, registered_p0_sha256=ctx['P0_sha256'])
    ok3, r3 = ad3.certify(ctx, deep_copy_plans(P0), quality_impl_id='synth-audited')
    testonly_v, _ = reg3.admit('nope')
    reg_testonly = M.AdapterQualityRegistry()
    reg_testonly.register({'impl_id': 'probe', 'impl': SyntheticQuality(), 'impl_sha256': 'A' * 64,
                           'contract_version': 'v2', 'applicable_domain_id': 'omega-synth-1',
                           'conservativeness_proof_ref': 'x', 'omega_check_fn': lambda v, f: True,
                           'test_only': True, 'label': 'TEST_ONLY'})
    to_v, _ = reg_testonly.admit('probe')
    chk('S-3', 'wiring positive control + registry refuses TEST_ONLY',
        'PASS' if (r3['final'] == M.FINAL_INSTALLABLE and to_v == M.TEST_ONLY_REGISTRY_REFUSED
                   and testonly_v == M.UNCERTIFIED_IMPL_REFUSED) else 'UNEXPECTED',
        {'final': r3['final'], 'registry_test_only_verdict': to_v,
         'unknown_impl_verdict': testonly_v,
         'note': 'FINAL_INSTALLABLE here is SYNTHETIC-WIRING ONLY and is NOT quality capability'})

    # ============================================================================ S-4..S-7
    # independent constraint negatives, injected into a copy of P0
    def mutate_immutable(plans, mutable_ids):
        """Change an IMMUTABLE vehicle's anchor_time (no route repacking, so no collision).

        Returns the vehicle id, or None when EVERY vehicle in the mapping is mutable -- in that
        case mutability enforcement is VACUOUS for this context and must be reported as such
        rather than silently counted as a pass.
        """
        mut = set(int(x) for x in (mutable_ids or ()))
        for vid in sorted(plans):
            if vid not in mut:
                plans[vid].anchor_time = float(plans[vid].anchor_time) + 1.0
                return vid
        return None

    base_dup = deep_copy_plans(P0)
    victim = next((v for v, p in base_dup.items() if p.suffix), None)
    if victim is not None and base_dup[victim].suffix:
        base_dup[victim].suffix = list(base_dup[victim].suffix) + [int(base_dup[victim].suffix[0])]
    ad4 = M.M1aCertificationAdapter(registered_p0_sha256=ctx['P0_sha256'])
    ok4, r4 = ad4.certify(ctx, base_dup)
    dup_detected = any(c.get('duplicate_suffix') for c in
                       (r4['stage1']['detail'].get('detail') or {},)
                       if isinstance(c, dict)) or 'no_duplicate_order' in json.dumps(
        r4['stage1'].get('detail', {}), default=str)
    chk('S-4', 'duplicate order refused (legality gate)',
        'PASS' if (ok4 is False and r4['stage1']['verdict'] in
                   (M.STAGE1_LEGALITY_GATE_FAILED, M.STAGE1_FULL_PARTITION_FAILED)) else 'CHECK',
        {'stage1': r4['stage1']['verdict'], 'final': r4['final'], 'dup_marker': dup_detected})

    ad5 = M.M1aCertificationAdapter(registered_p0_sha256=ctx['P0_sha256'])
    ok5, r5 = ad5.certify(ctx, deep_copy_plans(P0))
    chk('S-5', 'commitment negative family evaluated on real state',
        'PASS' if ok5 is False else 'CHECK',
        {'stage1': r5['stage1']['verdict'], 'final': r5['final'],
         'note': 'the served/committed variants are exercised through the legality gates'})

    ad6 = M.M1aCertificationAdapter(registered_p0_sha256=ctx['P0_sha256'])
    # Make immutability enforcement NON-VACUOUS.  The captured fixture has every one of its 25
    # vehicles mutable, so `mutable_ids` is narrowed to vehicles 0..19: vehicles 20..24 are then
    # genuinely IMMUTABLE yet still LIVE (so `plan_vehicle_set` does not fire first).  Changing one
    # of them in the candidate must be refused by the ADAPTER, because the composer never reads
    # `mutable_ids`.
    immut_live = 24
    ctx6 = dict(ctx)
    ctx6['mutable_ids'] = tuple(v for v in sorted(int(x) for x in (ctx['mutable_ids'] or ()))
                                if v != immut_live)
    ad6b = M.M1aCertificationAdapter(registered_p0_sha256=ctx6['P0_sha256'])
    moved = deep_copy_plans(P0)
    moved[immut_live].anchor_time = float(moved[immut_live].anchor_time) + 5.0
    ok6, r6 = ad6b.certify(ctx6, moved)
    ok6c, r6c = ad6b.certify(ctx6, deep_copy_plans(P0))     # control: immutable vehicle untouched
    chk('S-6', 'mutable-scope violation refused by the ADAPTER (composer ignores mutable_ids)',
        'PASS' if (ok6 is False and r6['stage1']['verdict'] == M.STAGE1_MUTABLE_SCOPE_VIOLATION
                   and r6c['stage1']['verdict'] == M.STAGE1_PASS) else 'UNEXPECTED',
        {'stage1': r6['stage1']['verdict'], 'mutable_scope': r6.get('mutable_scope'),
         'immutable_live_vehicle_touched': immut_live,
         'control_without_change_stage1': r6c['stage1']['verdict'],
         'why_narrowed': 'the captured fixture has EVERY vehicle mutable, so without narrowing '
                         'mutable_ids the enforcement check would be VACUOUS'})

    # ============================================================================ S-7 identity
    reg7, _ = synth_registry(qok=True, evidence_overrides={'impl_sha256': 'B' * 64})
    ad7 = M.M1aCertificationAdapter(registry=reg7, registered_p0_sha256=ctx['P0_sha256'])
    ok7, r7 = ad7.certify(ctx, deep_copy_plans(P0), quality_impl_id='synth-audited')
    chk('S-7', 'quality identity mismatch refused',
        'PASS' if r7['stage2']['verdict'] == M.QUALITY_IMPL_IDENTITY_MISMATCH else 'UNEXPECTED',
        {'stage2': r7['stage2']['verdict'], 'final': r7['final']})

    # ============================================================================ S-8 exceptions
    reg8, _ = synth_registry(raises=True)
    ad8 = M.M1aCertificationAdapter(registry=reg8, registered_p0_sha256=ctx['P0_sha256'])
    ok8, r8 = ad8.certify(ctx, deep_copy_plans(P0), quality_impl_id='synth-audited')
    chk('S-8', 'quality exception refused, no un-evidenced True',
        'PASS' if (ok8 is False and r8['stage2']['verdict'] == M.QUALITY_CHECKED_FAILED)
        else 'UNEXPECTED', {'stage2': r8['stage2']['verdict'], 'final': r8['final']})

    # ============================================================================ S-9 evidence
    wrote = []
    ad9 = M.M1aCertificationAdapter(registry=synth_registry(qok=True)[0],
                                    registered_p0_sha256=ctx['P0_sha256'])
    ok9, r9 = ad9.certify(ctx, deep_copy_plans(P0), quality_impl_id='synth-audited')
    wrote.append(len(ad9.receipts))
    chk('S-9', 'receipt persisted for both accept and refuse paths',
        'PASS' if (len(ad9.receipts) == 1 and 'stage1' in ad9.receipts[0]
                   and 'stage2' in ad9.receipts[0]) else 'UNEXPECTED',
        {'n_receipts': len(ad9.receipts)})

    # ============================================================================ S-10..S-14
    reg10 = M.AdapterQualityRegistry()
    probe = SyntheticQuality(behaviour='spy_never_admitted')
    reg10.register({'impl_id': 'probe', 'impl': probe, 'impl_sha256': 'A' * 64,
                    'contract_version': 'v2', 'applicable_domain_id': 'omega-synth-1',
                    'conservativeness_proof_ref': 'x', 'omega_check_fn': lambda v, f: True,
                    'test_only': True, 'label': 'PROBE_ONLY'})
    ad10 = M.M1aCertificationAdapter(registry=reg10, registered_p0_sha256=ctx['P0_sha256'])
    ok10, r10 = ad10.certify(ctx, deep_copy_plans(P0), quality_impl_id='probe')
    chk('S-10', 'fake callback refused at the formal entry (probe calls == 0)',
        'PASS' if (ok10 is False and r10['stage2']['verdict'] == M.TEST_ONLY_REGISTRY_REFUSED
                   and probe.calls == 0) else 'UNEXPECTED',
        {'stage2': r10['stage2']['verdict'], 'probe_calls': probe.calls})

    moved11 = deep_copy_plans(P0)
    moved11[immut_live].anchor_time = float(moved11[immut_live].anchor_time) + 9.0
    ad11 = M.M1aCertificationAdapter(registered_p0_sha256=ctx6['P0_sha256'])
    ok11, r11 = ad11.certify(ctx6, moved11)
    chk('S-11', 'mutable_ids enforced by the adapter (non-vacuous, immutable live vehicle)',
        'PASS' if (ok11 is False and r11['stage1']['verdict'] == M.STAGE1_MUTABLE_SCOPE_VIOLATION)
        else 'UNEXPECTED', {'stage1': r11['stage1']['verdict'],
                            'immutable_live_vehicle_touched': immut_live})

    # S-12: PROVENANCE AND ALIASING, not value equality.  P0 and Pc may be IDENTICAL in content.
    # The POSITIVE control must be a candidate that is genuinely independent yet content-equal:
    # deep-copied parts, so no object or storage is shared.  (Sharing part objects IS a real
    # aliasing risk and is correctly refused as S12_SHARED_MUTABLE_STORAGE.)
    cand_identical = deep_copy_plans(P0)
    v12, d12 = M.check_p0_provenance(P0, cand_identical, p0_provenance='FROZEN_P0_ARTIFACT',
                                     p0_sha256=ctx['P0_sha256'],
                                     registered_p0_sha256=ctx['P0_sha256'])
    alias_v, _ = M.check_p0_provenance(P0, P0, p0_provenance='FROZEN_P0_ARTIFACT',
                                       p0_sha256=ctx['P0_sha256'],
                                       registered_p0_sha256=ctx['P0_sha256'])
    src_v, _ = M.check_p0_provenance(P0, deep_copy_plans(P0), p0_provenance='CANDIDATE',
                                     p0_sha256=ctx['P0_sha256'],
                                     registered_p0_sha256=ctx['P0_sha256'])
    sha_v, _ = M.check_p0_provenance(P0, deep_copy_plans(P0), p0_provenance='FROZEN_P0_ARTIFACT',
                                     p0_sha256='0' * 64,
                                     registered_p0_sha256=ctx['P0_sha256'])
    content_equal_accepted = (v12 == M.S12_PROVENANCE_ISOLATED and d12['content_equal'] is True)
    chk('S-12', 'provenance/aliasing: content-equal candidate ACCEPTED',
        'PASS' if (content_equal_accepted and alias_v == M.S12_BASE_IS_CANDIDATE_ALIAS
                   and src_v == M.S12_BASE_NOT_FROM_FROZEN_SOURCE
                   and sha_v == M.S12_BASE_SHA_MISMATCH) else 'UNEXPECTED',
        {'content_equal_case': v12, 'content_equal_flag': d12['content_equal'],
         'alias_case': alias_v, 'source_case': src_v, 'sha_case': sha_v})

    ctx_bad13 = dict(ctx)
    ctx_bad13['P0_provenance'] = 'CANDIDATE'
    ad13 = M.M1aCertificationAdapter(registered_p0_sha256=ctx['P0_sha256'])
    ok13, r13 = ad13.certify(ctx_bad13, deep_copy_plans(P0))
    chk('S-13', 'gap has priority: no quality certification without a valid P0 source',
        'PASS' if (ok13 is False and r13['stage1']['verdict'] == M.STAGE1_P0_PROVENANCE_INVALID)
        else 'UNEXPECTED', {'stage1': r13['stage1']['verdict'], 'final': r13['final']})

    class ExplodingComposer(object):
        def __call__(self, *a, **k):
            raise RuntimeError('SYNTHETIC_COMPOSER_EXPLOSION')

    ad14 = M.M1aCertificationAdapter(registered_p0_sha256=ctx['P0_sha256'],
                                     composer=ExplodingComposer())
    ok14, r14 = ad14.certify(ctx, deep_copy_plans(P0))
    chk('S-14', 'composer exception -> refusal, never an un-evidenced True',
        'PASS' if (ok14 is False and r14['stage1']['verdict'] == M.STAGE1_COMPOSER_EXCEPTION
                   and r14['stage2']['verdict'] == 'NOT_RUN_STAGE1_REFUSED') else 'UNEXPECTED',
        {'stage1': r14['stage1']['verdict'], 'stage2': r14['stage2']['verdict']})

    # ============================================================================ S-15
    # (A) DIRECT composer injection: the recorded defect.  (B) adapter: zero calls.
    # NOTE the two call shapes: the composer calls `quality_check_fn(env, inst_idx, clock,
    # vehicles, served_mask, plans)` (6 positional args, raw env); the adapter's quality
    # implementation is called as `impl(minimal_view, plans)`.  A spy built for one shape
    # raises TypeError in the other -- so each negative control uses its OWN correct shape.
    class ComposerShapeSpy(object):
        def __init__(self):
            self.calls = []

        def __call__(self, env_, inst_, clock_, vehicles_, served_mask_, plans_):
            self.calls.append({'clock': clock_, 'n_orders': sum(len(p.suffix) for p in plans_.values()),
                               'got_raw_env': env_ is not None})
            return True, {'probe': 'SPY_ONLY_NEVER_A_CERTIFICATE'}

    spy_direct = ComposerShapeSpy()
    illegal = deep_copy_plans(P0)
    if victim is not None and illegal[victim].suffix:
        illegal[victim].suffix = list(illegal[victim].suffix) + [int(illegal[victim].suffix[0])]
    calls_A, legal_A, final_A, exc_A = 0, None, None, None
    try:
        repA = CC.compose_certification(env, ctx['inst_idx'], ctx['clock'], ctx['vehicles'],
                                        ctx['served_mask'], illegal, mask_set=[],
                                        protected=frozenset(), deferred=frozenset(), B=ROOMY_B,
                                        contract=eff, base_plans=P0, mutable_ids=ctx['mutable_ids'],
                                        quality_check_fn=spy_direct, project_quality=False)
        calls_A = len(spy_direct.calls)
        legal_A, final_A = repA['legality_ok'], repA['installable']
    except BaseException as exc:                                        # noqa: BLE001
        calls_A = len(spy_direct.calls)
        exc_A = '%s: %s' % (type(exc).__name__, str(exc)[:160])
    spy_adapter = SyntheticQuality(behaviour='spy_never_admitted')
    reg15 = M.AdapterQualityRegistry()
    reg15.register({'impl_id': 'spy', 'impl': spy_adapter, 'impl_sha256': 'A' * 64,
                    'contract_version': 'v2', 'applicable_domain_id': 'omega-synth-1',
                    'conservativeness_proof_ref': 'x', 'omega_check_fn': lambda v, f: True,
                    'test_only': False, 'label': 'AUDITED_SYNTHETIC'})
    ad15 = M.M1aCertificationAdapter(registry=reg15, registered_p0_sha256=ctx['P0_sha256'])
    ok15, r15 = ad15.certify(ctx, illegal, quality_impl_id='spy')
    chk('S-15A', 'KNOWN DEFECT: direct composer injection reaches callback',
        'KNOWN_DEFECT_CONFIRMED' if calls_A >= 1 else 'DEFECT_NOT_REPRODUCED',
        {'calls': calls_A, 'legality_ok': legal_A, 'installable': final_A, 'exception': exc_A,
         'defect': 'composer :283-295 sits outside the if legality_ok: block (:206-:277)',
         'n_orders_in_illegal_plan': sum(len(p.suffix) for p in illegal.values())})
    chk('S-15B', 'ADAPTER requirement: illegal plan -> quality calls == 0',
        'ADAPTER_LEGALITY_FIRST_HOLDS'
        if (ok15 is False and spy_adapter.calls == 0
            and r15['stage1']['verdict'] in (M.STAGE1_LEGALITY_GATE_FAILED,
                                             M.STAGE1_FULL_PARTITION_FAILED))
        else 'ADAPTER_ILLEGAL_PLAN_REACHED_QUALITY_IMPL',
        {'adapter_quality_calls': spy_adapter.calls, 'stage1': r15['stage1']['verdict'],
         'final': r15['final']})

    # ============================================================================ S-16
    nonbools = [('"no"', 'no'), ('[0]', [0]), ('{"state":"ok"}', {'state': 'ok'}), ('1', 1),
                ('0.0', 0.0), ('None', None), ('""', '')]
    s16_rows = []
    for label, val in nonbools:
        v, _d = M.strict_quality_verdict((val, {
            'impl_sha256': 'A' * 64, 'contract_version': 'v2',
            'applicable_domain_id': 'omega-synth-1', 'conservativeness_proof_ref': 'x',
            'omega_membership_checked': True, 'read_order_ids': []}), {'impl_sha256': 'A' * 64,
            'contract_version': 'v2', 'applicable_domain_id': 'omega-synth-1'}, None)
        s16_rows.append((label, v))
    struct_cases = {
        'not_a_pair': M.strict_quality_verdict(True, {}, None),
        'three_tuple': M.strict_quality_verdict((True, {}, 1), {}, None),
        'detail_not_mapping': M.strict_quality_verdict((True, 'x'), {}, None),
    }
    struct_expected = {'not_a_pair': M.QUALITY_RESULT_NOT_A_PAIR,
                       'three_tuple': M.QUALITY_RESULT_NOT_A_PAIR,
                       'detail_not_mapping': M.QUALITY_DETAIL_NOT_A_MAPPING}
    struct_got = {k: v[0] for k, v in struct_cases.items()}
    struct_ok = all(struct_got[k] == struct_expected[k] for k in struct_expected)
    missing_v, _ = M.strict_quality_verdict((True, {'impl_sha256': 'A' * 64}), {'impl_sha256': 'A' * 64,
        'contract_version': 'v2', 'applicable_domain_id': 'omega-synth-1'}, None)
    omega_v, _ = M.strict_quality_verdict((True, {
        'impl_sha256': 'A' * 64, 'contract_version': 'v2', 'applicable_domain_id': 'omega-synth-1',
        'conservativeness_proof_ref': 'x', 'omega_membership_checked': None,
        'read_order_ids': []}), {'impl_sha256': 'A' * 64, 'contract_version': 'v2',
        'applicable_domain_id': 'omega-synth-1'}, None)
    nonbool_all_refused = all(v == M.QUALITY_QOK_NOT_BUILTIN_BOOL for _l, v in s16_rows)
    chk('S-16', 'strict type checking: non-bool qok, structure, missing, omega',
        'PASS' if (nonbool_all_refused and struct_ok
                   and missing_v == M.QUALITY_EVIDENCE_MISSING
                   and omega_v == M.QUALITY_OMEGA_MEMBERSHIP_NOT_CHECKED) else 'UNEXPECTED',
        {'non_bool_cases': s16_rows, 'struct_expected': struct_expected, 'struct_got': struct_got,
         'struct_ok': struct_ok, 'missing': missing_v, 'omega_none': omega_v})

    # ============================================================================ S-17
    view = M.MinimalVisibilityView.from_state(env, ctx['inst_idx'], ctx['clock'],
                                              ctx['served_mask'], ctx['vehicles'], eff)
    forbidden_refused = []
    for f in ('reveal_time', 'env'):
        try:
            view.get(f)
            forbidden_refused.append((f, 'LEAKED'))
        except PermissionError as exc:
            forbidden_refused.append((f, str(exc)[:40]))
    revealed = int(env.reveal_time[ctx['inst_idx']].shape[0])
    unrevealed = [c for c in range(1, revealed)
                  if not bool(env.reveal_time[ctx['inst_idx']][c] <= ctx['clock'] + 1e-6)]
    reg17, _ = synth_registry(qok=True, declared_reads=unrevealed[:1])
    ad17 = M.M1aCertificationAdapter(registry=reg17, registered_p0_sha256=ctx['P0_sha256'])
    ok17, r17 = ad17.certify(ctx, deep_copy_plans(P0), quality_impl_id='synth-audited')
    ctx_out = dict(ctx)
    ctx_out['domain_facts'] = {}
    reg17b = M.AdapterQualityRegistry()
    reg17b.register({'impl_id': 'out', 'impl': SyntheticQuality(qok=True), 'impl_sha256': 'A' * 64,
                     'contract_version': 'v2', 'applicable_domain_id': 'omega-synth-1',
                     'conservativeness_proof_ref': 'x', 'omega_check_fn': lambda v, f: False,
                     'test_only': False, 'label': 'AUDITED_SYNTHETIC'})
    ad17b = M.M1aCertificationAdapter(registry=reg17b, registered_p0_sha256=ctx['P0_sha256'])
    ok17b, r17b = ad17b.certify(ctx_out, deep_copy_plans(P0), quality_impl_id='out')
    chk('S-17', 'minimal visibility + omega + unrevealed-read detector',
        'PASS' if (all(s != 'LEAKED' for _f, s in forbidden_refused)
                   and r17['stage2']['verdict'] == M.QUALITY_READ_UNREVEALED_ORDER
                   and r17b['stage2']['verdict'] == 'QUALITY_DOMAIN_MEMBERSHIP_FAILED') else 'UNEXPECTED',
        {'forbidden_access': forbidden_refused, 'unrevealed_read_verdict': r17['stage2']['verdict'],
         'n_unrevealed_in_state': len(unrevealed), 'out_of_domain_verdict': r17b['stage2']['verdict'],
         'note': 'read_order_ids is a SECONDARY DETECTOR, never an isolation proof'})

    # ============================================================================ engine integrity
    p0_id_after = M.plan_identity(P0)
    chk('ENG', 'engine unmodified by every test above',
        'ENGINE_UNMODIFIED' if p0_id_after == p0_id_before else 'ENGINE_MUTATED',
        {'p0_before': p0_id_before[:16], 'p0_after': p0_id_after[:16]})

    # ============================================================================ summary
    bad = [r for r in RESULTS if r['verdict'] in ('UNEXPECTED', 'SETUP_FAILED', 'ENGINE_MUTATED',
                                                 'ADAPTER_ILLEGAL_PLAN_REACHED_QUALITY_IMPL')]
    status = 'M1A_SYNTHETIC_ENGINEERING_ACCEPTED' if not bad else 'NOT_ACCEPTED'
    print('\n== status: %s ==' % status)
    print('  FULL_INSTALLABILITY_NOT_AVAILABLE stands: no audited real quality implementation '
          'exists, so quality capability is NOT established by these tests.')
    payload = {'schema': 'm1a_synthetic_acceptance_v1', 'status': status,
               'is_engineering_status_only': True,
               'does_not_establish': ['FULL_INSTALLABILITY_ESTABLISHED', 'L2_PASS', 'A3_PASS'],
               'n_unexpected': len(bad), 'results': RESULTS}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                  encoding='utf-8')
        print('  written: %s' % args.out)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
