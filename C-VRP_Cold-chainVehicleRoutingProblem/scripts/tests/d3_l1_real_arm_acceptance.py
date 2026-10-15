#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L1 REAL-ARM ACCEPTANCE: prove the real MaskCO scorer drives the two-arm protocol correctly.

This is the transition test the signing package depends on: it runs BOTH arms on ONE frozen state
with the REAL scorer (never a synthetic one) and checks the five things the protocol must guarantee.
It is a single-state engineering acceptance, NOT the L1 comparison: one state cannot answer the L1
question, and no verdict about divergence is claimed from it.

  R1 the real checkpoint/backbone load and the 83 trained leaves restore bit-exactly
  R2 at every step the logits length equals THAT step's ordered legal length
  R3 the two arms received a byte-identical initial input (state digest, first-step legal digest)
  R4 each arm's per-step legal fingerprint and logits digest are recorded, so a cross-step reuse is
     detectable; the recorded logits digests are pairwise distinct across steps
  R5 the D-Geo arm reads only its registered distance vector (no model is loaded for it)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
for _p in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/data', 'scripts/models',
           'scripts/tests'):
    sys.path.insert(0, str(EXT / _p))

import d3_geo_adapter as GEO                                                   # noqa: E402
import d3_l1_real_arm as RA                                                   # noqa: E402
import d3_l1_two_arm_protocol as TA                                           # noqa: E402

EV = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
V110 = EV / 'v110_evidence'
ROWS = []


def ck(name, ok, detail=None):
    ROWS.append({'name': name, 'ok': bool(ok), 'detail': detail})
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)[:230]))


def build_state_from_fixture(*, day, position, mask_size=2, nv_max=30, m_max=26):
    """Materialise the shared input through the ACCEPTED V1.10 path, exactly as the L1 run will."""
    import importlib.util
    import pickle
    spec = importlib.util.spec_from_file_location(
        'v110_mod', str(EXT / 'scripts/tests/d3_l1_v110_input_qualification.py'))
    v110 = importlib.util.module_from_spec(spec)
    sys.modules['v110_mod'] = v110
    spec.loader.exec_module(v110)
    import dataclasses
    from coldchain_contract import (ObjectiveProfile, apply_objective_profile,
                                    default_pilot_contract)
    from d3_l1_state_collector import build_env, make_replanner
    manifest = json.loads((EV / 'v19_evidence/STATE_MANIFEST_64_V19.json').read_text('utf-8'))
    entry = next(m for m in manifest
                 if int(m['day_id']) == day and int(m['position']) == position)
    snap = pickle.loads((EV / 'v19_evidence/formal_collection' / entry['snapshot_path'])
                        .read_bytes())
    data = dict(np.load(str(EXT / 'data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz')))
    doc = json.loads((EXT / 'results/o0cc/scale_v2/objective_profile.json').read_text('utf-8'))
    fields = {f.name for f in dataclasses.fields(ObjectiveProfile)}
    contract = apply_objective_profile(default_pilot_contract(),
                                       ObjectiveProfile(**{k: v for k, v in doc.items()
                                                           if k in fields}))
    mat = v110.materialize(data, snap, contract)
    planner, _ = make_replanner(50., 24., contract)
    env = build_env(data, 50., 25, 1., planner, contract, 24.)
    env.event_id = int(snap['event_id'])
    return env, snap, entry, mat, contract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--day', type=int, default=0)
    ap.add_argument('--position', type=int, default=1)
    args = ap.parse_args()
    started = time.monotonic()

    env, snap, entry, mat, contract = build_state_from_fixture(day=args.day,
                                                               position=args.position)
    from action_contract import apply_action
    from mpre_policy import enumerate_legal_actions

    mutable = list(mat['mutable_ids'])
    mask = list(mat['mask'])
    nv = int(mat['inputs']['Nv'])
    p0 = mat['P0']

    def legal_fn(P, remaining):
        return enumerate_legal_actions(env, args.day, P, remaining, mutable)

    def apply_fn(P, action, allowed_vehicle_ids=None):
        return apply_action(P, action, allowed_vehicle_ids=allowed_vehicle_ids)

    ledger = RA.Ledger()
    # R1 ------------------------------------------------------------------------------------
    # The JIT shapes are the DATA-DERIVED maxima, not this state's own sizes: the scorer pads every
    # state up to one fixed shape so a single compilation covers all 64, and a pad width computed
    # from a smaller per-state length is negative.
    nv_max, m_max = RA.frozen_jit_shapes(day=args.day, position=args.position)
    score, model_info = RA.load_real_f_scorer(nv_max=nv_max, m_max=m_max)
    ck('R1 the REAL MaskCO checkpoint and backbone load and the 83 trained leaves restore '
       'bit-exactly', model_info['restored_exact'] and model_info['n_train_leaves'] == 83
       and model_info['n_other_leaves'] == 0,
       {k: model_info[k] for k in ('n_train_leaves', 'restored_exact', 'n_other_leaves',
                                   'backend', 'Nv_max', 'M_max')})

    extract_fn = RA.make_real_extract_fn(
        env=env, inst_idx=args.day, clock=float(snap['clock']), vehicles=mat['vehicles'],
        served_mask=mat['served'], visible_ids=mat['visible_ids'], p0=p0, mask=mask,
        mutable_ids=mutable, node_to_local=mat['inputs']['node_to_local'],
        local_to_node=mat['inputs']['local_to_node'], nv=nv, ledger=ledger)

    # build the shared input: the reference legal list comes from the SAME legal_fn the arms use
    def base_extract_shared(P, remaining, legal):
        return {'plans_keys': sorted(P), 'n_remaining': len(remaining), 'n_legal': len(legal)}

    shared = TA.build_shared_input(
        env=env, inst_idx=args.day, day=args.day, clock=float(snap['clock']),
        vehicles=mat['vehicles'], served_mask=mat['served'], visible_ids=mat['visible_ids'],
        plans=p0, pool_B=list(mat['pool']), pool_diag=mat['pool_diag'], mask=mask,
        partial=mat['partial'], legal=mat['legal'], mutable_ids=mutable,
        protected_ids=mat['protected_ids'], inputs=mat['inputs'], contract=contract,
        snapshot=snap, state_hash=entry['state_hash'], legal_fn=legal_fn)

    f_score_fn = RA.make_score_fn(score, ledger=ledger, arm=TA.ARM_F)
    arms = TA.arm_scorers(f_score_fn, base_extract_fn=base_extract_shared,
                          allowed_vehicle_ids=mutable, f_label=RA.F_LABEL)
    # the D-Geo arm is instrumented the SAME way, so both arms are observed rather than one measured
    # and the other not
    arms['scorers'][TA.ARM_D_GEO] = RA.make_geo_score_fn(ledger=ledger)
    # the F arm must be driven by the REAL deployment extractor, so it sees deployment tensors
    arms['extract_fn'] = GEO.make_geo_extract_fn(extract_fn, allowed_vehicle_ids=mutable)

    gates = {a: TA.BudgetGate(max_scorer_calls=16, max_wall_seconds=60.0) for a in TA.ARMS}
    report = TA.run_two_arms(shared, arms, legal_fn=legal_fn, apply_fn=apply_fn, gates=gates,
                             certify_fn=None)
    audit = report['audit']

    # R2 ------------------------------------------------------------------------------------
    ok2, cross = ledger.cross_check()
    ck('R2 every scorer call returned exactly as many logits as THAT step\'s ordered legal list',
       ok2 and cross['n_calls'] >= 2, dict(cross, per_arm_counts=cross['per_arm_calls']))

    # R3 ------------------------------------------------------------------------------------
    ck('R3 the two arms received a byte-identical initial input: one state digest and one first-step '
       'legal fingerprint on both arms',
       audit['single_factor_ok'] and report['V1']['first_step_legal_identical']
       and audit['declared_varying_fields'] == ['scorer_id'],
       {'single_factor_ok': audit['single_factor_ok'],
        'first_step_legal_identical': report['V1']['first_step_legal_identical'],
        'shared_input_sha256': audit['shared_input_sha256'],
        'v1_verdict': report['V1']['verdict']})

    # R4 ------------------------------------------------------------------------------------
    per_arm_digests = {}
    for arm in TA.ARMS:
        per_arm_digests[arm] = [c['logits_sha256'] for c in ledger.calls if c['arm'] == arm]
    distinct = {a: len(set(v)) == len(v) for a, v in per_arm_digests.items()}
    step_legal = {a: [c['legal_sha256'] for c in ledger.calls if c['arm'] == a] for a in TA.ARMS}
    ck('R4 each arm\'s per-step legal fingerprint and logits digest are recorded, and the logits '
       'digests are pairwise distinct across steps (so a cross-step reuse would be visible)',
       all(distinct.values()) and all(len(v) >= 1 for v in step_legal.values()),
       {'per_arm_calls': {a: len(v) for a, v in per_arm_digests.items()},
        'digests_distinct_per_arm': distinct,
        'step_legal_fingerprints': {a: [s[:12] for s in v] for a, v in step_legal.items()},
        'post_fork_legal': report['V1']['post_fork_legal_sha256']})

    # R5 ------------------------------------------------------------------------------------
    ck('R5 the D-Geo arm used only the registered distance vector: no model was loaded for it, and '
       'its scorer identity is the registered selector',
       arms['identities'][TA.ARM_D_GEO]['scorer_id'] == GEO.SELECTOR_GEO
       and arms['identities'][TA.ARM_D_GEO]['available'] is True
       and arms['identities'][TA.ARM_F]['scorer_id'] == 'restored_maskco_logits',
       {'d_geo': arms['identities'][TA.ARM_D_GEO], 'f': arms['identities'][TA.ARM_F]})

    # V2 / V3 sanity: this is ONE state, so no L1 verdict is claimed
    v2 = report['V2']
    v3 = report['V3']
    ck('R6 the three verdicts are reported separately and V3 is NOT_RUN without a certifier '
       '(this single-state run makes NO L1 divergence claim)',
       v3.get('any_certified') == 'NOT_RUN' and 'both_completed' in v2
       and report['V1']['verdict'] in ('DIVERGENT', 'IDENTICAL', 'NOT_COMPARABLE',
                                       'NOT_APPLICABLE_EMPTY_MASK'),
       {'V1': report['V1']['verdict'], 'V2_both_completed': v2['both_completed'],
        'V2_failures': {a: v2[a]['failure_reason'] for a in TA.ARMS},
        'V3': v3.get('any_certified')})

    fails = [r for r in ROWS if not r['ok']]
    payload = {
        'schema': 'd3_l1_real_arm_acceptance',
        'state': {'day_id': args.day, 'position': args.position, 'state_hash': entry['state_hash'],
                  'n_legal_first_step': shared['common_conditions']['n_legal'],
                  'mask': mask, 'n_mask': len(mask)},
        'rows': ROWS, 'n_rows': len(ROWS), 'n_pass': len(ROWS) - len(fails), 'n_fail': len(fails),
        'ledger': {'extracts': ledger.extracts, 'calls': ledger.calls},
        'report': {'V1': report['V1'], 'V2': report['V2'], 'V3': report['V3'],
                   'audit': {k: v for k, v in report['audit'].items()}},
        'scope_limit': 'ONE state, engineering acceptance only: this cannot answer the L1 question '
                       'and makes no divergence claim',
        'F_calls_on_real_model_here': sum(1 for c in ledger.calls if c['arm'] == TA.ARM_F),
        'science_ready': False, 'stable_online_gain': False,
        'wall_s': time.monotonic() - started,
        'tool_sha256': RA.sha256_file(Path(__file__)),
        'real_arm_module_sha256': RA.sha256_file(EXT / 'scripts/simulation/d3_l1_real_arm.py'),
        'verdict': ('L1_REAL_ARM_ENGINEERING_ACCEPTED' if not fails
                    else 'L1_REAL_ARM_ENGINEERING_BLOCKED'),
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    print('\n=== L1 real-arm acceptance: %d rows, %d PASS, %d FAIL ==='
          % (len(ROWS), len(ROWS) - len(fails), len(fails)))
    print('WROTE', out)
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main())
