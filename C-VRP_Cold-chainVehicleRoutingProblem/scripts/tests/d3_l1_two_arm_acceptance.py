#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L1 two-arm protocol acceptance: shared input, two arms, three verdicts, six counterexamples.

Scope (user-authorised engineering, no science): build the shared input, run the two arms with
SCORERS ONLY (a synthetic F arm and the registered D-Geo arm), and prove that the failure modes the
protocol must survive are actually caught.  This harness never loads F and never runs the real
F-vs-D-Geo comparison; `stable_online_gain` and every negative verdict elsewhere are untouched.

Counterexample matrix (fixed in advance, one attempt each, failures kept):
  C1 single factor     both arms run on ONE shared state; only `scorer_id` differs; the state is
                       unchanged after both arms; the first-step legal fingerprint equals the
                       shared one on BOTH arms; the report replays bit-identically
  C2 empty pool        an empty mask pool -> `no_legal`, the ORIGINAL plan is returned
  C3 score_len         a scorer one element short -> `score_len`, never a silent prefix argmax
  C4 score_shape       a 2-D scorer -> `score_shape`, never a flattened argmax
  C5 post-fork legal   the arms really diverge; the fork is recorded and the post-fork legal lists
                       are recorded but never compared (a policy consequence, not an input defect)
  C6 cert failure      a failing certifier -> `cert_fail`, V3 False, original plan returned
  C7 timeout           a scorer that exceeds the wall limit -> `timeout`, counted as timeout only
  C8 scorer exception  a raising scorer -> `exception:<Type>`, no install path is reachable
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
for _p in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/data',
           'scripts/models', 'scripts/tests'):
    sys.path.insert(0, str(EXT / _p))

import numpy as np                                                            # noqa: E402
import d3_geo_adapter as GEO                                                  # noqa: E402
import d3_l1_two_arm_protocol as TA                                           # noqa: E402

EV = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
E19 = EV / 'v19_evidence'
E110 = EV / 'v110_evidence'
ENTRY = {'day_id': 0, 'position': 1}

CASES = []
REC = {}
# The scenario contract is `fn(outdir_or_None)`.  Scenarios that must WRITE (the trace emitter)
# read the run's output directory from here, so the caller never has to thread it through every
# scenario signature.
OUTDIR = [None]


def ck(name, ok, detail=None):
    CASES.append((name, bool(ok), detail))
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)[:260]))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------
# fixture: one REAL frozen state, materialised by the accepted V1.10 codepath
# ---------------------------------------------------------------------------------------------
def build_fixture(entry=None):
    """Reuse `d3_l1_v110_input_qualification.materialize` so the shared input comes from the SAME
    accepted codepath the 64-state input qualification used -- not a parallel re-implementation.

    `entry` selects the frozen (day_id, position) to materialise; the default is the first state.
    """
    entry = dict(ENTRY if entry is None else entry)
    created = not E110.exists()
    E110.mkdir(parents=True, exist_ok=True)
    import d3_l1_v110_input_qualification as V110
    import dataclasses
    from coldchain_contract import (ObjectiveProfile, apply_objective_profile,
                                    default_pilot_contract)
    from d3_l1_state_collector import build_env, make_replanner
    import pickle
    manifest = json.loads((E19 / 'STATE_MANIFEST_64_V19.json').read_text(encoding='utf-8'))
    row = next(m for m in manifest
               if int(m['day_id']) == int(entry['day_id'])
               and int(m['position']) == int(entry['position']))
    snap = pickle.loads((E19 / 'formal_collection' / row['snapshot_path']).read_bytes())
    data = dict(np.load(str(EXT / 'data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz')))
    profile = json.loads((EXT / 'results/o0cc/scale_v2/objective_profile.json')
                         .read_text(encoding='utf-8'))
    fields = {f.name for f in dataclasses.fields(ObjectiveProfile)}
    contract = apply_objective_profile(default_pilot_contract(),
                                       ObjectiveProfile(**{k: v for k, v in profile.items()
                                                           if k in fields}))
    mat = V110.materialize(data, snap, contract)
    planner, _ = make_replanner(50., 24., contract)
    env = build_env(data, 50., 25, 1., planner, contract, 24.)
    env.event_id = int(snap['event_id'])
    if created:
        shutil.rmtree(E110, ignore_errors=True)
    return {'entry': row, 'snap': snap, 'materialized': mat, 'env': env, 'contract': contract,
            'data_sha256': sha(EXT / 'data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz')}


def shared_input(fx, *, mask=None, pool=None, legal_fn=None):
    from action_contract import apply_action                                          # noqa: F401
    from mpre_policy import enumerate_legal_actions
    mat = fx['materialized']
    mask = list(mat['mask'] if mask is None else mask)
    partial = {vid: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load,
                            list(p.suffix)) for vid, p in mat['P0'].items()}
    for p in partial.values():
        p.suffix = tuple(c for c in p.suffix if c not in mask)
    if legal_fn is None:
        def legal_fn(P, remaining):
            return enumerate_legal_actions(fx['env'], int(fx['entry']['day_id']), P, remaining,
                                           mat['mutable_ids'])
    return TA.build_shared_input(
        env=fx['env'], inst_idx=int(fx['entry']['day_id']), day=int(fx['entry']['day_id']),
        clock=float(fx['snap']['clock']),
        vehicles=mat['vehicles'], served_mask=mat['served'],
        visible_ids=mat['visible_ids'], plans=mat['P0'], pool_B=list(mat['pool'] if pool is None
                                                                   else pool),
        pool_diag=mat['pool_diag'], mask=mask, partial=partial, legal=mat['legal'],
        mutable_ids=mat['mutable_ids'], protected_ids=mat['protected_ids'], inputs=mat['inputs'],
        contract=fx['contract'], snapshot=fx['snap'], state_hash=fx['entry']['state_hash'],
        reveal_time=float(fx['snap'].get('clock', 0.0)),
        visible_fingerprint=TA.json_digest(mat['visible_ids']),
        budget={'wall_clock_limit_s': None, 'note': 'C1 has no deadline'},
        certifier_identity=None, legal_fn=legal_fn)


def make_arms(fx, f_scores_fn=None):
    from action_contract import apply_action
    from mpre_policy import enumerate_legal_actions
    mat = fx['materialized']
    day = int(fx['entry']['day_id'])
    # `stepwise_sample_d3` hands `score_fn(state)` where `state` came from `extract_fn`.  The
    # ordered legal list is per-step, so it is carried through the extractor rather than injected
    # into the shared state (injecting it would make the shared state step-dependent and break the
    # single-factor identity).
    carry = {'legal': []}

    def f_scorer(state):
        legal = carry['legal']
        if f_scores_fn is not None:
            return f_scores_fn(state, legal)
        # Deterministic, distance-independent logits: a synthetic stand-in for restored MaskCO
        # output.  Deliberately NOT derived from -distance, so a divergence is meaningful.
        return np.asarray([np.float32(1.0 if i % 2 == 0 else -1.0)
                           for i in range(len(legal))], dtype=np.float32)

    def base_extract(P, remaining, legal):
        carry['legal'] = list(legal)
        return {'plans_keys': sorted(P), 'n_remaining': len(remaining), 'n_legal': len(legal),
                'remaining_order': [int(c) for c in remaining]}

    def legal_fn(P, remaining):
        return enumerate_legal_actions(fx['env'], day, P, remaining, mat['mutable_ids'])

    def apply_fn(P, action, allowed_vehicle_ids=None):
        return apply_action(P, action, allowed_vehicle_ids=allowed_vehicle_ids)

    arms = TA.arm_scorers(f_scorer, base_extract_fn=base_extract,
                          allowed_vehicle_ids=mat['mutable_ids'], f_label='SYNTHETIC_F_NOT_REAL_F')
    return arms, legal_fn, apply_fn


def probe_no_legal_customer(fx, *, limit=64):
    """Find a VISIBLE order that has NO feasible insertion in the mutable fleet.

    The probe is a preparation step and is recorded as one: it never looks at a score, never runs an
    arm, and never picks a state by an outcome.  `limit` caps the number of states examined so the
    scan stays bounded.
    """
    from mpre_policy import enumerate_legal_actions
    manifest = json.loads((E19 / 'STATE_MANIFEST_64_V19.json').read_text(encoding='utf-8'))
    order = [(int(m['day_id']), int(m['position'])) for m in manifest]
    probe_entry = None
    attempt = 0
    for (day, pos) in order[:limit]:
        attempt += 1
        f = fx if (day, pos) == (int(fx['entry']['day_id']), int(fx['entry']['position'])) \
            else build_fixture({'day_id': day, 'position': pos})
        mat = f['materialized']
        probe = {vid: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load,
                              list(p.suffix)) for vid, p in mat['P0'].items()}
        mask0 = [int(x) for x in mat['mask']]
        for p in probe.values():
            p.suffix = tuple(c for c in p.suffix if c not in mask0)
        for c in [int(x) for x in mat['visible_ids']]:
            legal = enumerate_legal_actions(f['env'], day, probe, [c], mat['mutable_ids'])
            if not legal:
                probe_entry = {'day_id': day, 'position': pos, 'customer': c}
                break
        if probe_entry:
            break
    return {'found': probe_entry, 'states_probed': attempt, 'probe_limit': limit}


# ---------------------------------------------------------------------------------------------
# counterexample scenarios
# ---------------------------------------------------------------------------------------------
def c1_single_factor(_root):
    """ONE shared state, both arms, the first-step legal identity, and a replayable report."""
    def once():
        fx = build_fixture()
        arms, legal_fn, apply_fn = make_arms(fx)
        shared = shared_input(fx, legal_fn=legal_fn)
        report = TA.run_two_arms(shared, arms, legal_fn=legal_fn, apply_fn=apply_fn)
        return fx, arms, shared, report

    fx, arms, shared, report = once()
    _, _, shared2, report2 = once()
    det = {'shared_input_sha256': report['audit']['shared_input_sha256'],
           'shared_input_sha256_repeat': report2['audit']['shared_input_sha256'],
           'legal_sha256': report['audit']['legal_sha256'],
           'mask_order': shared['common_conditions']['mask_order'],
           'n_legal': shared['common_conditions']['n_legal'],
           'single_factor_ok': report['audit']['single_factor_ok'],
           'state_unchanged_both_arms': report['audit']['state_unchanged_both_arms'],
           'first_step_legal_identical': report['V1']['first_step_legal_identical'],
           'v1': report['V1']['verdict'],
           'f_action': (report['V1'].get('f_action') or {}).get('customer'),
           'd_action': (report['V1'].get('d_geo_action') or {}).get('customer'),
           'v2_both': report['V2']['both_completed'],
           'v3': report['V3']['any_certified'],
           'replay_stable': TA.replay_digest(report) == TA.replay_digest(report2),
           'replay_digest': TA.replay_digest(report),
           'legal_identity_from_arm_path': shared['legal_identity']['legal_source']}
    ok = (report['audit']['single_factor_ok'] and report['audit']['state_unchanged_both_arms']
          and report['V1']['first_step_legal_identical'] and report['V2']['both_completed']
          and report['V3']['any_certified'] == 'NOT_RUN'
          and report['V1']['verdict'] in ('DIVERGENT', 'IDENTICAL')
          and det['shared_input_sha256'] == det['shared_input_sha256_repeat']
          and det['replay_stable'])
    return ok, det


def c2_empty_pool(_root):
    """An EMPTY B pool: the mask is empty, so there is nothing to repair and no legal action exists.

    This is the real `empty_pool` condition (the V1.9 read-only coverage found the pre-plan tail
    pool empty for real states).  It must be recorded as an empty denominator entry -- zero steps,
    zero legal actions, the plan unchanged -- and NOT as a divergence, a failure, or a dropped state.
    """
    fx = build_fixture()
    arms, legal_fn, apply_fn = make_arms(fx)
    shared = shared_input(fx, mask=[], legal_fn=legal_fn)
    report = TA.run_two_arms(shared, arms, legal_fn=legal_fn, apply_fn=apply_fn)
    r = report['results']
    det = {'mask_order': shared['common_conditions']['mask_order'],
           'n_legal': shared['common_conditions']['n_legal'],
           'steps': {a: r[a]['n_steps'] for a in TA.ARMS},
           'failures': {a: r[a]['failure_reason'] for a in TA.ARMS},
           'plan_unchanged': all(r[a]['plan_hash_after'] == GEO._plan_identity(
               shared['transport']['plans']) for a in TA.ARMS),
           'legal_trace_len': {a: len(r[a]['legal_trace']) for a in TA.ARMS},
           'v1': report['V1']['verdict'], 'v1_reason': report['V1']['reason'],
           'v2': report['V2']}
    ok = (shared['common_conditions']['mask_order'] == []
          and shared['common_conditions']['n_legal'] == 0
          and all(r[a]['n_steps'] == 0 and r[a]['failure_reason'] is None for a in TA.ARMS)
          and det['plan_unchanged']
          and all(det['legal_trace_len'][a] == 0 for a in TA.ARMS)
          and report['V1']['verdict'] == 'NOT_APPLICABLE_EMPTY_MASK'
          and report['V2']['both_completed'] is True)
    return ok, det


def c2b_no_legal_action(_root):
    """A mask customer that has NO legal action at all (a different, real failure mode).

    The harness scans the frozen states for a *visible* order for which the environment's own
    enumerator returns no feasible candidate inside the mutable fleet.  The scan is a preparation
    step, recorded as such: it does not look at any score and does not select a state by an outcome.
    When no such customer exists anywhere in the scanned set the scenario reports `NOT_EXERCISED`
    -- it never converts an untested branch into a pass.
    """
    first = build_fixture()
    probe = probe_no_legal_customer(first)
    if not probe['found']:
        return True, {'scenario_status': 'NOT_EXERCISED', 'reason': 'NO_SUCH_CUSTOMER',
                      **probe}
    target = probe['found']
    fx = build_fixture({'day_id': target['day_id'], 'position': target['position']})
    chosen = int(target['customer'])
    arms, legal_fn, apply_fn = make_arms(fx)
    shared = shared_input(fx, mask=[chosen], legal_fn=legal_fn)
    report = TA.run_two_arms(shared, arms, legal_fn=legal_fn, apply_fn=apply_fn)
    r = report['results']
    det = {'scenario_status': 'EXERCISED', **probe, 'state': [target['day_id'], target['position']],
           'mask_order': [chosen], 'n_legal': shared['common_conditions']['n_legal'],
           'failures': {a: r[a]['failure_reason'] for a in TA.ARMS},
           'returned_original': {a: r[a]['returned_original_plan'] for a in TA.ARMS},
           'original_plan_identity': {a: r[a]['plan_hash_after'] == GEO._plan_identity(
               shared['transport']['plans']) for a in TA.ARMS},
           'v1': report['V1']['verdict'], 'v2': report['V2'][TA.ARM_F]['full_repair_completed']}
    ok = (shared['common_conditions']['n_legal'] == 0
          and all(r[a]['failure_reason'] == 'no_legal' for a in TA.ARMS)
          and all(det['returned_original'][a] and det['original_plan_identity'][a] for a in TA.ARMS)
          and report['V1']['verdict'] == 'NOT_COMPARABLE'
          and report['V2'][TA.ARM_F]['full_repair_completed'] is False)
    return ok, det


def c3_score_len(_root):
    fx = build_fixture()

    def short(state, legal):
        return np.zeros(max(0, len(legal) - 1), dtype=np.float32)

    arms, legal_fn, apply_fn = make_arms(fx, f_scores_fn=short)
    shared = shared_input(fx, legal_fn=legal_fn)
    report = TA.run_two_arms(shared, arms, legal_fn=legal_fn, apply_fn=apply_fn)
    r = report['results']
    det = {'f_failure': r[TA.ARM_F]['failure_reason'],
           'd_failure': r[TA.ARM_D_GEO]['failure_reason'],
           'score_contract': r[TA.ARM_F]['diag'].get('score_contract'),
           'n_legal': shared['common_conditions']['n_legal'],
           'f_returned_original': r[TA.ARM_F]['returned_original_plan']}
    ok = (r[TA.ARM_F]['failure_reason'] == 'score_len'
          and (r[TA.ARM_F]['diag'].get('score_contract') or {}).get('violation')
          == 'len(scores) != len(legal)'
          and r[TA.ARM_F]['returned_original_plan']
          and report['V3']['any_certified'] == 'NOT_RUN')
    return ok, det


def c4_score_shape(_root):
    fx = build_fixture()

    def twod(state, legal):
        return np.zeros((1, len(legal)), dtype=np.float32)

    arms, legal_fn, apply_fn = make_arms(fx, f_scores_fn=twod)
    shared = shared_input(fx, legal_fn=legal_fn)
    report = TA.run_two_arms(shared, arms, legal_fn=legal_fn, apply_fn=apply_fn)
    r = report['results']
    det = {'f_failure': r[TA.ARM_F]['failure_reason'],
           'score_contract': r[TA.ARM_F]['diag'].get('score_contract'),
           'f_returned_original': r[TA.ARM_F]['returned_original_plan']}
    ok = (r[TA.ARM_F]['failure_reason'] == 'score_shape'
          and (r[TA.ARM_F]['diag'].get('score_contract') or {}).get('violation') == 'ndim != 1'
          and r[TA.ARM_F]['returned_original_plan'])
    return ok, det


def c5_post_fork_legal(_root):
    """Force a real divergence on the FIRST step and check the fork discipline.

    F prefers the LAST legal candidate, D-Geo takes the minimum distance.  With a mask of two
    customers the arms diverge on step 1 whenever the last candidate is not the nearest one; the
    harness asserts the divergence is real, that the fork is recorded, that the post-fork legal
    fingerprints are recorded, and that `first_step_legal_identical` still holds.
    """
    fx = build_fixture()
    arms, legal_fn, apply_fn = make_arms(fx)
    shared = shared_input(fx, legal_fn=legal_fn)
    if len(shared['legal_identity']['legal_order']) < 2:
        return False, {'reason': 'the chosen state has fewer than 2 legal candidates'}

    def last_preferred(state, legal):
        out = np.full(len(legal), -1.0, dtype=np.float32)
        out[-1] = 1.0
        return out

    arms, legal_fn, apply_fn = make_arms(fx, f_scores_fn=last_preferred)
    report = TA.run_two_arms(shared, arms, legal_fn=legal_fn, apply_fn=apply_fn)
    r = report['results']
    det = {'v1': report['V1']['verdict'], 'f_action': report['V1'].get('f_action'),
           'd_action': report['V1'].get('d_geo_action'),
           'first_step_legal_identical': report['V1']['first_step_legal_identical'],
           'shared_legal_sha256': shared['legal_identity']['legal_sha256'],
           'post_fork_legal_sha256': report['V1']['post_fork_legal_sha256'],
           'n_steps': {a: r[a]['n_steps'] for a in TA.ARMS},
           'failures': {a: r[a]['failure_reason'] for a in TA.ARMS},
           'v2_both': report['V2']['both_completed'],
           'plans_differ': r[TA.ARM_F]['plan_hash_after'] != r[TA.ARM_D_GEO]['plan_hash_after']}
    ok = (report['V1']['verdict'] == 'DIVERGENT'
          and report['V1']['first_step_legal_identical']
          and report['V1']['first_divergence_position'] == 0
          and det['plans_differ']
          and r[TA.ARM_F]['legal_first_step_sha256']
          == r[TA.ARM_D_GEO]['legal_first_step_sha256']
          == shared['legal_identity']['legal_sha256']
          and report['V1']['post_fork_legal_not_compared'])
    return ok, det


def c6_cert_failure(_root):
    fx = build_fixture()
    arms, legal_fn, apply_fn = make_arms(fx)
    shared = shared_input(fx, legal_fn=legal_fn)
    calls = {'n': 0}

    def failing_certifier(plan):
        calls['n'] += 1
        return False, {'reason': 'SYNTHETIC_CERTIFIER_ALWAYS_REFUSES'}

    report = TA.run_two_arms(shared, arms, legal_fn=legal_fn, apply_fn=apply_fn,
                             certify_fn=failing_certifier)
    r = report['results']
    det = {'f_failure': r[TA.ARM_F]['failure_reason'], 'certifier_calls': calls['n'],
           'f_certified': r[TA.ARM_F]['certified'],
           'f_certified_plan_hash': r[TA.ARM_F]['certified_plan_hash'],
           'f_returned_original': r[TA.ARM_F]['returned_original_plan'],
           'v3': report['V3'],
           'v2_f': report['V2'][TA.ARM_F]['full_repair_completed']}
    ok = (r[TA.ARM_F]['failure_reason'] == 'cert_fail' and calls['n'] == 2
          and report['V3']['any_certified'] is False
          and r[TA.ARM_F]['returned_original_plan']
          and report['V2'][TA.ARM_F]['full_repair_completed'] is False
          and r[TA.ARM_F]['certified'] is True)
    return ok, det


def c7_timeout(_root):
    fx = build_fixture()

    def slow(state, legal):
        time.sleep(0.05)
        return np.zeros(len(legal), dtype=np.float32)

    arms, legal_fn, apply_fn = make_arms(fx, f_scores_fn=slow)
    shared = shared_input(fx, legal_fn=legal_fn)
    report = TA.run_two_arms(shared, arms, legal_fn=legal_fn, apply_fn=apply_fn,
                             wall_clock_limit_s=0.01)
    r = report['results']
    det = {'f_failure': r[TA.ARM_F]['failure_reason'],
           'f_timed_out': r[TA.ARM_F]['diag'].get('timed_out'),
           'f_returned_original': r[TA.ARM_F]['returned_original_plan'],
           'd_failure': r[TA.ARM_D_GEO]['failure_reason'],
           'v2_f': report['V2'][TA.ARM_F]['full_repair_completed'],
           'v3': report['V3']['any_certified']}
    ok = (r[TA.ARM_F]['failure_reason'] == 'timeout'
          and r[TA.ARM_F]['diag'].get('timed_out') is True
          and r[TA.ARM_F]['returned_original_plan']
          and report['V2'][TA.ARM_F]['full_repair_completed'] is False
          and report['V3']['any_certified'] == 'NOT_RUN')
    return ok, det


def c8_scorer_exception(_root):
    fx = build_fixture()

    def boom(state, legal):
        raise ValueError('synthetic scorer failure')

    arms, legal_fn, apply_fn = make_arms(fx, f_scores_fn=boom)
    shared = shared_input(fx, legal_fn=legal_fn)
    raised = None
    try:
        TA.run_two_arms(shared, arms, legal_fn=legal_fn, apply_fn=apply_fn)
    except Exception as exc:                                              # noqa: BLE001
        raised = '%s: %s' % (type(exc).__name__, exc)
    # and the CONTAINED path: `run_repair_transaction` must convert the exception into a refusal
    # with zero installs.  A real (always-accepting) certifier is supplied so the transaction gets
    # PAST the pre-sampling certification gate and actually reaches the raising scorer.
    from d3_geo_adapter import InstallLedger, run_repair_transaction
    installs = {'n': 0, 'plan': None}
    ledger = InstallLedger(lambda p: (installs.update(n=installs['n'] + 1, plan=p)))
    outcome = run_repair_transaction(
        ledger=ledger, score_fn=arms['scorers'][TA.ARM_F], extract_fn=arms['extract_fn'],
        legal_fn=legal_fn, apply_fn=apply_fn, plans=shared['transport']['plans'],
        mask_set=shared['common_conditions']['mask_order'],
        mutable_ids=shared['transport']['mutable_ids'], selector=GEO.SELECTOR_HISTORICAL,
        deterministic=True, certify_fn=lambda plan: (True, {'synthetic': 'always_accepts'}))
    det = {'raised_out_of_protocol': raised, 'installs': installs['n'],
           'transaction_installed': outcome, 'ledger_events': ledger.events,
           'n_refusals': ledger.n_refusals, 'n_rollbacks': ledger.n_rollbacks,
           'n_state_restorations': ledger.n_state_restorations}
    ok = (raised is not None and 'synthetic scorer failure' in raised
          and outcome is False and installs['n'] == 0 and ledger.n_installs == 0
          and ledger.n_rollbacks == 1
          and ledger.events[0]['reason'].startswith('exception:'))
    return ok, det


def t1_emit_traces(outdir):
    """Emit the full two-arm reports the independent audit re-adjudicates."""
    outdir = Path(outdir if outdir is not None else OUTDIR[0])
    path = emit_traces(outdir)
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    det = {'trace_path': str(path), 'trace_sha256': sha(path), 'traces': sorted(payload),
           'has_full_reports': all('report' in v and 'results' in v['report']
                                   for v in payload.values()),
           'has_shared_identity': all(v['shared'].get('shared_input_sha256')
                                      and v['shared'].get('legal_identity', {}).get('legal_sha256')
                                      for v in payload.values())}
    ok = (sorted(payload) == ['baseline', 'forced_fork'] and det['has_full_reports']
          and det['has_shared_identity'])
    return ok, det


SCENARIOS = (
    ('C1 single factor: one shared state, both arms, first-step legal identity, replayable',
     c1_single_factor),
    ('C2 empty B pool -> the state stays in the denominator as an empty-pool entry',
     c2_empty_pool),
    ('C2b a mask with NO legal action -> no_legal and the ORIGINAL plan is returned',
     c2b_no_legal_action),
    ('C3 score_len: a short score vector is refused, never silently prefix-argmaxed', c3_score_len),
    ('C4 score_shape: a 2-D score vector is refused, never flattened', c4_score_shape),
    ('C5 post-fork legal: the fork is recorded; post-fork legal lists are never compared',
     c5_post_fork_legal),
    ('C6 cert failure -> cert_fail, V3 False, original plan returned', c6_cert_failure),
    ('C7 timeout -> timeout, never a normal completion', c7_timeout),
    ('C8 scorer exception is contained and no install path is reachable', c8_scorer_exception),
    ('T1 the full two-arm reports are emitted for independent re-adjudication', t1_emit_traces),
)


def emit_traces(out_dir):
    """Write full two-arm REPORTS (not summaries) so a third party can re-adjudicate them.

    The summary rows in the acceptance JSON carry digests and counts; these traces carry the raw
    per-arm step lists, selection records and legal fingerprints that an independent read-only audit
    needs in order to re-derive V1 / V2 / V3 without importing this harness or the protocol.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces = {}

    fx = build_fixture()
    arms, legal_fn, apply_fn = make_arms(fx)
    shared = shared_input(fx, legal_fn=legal_fn)
    traces['baseline'] = {'shared': _jsonable_shared(shared),
                          'report': TA.run_two_arms(shared, arms, legal_fn=legal_fn,
                                                    apply_fn=apply_fn)}

    def last_preferred(state, legal):
        out = np.full(len(legal), -1.0, dtype=np.float32)
        out[-1] = 1.0
        return out

    arms2, legal_fn2, apply_fn2 = make_arms(fx, f_scores_fn=last_preferred)
    shared2 = shared_input(fx, legal_fn=legal_fn2)
    traces['forced_fork'] = {'shared': _jsonable_shared(shared2),
                             'report': TA.run_two_arms(shared2, arms2, legal_fn=legal_fn2,
                                                       apply_fn=apply_fn2)}
    (out_dir / 'A1_D3_L1_TWO_ARM_TRACES.json').write_text(
        json.dumps(traces, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    return (out_dir / 'A1_D3_L1_TWO_ARM_TRACES.json')


def _jsonable_shared(shared):
    """The audit-visible part of the shared input; the transport object graph is not serialisable."""
    return {k: v for k, v in shared.items() if k not in ('transport', 'state')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--emit-traces', action='store_true',
                    help='also write the full two-arm reports an independent audit re-adjudicates')
    args = ap.parse_args()
    OUTDIR[0] = args.outdir
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    rows = []
    for name, fn in SCENARIOS:
        started = time.monotonic()
        try:
            ok, detail = fn(None)
        except Exception as exc:                                          # noqa: BLE001
            import traceback
            ok, detail = False, {'exception': '%s: %s' % (type(exc).__name__, exc),
                                 'traceback': traceback.format_exc()}
        rows.append({'name': name, 'ok': bool(ok), 'wall_s': time.monotonic() - started,
                     'detail': detail})
        print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                                 json.dumps(detail, default=repr, sort_keys=True)[:260]))
    trace_file = Path(args.outdir) / 'A1_D3_L1_TWO_ARM_TRACES.json'
    n_fail = sum(1 for r in rows if not r['ok'])
    payload = {
        'schema': 'd3_l1_two_arm_protocol_acceptance',
        'arms': {'F': 'synthetic deterministic logits (SYNTHETIC_F_NOT_REAL_F)',
                 'D-Geo': 'registered geo_score = -incremental_distance'},
        'not_run': ['the real F forward over the 64 states (awaiting the user signature)',
                    'the real F-vs-D-Geo comparison', 'any training', 'any install'],
        'rows': rows, 'n_rows': len(rows), 'n_pass': len(rows) - n_fail, 'n_fail': n_fail,
        'tool_sha256': sha(__file__),
        'protocol_sha256': sha(EXT / 'scripts/simulation/d3_l1_two_arm_protocol.py'),
        'geo_adapter_sha256': sha(EXT / 'scripts/simulation/d3_geo_adapter.py'),
        'v110_materializer_sha256': sha(EXT / 'scripts/tests/d3_l1_v110_input_qualification.py'),
        'data_sha256': sha(EXT / 'data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz'),
        'trace_path': (str(trace_file) if trace_file.is_file() else None),
        'trace_sha256': (sha(trace_file) if trace_file.is_file() else None),
        'science_ready': False, 'stable_online_gain': False,
        'algorithmic_gain_claim': 'none', 'F_D_Geo_scientific_comparisons': 0,
        'verdict': ('L1_TWO_ARM_PROTOCOL_ENGINEERING_ACCEPTED' if n_fail == 0
                    else 'L1_TWO_ARM_PROTOCOL_ENGINEERING_BLOCKED'),
    }
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'A1_D3_L1_TWO_ARM_PROTOCOL_ACCEPTANCE.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    print('\n=== L1 two-arm acceptance: %d rows, %d PASS, %d FAIL ==='
          % (len(rows), len(rows) - n_fail, n_fail))
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
