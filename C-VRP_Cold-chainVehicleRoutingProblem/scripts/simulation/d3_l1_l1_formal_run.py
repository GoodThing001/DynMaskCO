#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L1 FORMAL RUN ENTRY: the single command the signer authorises, plus its synthetic acceptance.

Modes
-----
  --freeze-identity   materialise the L1 run identity, the frozen watch set, the budget caps and the
                      output-directory binding, and write the authorization template skeleton
  --supervise         the ONLY legal way to start a real L1 comparison: validate the credential
                      against the frozen identity, consume it ONE-SHOT, publish the execution
                      context, then run the worker under the V1.16 contained supervisor
  --worker            the guarded real worker: refuses before loading anything unless the execution
                      context is present and valid (shared with the V1.22 F path)
  --synthetic-run     run BOTH arms in-process with a SYNTHETIC scorer, for acceptance only.  It
                      writes an L1 report to a throw-away directory, records `synthetic=true`, and
                      can never be mistaken for the real comparison
  --guard-probe       run ONLY the execution-context guard and report its verdict

The real path is deliberately identical in shape to the V1.22 F forward (credential -> consumption ->
execution context -> guarded worker -> contained supervisor -> round adjudication), so the two
benefit from the same hardened boundary instead of a second, weaker one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
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

EV = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
V122_RUN = EV / 'v122_evidence/run_v122'
L1_GATES = EV / 'l1_gates'
PREREG = EV / 'A1_D3_L1_L1_FORMAL_EXECUTION_PREREGISTRATION_V1.md'
RUN_ID = 'd3l1-l1-two-arm-20261015-l1v1'

# The budget the pre-registration fixes.  Carried here as data so the identity, the credential and
# the runtime all read one source.
BUDGET = {
    'mask_steps_m': 2,
    'per_arm_per_state_scorer_calls': 2,
    'per_arm_total_scorer_calls': 128,
    'per_state_wall_clock_s': 10.0,
    'initialize_compile_s': 60.0,
    'batch_wall_clock_s': 180.0,
}


def sha(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    p = Path(path)
    return json.loads(p.read_text(encoding='utf-8')) if p.is_file() else None


def load_helpers():
    import importlib.util
    import d3_l1_v118_authorization as auth
    import d3_l1_v118_persistence as persistence
    import d3_l1_two_arm_protocol as protocol
    import d3_l1_real_arm as real_arm
    spec = importlib.util.spec_from_file_location(
        'l1_control', str(EXT / 'scripts/tests/d3_l1_v116_forward_control.py'))
    control = importlib.util.module_from_spec(spec)
    sys.modules['l1_control'] = control
    spec.loader.exec_module(control)
    return auth, persistence, protocol, real_arm, control


# ---------------------------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------------------------
def build_l1_identity(run_id=RUN_ID, *, created_utc=None):
    """Freeze the L1 run: states, arms, watch set, budgets and the output-directory binding."""
    import d3_l1_v118_run_identity as fwd_identity
    import d3_l1_v121_dependency_closure as closure

    v122_ident = None
    for p in sorted(V122_RUN.glob('RUN_IDENTITY_*.json')):
        v122_ident = json.loads(p.read_text(encoding='utf-8'))
        break
    v122_fwd = None
    for p in sorted(V122_RUN.glob('FORWARD_TARGETS_*.json')):
        v122_fwd = json.loads(p.read_text(encoding='utf-8'))
        break
    if not v122_ident or not v122_fwd:
        raise RuntimeError('the V1.22 run identity/targets are required: the L1 comparison reuses '
                           'the same 64 states and the same pre-qualified inputs')
    states = v122_ident['state_list']
    watch = fwd_identity.watched_identity_paths()
    closure_info = closure.build_watch_set()
    if len(watch) != closure_info['n_watch'] + len(closure_info['non_source_assets']):
        raise RuntimeError('the watch set does not match the derived closure')
    return {
        'schema': 'd3_l1_l1_run_identity',
        'run_id': run_id,
        'created_utc': created_utc or time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'supersedes_nothing': 'this is a NEW artifact; no V1.x run identity is overwritten',
        'reuses': {'run_id': v122_ident['run_id'],
                   'what': 'the 64 frozen states and the pre-qualified inputs, unchanged',
                   'why': 'L1 asks a mechanism question on the states whose F inputs are already '
                          'qualified; re-deriving them would be a new derivation'},
        'state_list': states,
        'state_binding_sha256': v122_ident['state_binding_sha256'],
        'forward_targets_sha256': v122_fwd['targets_sha256'],
        'identities': v122_ident['identities'],
        'contracts': v122_ident['contracts'],
        'dependency_identity_sha256': v122_ident['dependency_identity_sha256'],
        'watch': {str(p): sha(p) for p in watch},
        'watch_paths': [str(p) for p in watch],
        'arms': {
            'F': {'scorer': 'the REAL MaskCO scorer accepted by V1.22',
                  'f_ckpt_sha256': v122_ident['identities']['f_ckpt']['sha256'],
                  'backbone_sha256': v122_ident['identities']['backbone']['sha256'],
                  'label': v122_ident['labels']['F'], 'synthetic_fallback': 'NONE'},
            'D_Geo': {'scorer': 'registered -incremental_distance',
                      'selector': 'geo_unified_minimum', 'eps_rel': 0.0, 'eps_abs': 0.0,
                      'tie_break': 'legal_order_first'},
        },
        'verdicts': {'V1': 'first-step divergence', 'V2': 'full repair per arm',
                     'V3': 'NOT_RUN (no certifier, no install)'},
        'budget': dict(BUDGET),
        'authorization_vocabulary': {
            'note': 'the credential module speaks the F-forward vocabulary; the mapping is exact so '
                    'no new credential schema is invented',
            'warmup_calls': 1, 'calls_per_state': BUDGET['per_arm_per_state_scorer_calls'],
            'max_forward_calls': BUDGET['per_arm_total_scorer_calls'],
            'budget_slot_span_s': BUDGET['per_state_wall_clock_s'],
        },
        'labels': {'science': 'MECHANISM_ONLY_NO_UTILITY_CLAIM',
                   'stable_online_gain': False, 'algorithmic_gain_claim': 'none'},
    }


def freeze_identity(out_dir, run_id=RUN_ID):
    _, persistence, _, _, _ = load_helpers()
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit('refusing to reuse a non-empty L1 run directory: %s' % out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'scores').mkdir(exist_ok=True)
    ident = build_l1_identity(run_id)
    persistence.write_once(out_dir / 'L1_IDENTITY.json', ident)
    persistence.write_once(out_dir / 'L1_WATCH.json',
                           {'run_id': run_id, 'paths': ident['watch'],
                            'n_paths': len(ident['watch'])})
    persistence.write_once(out_dir / 'L1_BUDGET.json',
                           {'run_id': run_id, 'budget': ident['budget']})
    return ident


def load_frozen(run_dir):
    run_dir = Path(run_dir)
    ident = load(run_dir / 'L1_IDENTITY.json')
    if not ident:
        raise SystemExit('the L1 run directory is not frozen: L1_IDENTITY.json is absent')
    return ident


# ---------------------------------------------------------------------------------------------
# the run itself
# ---------------------------------------------------------------------------------------------
def _materialise(day, position):
    """Materialise the shared input for one state through the ACCEPTED V1.10 path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'v110_mod_l1', str(EXT / 'scripts/tests/d3_l1_v110_input_qualification.py'))
    v110 = importlib.util.module_from_spec(spec)
    sys.modules['v110_mod_l1'] = v110
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


def run_one_state(day, position, *, synthetic=False, scorer_bundle=None):
    """Run BOTH arms on ONE frozen state and return the per-state record."""
    _, _, protocol, real_arm, _ = load_helpers()
    from action_contract import apply_action
    from mpre_policy import enumerate_legal_actions

    env, snap, entry, mat, contract = _materialise(day, position)
    mutable = list(mat['mutable_ids'])
    mask = list(mat['mask'])[:BUDGET['mask_steps_m']]

    def legal_fn(P, remaining):
        return enumerate_legal_actions(env, day, P, remaining, mutable)

    def apply_fn(P, action, allowed_vehicle_ids=None):
        return apply_action(P, action, allowed_vehicle_ids=allowed_vehicle_ids)

    ledger = real_arm.Ledger()
    if synthetic:
        # SYNTHETIC ACCEPTANCE ONLY.  Deterministic and distance-independent, so a divergence is
        # meaningful for the harness and meaningless for science; the record says so.
        def f_score_fn(state):
            legal = state.get('legal') or []
            out = np.asarray([np.float32(1.0 if i % 2 == 0 else -1.0) for i in range(len(legal))])
            ledger.note_score_call(protocol.ARM_F, legal, out)
            return out
        model_info = {'synthetic': True, 'label': 'SYNTHETIC_SCORER_NOT_F'}
    else:
        nv_max, m_max = real_arm.frozen_jit_shapes()
        score, model_info = scorer_bundle if scorer_bundle else \
            real_arm.load_real_f_scorer(nv_max=nv_max, m_max=m_max)
        f_score_fn = real_arm.make_score_fn(score, ledger=ledger, arm=protocol.ARM_F)

    extract_fn = real_arm.make_real_extract_fn(
        env=env, inst_idx=day, clock=float(snap['clock']), vehicles=mat['vehicles'],
        served_mask=mat['served'], visible_ids=mat['visible_ids'], p0=mat['P0'], mask=mask,
        mutable_ids=mutable, node_to_local=mat['inputs']['node_to_local'],
        local_to_node=mat['inputs']['local_to_node'], nv=int(mat['inputs']['Nv']))

    def base_extract_shared(P, remaining, legal):
        return {'plans_keys': sorted(P), 'n_remaining': len(remaining), 'n_legal': len(legal)}

    import d3_geo_adapter as GEO
    shared = protocol.build_shared_input(
        env=env, inst_idx=day, day=day, clock=float(snap['clock']), vehicles=mat['vehicles'],
        served_mask=mat['served'], visible_ids=mat['visible_ids'], plans=mat['P0'],
        pool_B=list(mat['pool']), pool_diag=mat['pool_diag'], mask=mask, partial=mat['partial'],
        legal=mat['legal'], mutable_ids=mutable, protected_ids=mat['protected_ids'],
        inputs=mat['inputs'], contract=contract, snapshot=snap, state_hash=entry['state_hash'],
        legal_fn=legal_fn)
    arms = protocol.arm_scorers(f_score_fn, base_extract_fn=base_extract_shared,
                                allowed_vehicle_ids=mutable,
                                f_label=model_info.get('label', real_arm.F_LABEL))
    arms['scorers'][protocol.ARM_D_GEO] = real_arm.make_geo_score_fn(ledger=ledger)
    arms['extract_fn'] = GEO.make_geo_extract_fn(extract_fn, allowed_vehicle_ids=mutable)

    gates = {a: protocol.BudgetGate(
        max_scorer_calls=BUDGET['per_arm_total_scorer_calls'],
        max_wall_seconds=BUDGET['per_state_wall_clock_s']) for a in protocol.ARMS}
    t0 = time.monotonic()
    report = protocol.run_two_arms(shared, arms, legal_fn=legal_fn, apply_fn=apply_fn, gates=gates,
                                   certify_fn=None,
                                   certify_not_run_reason='L1 V3 is NOT_RUN by pre-registration: '
                                                          'no certifier, no install')
    ok_ledger, ledger_report = ledger.cross_check()
    return {
        'day_id': day, 'position': position, 'state_hash': entry['state_hash'],
        'mask': mask, 'n_mask': len(mask),
        'n_legal_first_step': shared['common_conditions']['n_legal'],
        'wall_s': time.monotonic() - t0,
        'V1': report['V1'], 'V2': report['V2'], 'V3': report['V3'],
        'audit': report['audit'],
        'ledger_cross_check': ledger_report, 'ledger_ok': ok_ledger,
        'per_arm_budget': {a: (report['results'][a].get('budget_gate') or {}) for a in protocol.ARMS},
        'per_arm_n_steps': {a: report['results'][a]['n_steps'] for a in protocol.ARMS},
        'per_arm_failure': {a: report['results'][a]['failure_reason'] for a in protocol.ARMS},
        'plan_hashes': {a: report['results'][a]['plan_hash_after'] for a in protocol.ARMS},
        'replay_digest': protocol.replay_digest(report),
        'synthetic': bool(synthetic),
    }


def adjudicate(records, *, synthetic=False):
    """The ROUND verdict over the frozen state list: every state stays in the denominator."""
    divergent = sum(1 for r in records if r['V1']['verdict'] == 'DIVERGENT')
    identical = sum(1 for r in records if r['V1']['verdict'] == 'IDENTICAL')
    not_comparable = sum(1 for r in records if r['V1']['verdict'] == 'NOT_COMPARABLE')
    empty = sum(1 for r in records if r['V1']['verdict'] == 'NOT_APPLICABLE_EMPTY_MASK')
    both_complete = sum(1 for r in records if r['V2']['both_completed'])
    per_arm_complete = {a: sum(1 for r in records if r['V2'][a]['full_repair_completed'])
                        for a in ('F', 'D-Geo')}
    failures = {}
    for r in records:
        for a, reason in r['per_arm_failure'].items():
            if reason:
                failures['%s:%s' % (a, reason)] = failures.get('%s:%s' % (a, reason), 0) + 1
    return {
        'schema': 'd3_l1_l1_round_adjudication',
        'n_states_in_denominator': len(records),
        'V1_counts': {'DIVERGENT': divergent, 'IDENTICAL': identical,
                      'NOT_COMPARABLE': not_comparable, 'NOT_APPLICABLE_EMPTY_MASK': empty},
        'V2_counts': {'both_completed': both_complete, 'per_arm_completed': per_arm_complete,
                      'failures_by_reason': failures},
        'V3': 'NOT_RUN for every state (no certifier was supplied; no plan was installed)',
        'all_ledgers_cross_checked': all(r['ledger_ok'] for r in records),
        'synthetic': bool(synthetic),
        'claim': ('SYNTHETIC ACCEPTANCE ONLY: no scientific reading is permitted from this record'
                  if synthetic else
                  'L1 reports attributable ACTION DIVERGENCE only; not a gain, not a significance '
                  'test, not an online-utility claim, and no divergence is a certified plan'),
    }



def authorization_identity_view(ident):
    """The frozen identity as the credential module sees it.  ONE builder, two call sites.

    Reusing the F-forward vocabulary is deliberate: it means the L1 comparison inherits the same
    hardened credential gate instead of a second, weaker one.  The mapping is exact, not
    approximate -- warm-up 1, calls per state = the L1 per-arm step cap, max calls = the L1 per-arm
    total, slot cap = the L1 per-state wall clock.
    """
    return {
        'run_id': ident['run_id'],
        'identities': ident['identities'],
        'contracts': ident['contracts'],
        'state_list': ident['state_list'],
        'state_binding_sha256': ident['state_binding_sha256'],
        'dependency_identity_sha256': ident['dependency_identity_sha256'],
        'call_schedule': {'warmup_calls': 1,
                          'calls_per_state': BUDGET['per_arm_per_state_scorer_calls'],
                          'max_forward_calls': BUDGET['per_arm_total_scorer_calls']},
        'budgets': {'caps': {'initialize_compile_s': BUDGET['initialize_compile_s'],
                             'slot_span_s': BUDGET['per_state_wall_clock_s'],
                             'batch_s': BUDGET['batch_wall_clock_s']}},
    }


def authorization_targets_view(ident):
    return {'targets_sha256': ident['forward_targets_sha256']}


def worker(out_dir, *, synthetic_run=False):
    """The guarded worker.  Refuses before loading ANYTHING unless the execution context is valid."""
    auth, persistence, protocol, real_arm, _ = load_helpers()
    import d3_l1_v120_execution_context as execctx
    out_dir = Path(out_dir)
    try:
        context = execctx.validate_execution_context(run_dir=out_dir)
    except execctx.ExecutionContextRefused as exc:
        try:
            persistence.write_once(out_dir / 'L1_GUARD_REFUSED.json',
                                   {'stage': 'execution_context_guard', 'refused': True,
                                    'reason': str(exc), 'argv': sys.argv, 'pid': os.getpid(),
                                    'synthetic_run': bool(synthetic_run)})
        except Exception:                                                  # noqa: BLE001
            pass
        print('EXECUTION CONTEXT REFUSED: %s' % exc, file=sys.stderr)
        return 1
    ident = load_frozen(out_dir)
    # identity closure before any scoring
    drift = [p for p, want in (ident.get('watch') or {}).items() if sha(p) != want]
    if drift:
        persistence.write_once(out_dir / 'L1_IDENTITY_DRIFT.json',
                               {'n_drifted': len(drift), 'drifted': drift[:8]})
        print('IDENTITY DRIFT: %d path(s)' % len(drift), file=sys.stderr)
        return 1
    scorer_bundle = None
    if not synthetic_run:
        nv_max, m_max = real_arm.frozen_jit_shapes()
        scorer_bundle = real_arm.load_real_f_scorer(nv_max=nv_max, m_max=m_max)
    states = ident['state_list']['states']
    records, errors = [], []
    for s in states:
        day, pos = int(s['day_id']), int(s['position'])
        try:
            rec = run_one_state(day, pos, synthetic=synthetic_run, scorer_bundle=scorer_bundle)
        except Exception as exc:                                          # noqa: BLE001
            import traceback
            rec = {'day_id': day, 'position': pos, 'state_hash': s.get('state_hash'),
                   'status': 'FAILED_EXCEPTION', 'exception_type': type(exc).__name__,
                   'error': str(exc), 'traceback': traceback.format_exc(), 'synthetic': synthetic_run}
            errors.append({'row': [day, pos], 'error': str(exc)})
        records.append(rec)
    # the per-state records are published as one immutable JSONL plus an adjudication
    raw = '\n'.join(json.dumps(r, sort_keys=True, default=repr) for r in records).encode('utf-8')
    persistence.write_once(out_dir / 'L1_STATE_RECORDS.jsonl', raw)
    verdict = adjudicate([r for r in records if r.get('status') != 'FAILED_EXCEPTION'],
                         synthetic=synthetic_run)
    verdict['n_failed_exception'] = len(errors)
    verdict['failed_exceptions'] = errors[:8]
    persistence.write_once(out_dir / 'L1_ADJUDICATION.json', verdict)
    print(json.dumps({k: v for k, v in verdict.items()
                      if k not in ('failed_exceptions',)}, indent=1, default=repr))
    return 0 if not errors else 1


def supervise(out_dir, *, authorization, preregistration, consumption_root=None, synthetic_run=False):
    """"The only legal real start: authorize -> consume -> execution context -> contained worker."""
    auth, persistence, _, _, control = load_helpers()
    import d3_l1_v120_execution_context as execctx
    out_dir = Path(out_dir)
    ident = load_frozen(out_dir)
    if not authorization:
        raise SystemExit('refusing to start the L1 comparison without an authorization file')
    if not preregistration:
        raise SystemExit('refusing to start without --preregistration: the credential binds its SHA')
    credential = auth.load_credential(authorization)
    view = authorization_identity_view(ident)
    targets_view = authorization_targets_view(ident)
    ok, detail = auth.validate_credential(
        credential, identity=view, forward_targets=targets_view, output_dir=out_dir,
        preregistration_sha256=sha(preregistration))
    if not ok:
        persistence.write_once(out_dir / 'L1_AUTHORIZATION_REFUSED.json',
                               {'granted': False, 'problems': detail['problems']})
        raise SystemExit('AUTHORIZATION REFUSED: %s' % json.dumps(detail['problems']))
    granted = auth.authorize(
        credential, identity=view, forward_targets=targets_view, output_dir=out_dir,
        preregistration_path=Path(preregistration), run_dir=out_dir,
        consumption_root=consumption_root)
    published = execctx.publish_execution_context(out_dir, authorization_result=granted)
    env = dict(os.environ, JAX_PLATFORMS='cpu', PYTHONIOENCODING='utf-8')
    env.update(published['env'])
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--out-dir', str(out_dir)]
    if synthetic_run:
        command.append('--synthetic-run')
    # the containment identity is the L1 watch set
    identity = dict(ident['watch'])
    result = control.run_contained(
        command, out_dir, identity, init_cap=float(BUDGET['initialize_compile_s']),
        slot_cap=float(BUDGET['per_state_wall_clock_s']),
        batch_cap=float(BUDGET['batch_wall_clock_s']), env=env)
    persistence.write_once(out_dir / 'L1_SUPERVISION.json',
                           {k: v for k, v in result.items() if k != 'end_identity'})
    print(json.dumps({'reason': result.get('reason'), 'ok': result.get('ok'),
                      'returncode': result.get('returncode'),
                      'tree_active_after': result.get('tree_active_after')}, indent=1, default=repr))
    return 0 if result.get('ok') else 1


def synthetic_run(out_dir):
    """BOTH arms in-process with a synthetic scorer, for acceptance.  Never the real comparison."""
    out_dir = Path(out_dir)          # main() hands over a string; every path join needs a Path
    ident = load_frozen(out_dir)
    records = []
    for s in ident['state_list']['states'][:3]:
        records.append(run_one_state(int(s['day_id']), int(s['position']), synthetic=True))
    verdict = adjudicate(records, synthetic=True)
    _, persistence, _, _, _ = load_helpers()
    persistence.write_once(out_dir / 'L1_SYNTHETIC_ACCEPTANCE.json',
                           {'verdict': verdict, 'records': records,
                            'scope': 'SYNTHETIC ACCEPTANCE ONLY: three states, synthetic scorer, '
                                     'no scientific reading permitted'})
    print(json.dumps(verdict, indent=1, default=repr))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze-identity', action='store_true')
    ap.add_argument('--supervise', action='store_true')
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--synthetic-run', action='store_true')
    ap.add_argument('--guard-probe', action='store_true')
    ap.add_argument('--out-dir')
    ap.add_argument('--run-id', default=RUN_ID)
    ap.add_argument('--authorization')
    ap.add_argument('--preregistration')
    ap.add_argument('--consumption-root')
    args = ap.parse_args()
    if not args.out_dir:
        raise SystemExit('--out-dir is required')
    if args.freeze_identity:
        ident = freeze_identity(args.out_dir, args.run_id)
        print(json.dumps({'run_id': ident['run_id'], 'n_watch': len(ident['watch']),
                          'n_states': ident['state_list']['n_states'],
                          'state_binding_sha256': ident['state_binding_sha256'],
                          'budget': ident['budget']}, indent=1, default=repr))
        return 0
    if args.guard_probe:
        import d3_l1_v120_execution_context as execctx
        try:
            ctx = execctx.validate_execution_context(run_dir=Path(args.out_dir))
        except execctx.ExecutionContextRefused as exc:
            print(json.dumps({'guard': 'REFUSED', 'reason': str(exc)}, indent=1, default=repr))
            return 1
        print(json.dumps({'guard': 'PASSED', 'run_id': ctx.get('run_id'),
                          'nonce': ctx.get('nonce'), 'loaded_anything': False}, indent=1))
        return 0
    if args.worker:
        return worker(args.out_dir, synthetic_run=args.synthetic_run)
    if args.supervise:
        # `--synthetic-run` is a MODIFIER here (substitute the scorer, keep the real supervision
        # path); checking the standalone branch first made the supervised acceptance unreachable.
        return supervise(args.out_dir, authorization=args.authorization,
                         preregistration=args.preregistration,
                         consumption_root=args.consumption_root,
                         synthetic_run=args.synthetic_run)
    if args.synthetic_run:
        return synthetic_run(args.out_dir)
    raise SystemExit('HOLD: pass --freeze-identity, --supervise, --worker, --synthetic-run or '
                     '--guard-probe')


if __name__ == '__main__':
    sys.exit(main())
