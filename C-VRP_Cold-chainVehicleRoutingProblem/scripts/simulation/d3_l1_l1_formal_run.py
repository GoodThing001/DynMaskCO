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
# The pre-registration the signature is issued against.  It MUST be the version whose §7 registers
# the identity this entry actually freezes: V1 registered a 34-path F-ONLY watch set while the frozen
# identity carries 42 L1 paths, so V1 can no longer be the basis of a signature.  V1 is preserved
# byte-intact and superseded, never edited.
PREREG = EV / 'A1_D3_L1_L1_FORMAL_EXECUTION_PREREGISTRATION_V2.md'
PREREG_VERSION = 'L1-v2'
PREREG_SUPERSEDES = EV / 'A1_D3_L1_L1_FORMAL_EXECUTION_PREREGISTRATION_V1.md'
# A NEW run id.  The identity, the watch set, the order-confound rule and the pre-registration all
# changed, so reusing the old id would present a new execution as the old one.
RUN_ID = 'd3l1-l1-two-arm-20261016-l1v2'

# The budget the pre-registration fixes.  Carried here as data so the identity, the credential and
# the runtime all read one source.  These are the FROZEN 60 / 10 / 180 s caps: no fix below changes
# any of them.
BUDGET = {
    'mask_steps_m': 2,
    'per_arm_per_state_scorer_calls': 2,
    'per_arm_total_scorer_calls': 128,
    'per_state_wall_clock_s': 10.0,
    'initialize_compile_s': 60.0,
    'batch_wall_clock_s': 180.0,
}

# BLOCKER B(3).  `per_state_wall_clock_s` is the allowance for ONE STATE COMPARISON -- both arms
# together -- NOT ten seconds per arm.  Both arms' gates are therefore charged from ONE monotonic
# origin, so the second arm pays for the time the first one spent.  The alternative reading (10 s for
# F plus 10 s for D-Geo) would let a single state consume 20 s while every durable record still
# declared "10 s", which is precisely the inconsistency that had to be removed rather than papered
# over.  The per-arm CALL bound stays per arm because the pre-registration names it per arm.
WALL_CLOCK_SCOPE = 'per_state_both_arms_share_one_10s_allowance'

# THE ORDER CONFOUND, PRE-REGISTERED RATHER THAN DISCOVERED LATER.
#
# The arms run SEQUENTIALLY in the frozen order (`ARMS`), charged from one origin.  So the SECOND
# arm's remaining time is `10.0 s - (what the first arm spent)`, and its budget therefore depends on
# the first arm's COST.  A cheaper first arm leaves the second arm more room; a slower first arm
# leaves it less.  That is a property of the instrument, not of either scorer, and the classification
# below is what keeps it from being read as a mechanism difference:
#
#   * `V1` (the first-step comparison) is ORDER-INVARIANT BY CONSTRUCTION -- no scorer call happens
#     before the first step, so the second arm has spent nothing and no deadline can have fired.
#     V1 can never be order-sensitive, so this rule does not touch the primary verdict at all.
#   * `V2` (per-arm completion) IS order-sensitive, and only through `timeout` on the SECOND arm.
#
# The rule is applied to the RECORDED OUTCOME, by a deterministic function, at adjudication time.  It
# is not a post-hoc adjustment of any measurement: the measurement is reported unchanged, and what the
# rule fixes is the INTERPRETATION -- which the pre-registration now states in advance, exactly as the
# task requires ("不得事后依据结果调整").
ORDER_CONFOUND_RULE = (
    'a state whose ONLY inter-arm difference is a `timeout` on the SECOND arm, while the arms ran in '
    'the frozen sequential order under the shared per-state allowance, is classified as '
    'ORDER_ATTRIBUTABLE.  Its V2 difference is reported but may NOT be read as a scorer mechanism '
    'advantage: the second arm was charged for time it did not spend.  A `timeout` on the FIRST arm '
    'is never order-attributable.  Every affected state keeps its record and stays in the 64-state '
    'denominator; none is dropped, replaced or re-run.')

# The SEQUENTIAL execution order the protocol freezes (`d3_l1_two_arm_protocol.ARMS`).  Repeated here
# as a constant because the order is now part of what the pre-registration states, and a record must be
# readable without importing the protocol.  `blocker_acceptance` asserts the two agree.
ARM_ORDER = ('F', 'D-Geo')


def classify_order_confound(record):
    """`str`: this state's V2 timing attribution.  Deterministic, computed from the record alone.

    Returns one of:
      `NOT_ORDER_ATTRIBUTABLE`   no timeout, or the first arm timed out, or the order is unknown;
      `ORDER_ATTRIBUTABLE`       the second arm timed out and the first did not -- the shared
                                 allowance consumed by the first arm is a viable cause, so the V2
                                 difference may not be read as a mechanism difference;
      `ORDER_INDETERMINATE`      BOTH arms timed out, so the state carries no usable completion
                                 evidence under this budget and neither arm may be called better.
    """
    v2 = record.get('V2') or {}
    order = v2.get('arm_order') or []
    failures = record.get('per_arm_failure') or {}
    if len(order) < 2 or not failures:
        return 'NOT_ORDER_ATTRIBUTABLE'
    first, second = order[0], order[1]
    f_first = failures.get(first) == 'timeout'
    f_second = failures.get(second) == 'timeout'
    if f_first and f_second:
        return 'ORDER_INDETERMINATE'
    if f_second and not f_first:
        return 'ORDER_ATTRIBUTABLE'
    return 'NOT_ORDER_ATTRIBUTABLE'

# The pre-registered denominator.  Every state in the frozen list keeps a record; a state that could
# not be compared is REPORTED as such and is never deleted from the denominator.
EXPECTED_STATES = 64


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
def _registered_watch_count(prereg_path):
    """`int | None`: the watch count the pre-registration's §7 STATES, parsed from the document.

    The V2 pre-registration is generated from the identity manifest, so §7 carries the count in a
    fixed form (`运行期 watch 集 = N 条**（S 个源码文件 + A 项非源码资产）`).  Reading it back here is
    what turns "the document and the identity agree" into a checked precondition instead of a hope:
    the identity refuses to materialise when the two numbers differ.
    """
    p = Path(prereg_path)
    if not p.is_file():
        return None
    import re
    m = re.search(r'运行期 watch 集 = (\d+) 条', p.read_text(encoding='utf-8'))
    return int(m.group(1)) if m else None


def build_l1_identity(run_id=RUN_ID, *, created_utc=None):
    """Freeze the L1 run: states, arms, watch set, budgets and the output-directory binding."""
    import d3_l1_closure as l1_closure

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

    # BLOCKER A.  The watch set must be derived from THIS entry point.  The F-forward closure left
    # `l1_sources_in_closure == []` while M7 still passed: it re-hashed code that decides the F
    # forward and never saw `d3_l1_l1_formal_run.py`, `d3_l1_real_arm.py`, the two-arm protocol, the
    # D-Geo adapter or the V1.10 state materialisation -- i.e. the modules that decide the L1 result.
    # `d3_l1_closure` seeds from this file and walks the transitive local-import graph, so the freeze
    # and the freeze CHECK now come from one derivation.
    derived = l1_closure.build_watch_set()
    if derived['unresolved_seeds'] or derived['excluded_reachable']:
        raise RuntimeError('the derived L1 closure is not clean: unresolved=%s excluded_reachable=%s'
                           % (derived['unresolved_seeds'], derived['excluded_reachable']))
    watch, missing = [], []
    for rel in derived['watch']:
        p = REPO / rel
        (watch if p.is_file() else missing).append(str(p))
    for root, rel in l1_closure.NON_SOURCE_ASSETS:
        p = (EXT if root == 'ext' else REPO) / rel
        (watch if p.is_file() else missing).append(str(p))
    if missing:
        raise RuntimeError('the L1 watch set names %d non-existent path(s): %s'
                           % (len(missing), missing))
    if len(set(watch)) != len(watch):
        raise RuntimeError('the derived L1 watch set contains duplicates')
    # The critical check reads the DERIVED repo-relative list, not absolute paths: comparing an
    # absolute Windows path against a repo-relative suffix is how this check first mis-fired.
    critical = list(l1_closure.critical_l1_sources())
    critical_missing = [c for c in critical if c not in set(derived['watch'])]
    if critical_missing:
        raise RuntimeError('the L1 watch set omits decision-relevant module(s): %s'
                           % critical_missing)
    # The pre-registration must be PRESENT and must register the count this identity freezes.  If a
    # signer is handed a document whose §7 says a different number, the two artifacts disagree and
    # neither can authorise anything -- so this refuses rather than warning.
    prereg_sha = sha(PREREG)
    if prereg_sha is None:
        raise RuntimeError('the execution pre-registration is absent: %s' % PREREG)
    prereg_n = _registered_watch_count(PREREG)
    if prereg_n is None:
        raise RuntimeError('the pre-registration does not state a watch count in the expected '
                           'machine-readable form; it cannot be checked against the identity')
    if prereg_n != len(watch):
        raise RuntimeError('the pre-registration registers %d watched path(s) but this identity '
                           'freezes %d; a signature must never be issued across that disagreement'
                           % (prereg_n, len(watch)))
    prereg_binding = {
        'path': str(PREREG.relative_to(REPO)),
        'sha256': prereg_sha,
        'version': PREREG_VERSION,
        'supersedes': {'path': str(PREREG_SUPERSEDES.relative_to(REPO)),
                       'sha256': sha(PREREG_SUPERSEDES),
                       'reason': 'V1 §7 registered a 34-path F-ONLY watch set while this identity '
                                 'freezes the L1 closure; an identity and a pre-registration that '
                                 'disagree cannot share a signature',
                       'not_edited': True, 'not_deleted': True},
        'n_watch_registered_here': prereg_n,
        'agrees_with_the_frozen_watch_set': prereg_n == len(watch),
    }
    return {
        'schema': 'd3_l1_l1_run_identity',
        'run_id': run_id,
        'created_utc': created_utc or time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'supersedes_nothing': 'this is a NEW artifact; no V1.x run identity is overwritten',
        'preregistration': prereg_binding,
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
        'watch': {p: sha(p) for p in watch},
        'watch_paths': watch,
        'watch_derivation': {
            'basis': 'transitive local-import closure seeded from the L1 formal entry',
            'module': 'scripts/data/d3_l1_closure.py',
            'n_source_paths': len(derived['watch']),
            'n_non_source_assets': len(l1_closure.NON_SOURCE_ASSETS),
            'n_total': len(watch),
            'critical_l1_sources': critical,
            'critical_missing': critical_missing,
            'unresolved_seeds': derived['unresolved_seeds'],
            'excluded_reachable': derived['excluded_reachable'],
            'non_source_assets_note': derived['non_source_assets_note'],
            'why_not_the_f_closure': 'the F-forward closure omitted every module that decides the '
                                     'L1 comparison; reusing it let the freeze gate pass with '
                                     'l1_sources_in_closure == []',
        },
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


def run_one_state(day, position, *, synthetic=False, scorer_bundle=None, wall_origin_mono=None,
                  warmup_only=False):
    """Run BOTH arms on ONE frozen state and return the per-state record.

    `wall_origin_mono` is the single monotonic origin BOTH arms are charged from, so the frozen
    `per_state_wall_clock_s` bounds the state comparison rather than each arm (see
    ``WALL_CLOCK_SCOPE``).  `warmup_only` materialises the state and makes exactly ONE compiled F
    forward -- the initialization/compile step the READY record reports -- without running either arm
    for a verdict, so the warm-up can never contribute a comparison result.
    """
    _, _, protocol, real_arm, _ = load_helpers()
    from action_contract import apply_action
    from mpre_policy import enumerate_legal_actions

    if wall_origin_mono is None:
        wall_origin_mono = time.monotonic()
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

    import d3_geo_adapter as GEO
    shared = protocol.build_shared_input(
        env=env, inst_idx=day, day=day, clock=float(snap['clock']), vehicles=mat['vehicles'],
        served_mask=mat['served'], visible_ids=mat['visible_ids'], plans=mat['P0'],
        pool_B=list(mat['pool']), pool_diag=mat['pool_diag'], mask=mask, partial=mat['partial'],
        legal=mat['legal'], mutable_ids=mutable, protected_ids=mat['protected_ids'],
        inputs=mat['inputs'], contract=contract, snapshot=snap, state_hash=entry['state_hash'],
        legal_fn=legal_fn)

    if warmup_only:
        # ONE compiled forward on the FIRST state, through the SAME gated scorer the comparison uses.
        # This is the call the READY record accounts for; it produces no verdict and touches no arm.
        #
        # The initial `remaining` is the MASK, exactly as `stepwise_sample_d3` sets it (`remaining =
        # list(mask)`), NOT the mutable-id list: the two differ, and passing the mutable ids here made
        # the extractor look up a customer outside the node map (`KeyError: 3`).  The warm-up has to
        # reproduce the real first step, or it compiles a different problem than the run uses.
        import hashlib as _hashlib
        legal0 = legal_fn(mat['P0'], mask)
        warm_state = {'legal': legal0, 'legal_order': legal0,
                      'timestep': 0.0, 'plans_keys': sorted(mat['P0'])}
        warm_state.update(extract_fn(mat['P0'], mask, legal0))
        t0 = time.monotonic()
        warm = np.asarray(f_score_fn(warm_state))
        return {'day_id': day, 'position': position, 'state_hash': entry['state_hash'],
                'warmup': True, 'n_warmup_calls': 1, 'wall_s': time.monotonic() - t0,
                'n_legal': len(legal0), 'remaining_is_mask': True,
                'scores_shape': list(warm.shape), 'scores_dtype': warm.dtype.str,
                'scores_sha256': _hashlib.sha256(warm.tobytes()).hexdigest(),
                'model_info': {k: v for k, v in model_info.items() if k != 'model'},
                'label': model_info.get('label', real_arm.F_LABEL)}

    def base_extract_shared(P, remaining, legal):
        return {'plans_keys': sorted(P), 'n_remaining': len(remaining), 'n_legal': len(legal)}

    arms = protocol.arm_scorers(f_score_fn, base_extract_fn=base_extract_shared,
                                allowed_vehicle_ids=mutable,
                                f_label=model_info.get('label', real_arm.F_LABEL))
    arms['scorers'][protocol.ARM_D_GEO] = real_arm.make_geo_score_fn(ledger=ledger)
    arms['extract_fn'] = GEO.make_geo_extract_fn(extract_fn, allowed_vehicle_ids=mutable)

    gates = {a: protocol.BudgetGate(
        max_scorer_calls=BUDGET['per_arm_total_scorer_calls'],
        max_wall_seconds=BUDGET['per_state_wall_clock_s'],
        started_mono=wall_origin_mono) for a in protocol.ARMS}
    report = protocol.run_two_arms(shared, arms, legal_fn=legal_fn, apply_fn=apply_fn, gates=gates,
                                   certify_fn=None,
                                   certify_not_run_reason='L1 V3 is NOT_RUN by pre-registration: '
                                                          'no certifier, no install')
    ok_ledger, ledger_report = ledger.cross_check()
    per_arm_calls = ledger_report['per_arm_calls']
    # BLOCKER B(3) -- THE SEMANTICS OF THE 10 s, MADE MECHANICAL.  `per_state_wall_clock_s` bounds the
    # WHOLE STATE COMPARISON (both arms charged from ONE origin), not each arm separately.  Two
    # separate ten-second allowances would let a state spend twenty seconds while every record still
    # said "10 s", which is the inconsistency this run must not ship.  The pre-registered
    # `per_arm_total_scorer_calls` still bounds each arm's CALLS independently, and the call bound is
    # per arm by name and by intent.
    wall_s = time.monotonic() - wall_origin_mono
    ledger_problems = list(ledger_report['problems'])
    for a in protocol.ARMS:
        if per_arm_calls.get(a, 0) > BUDGET['per_arm_per_state_scorer_calls']:
            ledger_problems.append({'arm': a, 'per_state_calls': per_arm_calls.get(a, 0),
                                    'cap': BUDGET['per_arm_per_state_scorer_calls']})
    ok_ledger = bool(ok_ledger and not ledger_problems)
    rec = {
        'day_id': day, 'position': position, 'state_hash': entry['state_hash'],
        'status': 'COMPLETED',
        'mask': mask, 'n_mask': len(mask),
        'n_legal_first_step': shared['common_conditions']['n_legal'],
        'wall_s': wall_s,
        'per_state_wall_clock_s': BUDGET['per_state_wall_clock_s'],
        'wall_clock_scope': WALL_CLOCK_SCOPE,
        'wall_origin_shared_across_arms': True,
        'arm_order': list(protocol.ARMS),
        'V1': report['V1'], 'V2': report['V2'], 'V3': report['V3'],
        'audit': report['audit'],
        'ledger_cross_check': ledger_report, 'ledger_ok': ok_ledger,
        'ledger_problems': ledger_problems,
        'per_arm_budget': {a: (report['results'][a].get('budget_gate') or {}) for a in protocol.ARMS},
        'per_arm_n_steps': {a: report['results'][a]['n_steps'] for a in protocol.ARMS},
        'per_arm_failure': {a: report['results'][a]['failure_reason'] for a in protocol.ARMS},
        'plan_hashes': {a: report['results'][a]['plan_hash_after'] for a in protocol.ARMS},
        'replay_digest': protocol.replay_digest(report),
        'synthetic': bool(synthetic),
    }
    # the order/timing attribution is part of the RECORD, not a reading added later
    rec['v2_order_attribution'] = classify_order_confound(rec)
    rec['order_confound_rule'] = ORDER_CONFOUND_RULE
    return rec


def blank_state_record(day, position, *, state_hash=None, status, synthetic=False, **extra):
    """A record for a pre-registered state that produced no comparison.

    BLOCKER C.  The denominator is the 64 PRE-REGISTERED STATES, so a state that raised, timed out or
    was never attempted still gets its own record.  Filtering such states out of the list before
    adjudication would silently shrink the denominator to "the states that happened to work", which is
    the defect this function exists to make impossible.
    """
    rec = {'day_id': day, 'position': position, 'state_hash': state_hash, 'status': status,
           'mask': None, 'n_mask': None, 'n_legal_first_step': None, 'wall_s': None,
           'per_state_wall_clock_s': BUDGET['per_state_wall_clock_s'],
           'wall_clock_scope': WALL_CLOCK_SCOPE, 'wall_origin_shared_across_arms': None,
           'V1': {'verdict': 'NOT_ATTEMPTED', 'reason': status},
           'V2': {'both_completed': False, 'F': {'full_repair_completed': False},
                  'D-Geo': {'full_repair_completed': False}},
           'V3': {'verdict': 'NOT_RUN', 'reason': 'no certifier, no install'},
           'arm_order': None, 'v2_order_attribution': 'NOT_ORDER_ATTRIBUTABLE',
           'audit': None, 'ledger_cross_check': {'n_calls': 0, 'per_arm_calls': {},
                                                 'problems': [], 'note': 'no scoring took place'},
           'ledger_ok': False, 'ledger_problems': [{'reason': 'no ledger: the state never scored'}],
           'per_arm_budget': {}, 'per_arm_n_steps': {}, 'per_arm_failure': {},
           'plan_hashes': {}, 'replay_digest': None, 'synthetic': bool(synthetic)}
    rec.update(extra)
    return rec


def adjudicate(records, *, synthetic=False, expected_states=EXPECTED_STATES):
    """The ROUND verdict over the FROZEN state list.  Every state keeps its record.

    BLOCKER C.  The previous version dropped `FAILED_EXCEPTION` states before adjudicating, so a run
    in which one state crashed reported its verdict over 63 states while the report still read like a
    round verdict.  Here the denominator is the PRE-REGISTERED list:
      * `n_states_in_denominator` counts the records passed in and is cross-checked against
        `expected_states`; a missing state is a BLOCKED round, not a smaller denominator;
      * every exception / timeout / empty-pool / not-comparable / not-attempted state is listed
        SEPARATELY with its own row, so the counts add up to the denominator;
      * `ledger_ok=false`, a missing state or an identity mismatch sets `status='BLOCKED'`, so none of
        them can be promoted to a formal PASS.
    """
    keys = [(int(r['day_id']), int(r['position'])) for r in records]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    states_ok = len(records) == int(expected_states) and not duplicates

    divergent = sum(1 for r in records if (r.get('V1') or {}).get('verdict') == 'DIVERGENT')
    identical = sum(1 for r in records if (r.get('V1') or {}).get('verdict') == 'IDENTICAL')
    not_comparable = sum(1 for r in records if (r.get('V1') or {}).get('verdict') == 'NOT_COMPARABLE')
    empty = sum(1 for r in records
                if (r.get('V1') or {}).get('verdict') == 'NOT_APPLICABLE_EMPTY_MASK')
    not_attempted = sum(1 for r in records
                        if (r.get('V1') or {}).get('verdict') == 'NOT_ATTEMPTED')
    exempt = sum(1 for r in records
                 if r.get('status') == 'EXEMPT_EMPTY_MASK')
    exceptions = [{'day_id': r['day_id'], 'position': r['position'],
                   'exception_type': r.get('exception_type'), 'error': r.get('error'),
                   'phase': r.get('failure_phase')}
                  for r in records if r.get('status') == 'FAILED_EXCEPTION']
    timeouts = [{'day_id': r['day_id'], 'position': r['position'],
                 'wall_s': r.get('wall_s'), 'per_arm_failure': r.get('per_arm_failure')}
                for r in records
                if r.get('status') == 'TIMEOUT'
                or any('timeout' in str(v) for v in (r.get('per_arm_failure') or {}).values())]
    both_complete = sum(1 for r in records if (r.get('V2') or {}).get('both_completed'))
    per_arm_complete = {a: sum(1 for r in records
                               if ((r.get('V2') or {}).get(a) or {}).get('full_repair_completed'))
                        for a in ('F', 'D-Geo')}
    failures = {}
    for r in records:
        for a, reason in (r.get('per_arm_failure') or {}).items():
            if reason:
                failures['%s:%s' % (a, reason)] = failures.get('%s:%s' % (a, reason), 0) + 1

    ledgers_bad = [{'day_id': r['day_id'], 'position': r['position'],
                    'status': r.get('status'), 'problems': (r.get('ledger_problems') or [])[:4]}
                   for r in records if not r.get('ledger_ok')]
    identity_bad = [{'day_id': r['day_id'], 'position': r['position'],
                     'reason': (r.get('audit') or {}).get('state_identity_agrees_both_arms')}
                    for r in records
                    if r.get('audit') is not None
                    and not (r['audit'].get('single_factor_ok') is True)]

    blocking = []
    if not states_ok:
        blocking.append('the denominator is not the pre-registered state list: %d record(s) for %d '
                        'pre-registered state(s), duplicates=%s'
                        % (len(records), expected_states, duplicates[:4]))
    if ledgers_bad:
        blocking.append('%d state(s) failed the scorer-ledger cross-check (ledger_ok=false); a round '
                        'with an unverified ledger is never a formal PASS' % len(ledgers_bad))
    if identity_bad:
        blocking.append('%d state(s) failed the single-factor/shared-state identity audit'
                        % len(identity_bad))
    if exceptions:
        blocking.append('%d pre-registered state(s) raised an exception during the comparison'
                        % len(exceptions))

    return {
        'schema': 'd3_l1_l1_round_adjudication',
        'status': 'BLOCKED' if blocking else 'ADJUDICATED',
        'blocking_reasons': blocking,
        'n_states_in_denominator': len(records),
        'n_states_expected': int(expected_states),
        'denominator_is_the_preregistered_list': states_ok,
        'duplicate_states': duplicates,
        'V1_counts': {'DIVERGENT': divergent, 'IDENTICAL': identical,
                      'NOT_COMPARABLE': not_comparable,
                      'NOT_APPLICABLE_EMPTY_MASK': empty, 'NOT_ATTEMPTED': not_attempted},
        'outcome_counts': {'COMPLETED': sum(1 for r in records if r.get('status') == 'COMPLETED'),
                           'FAILED_EXCEPTION': len(exceptions),
                           'TIMEOUT': len(timeouts),
                           'EXEMPT_EMPTY_MASK': exempt,
                           'NOT_ATTEMPTED': not_attempted},
        'V2_counts': {'both_completed': both_complete, 'per_arm_completed': per_arm_complete,
                      'failures_by_reason': failures},
        'V3': 'NOT_RUN for every state (no certifier was supplied; no plan was installed)',
        # ---- the arm-order / shared-allowance confound, reported and bounded -------------------
        'arm_order': {'order': list(ARM_ORDER),
                      'wall_clock_scope': WALL_CLOCK_SCOPE,
                      'wall_clock_scope_s': BUDGET['per_state_wall_clock_s'],
                      'consequence': 'the SECOND arm is charged for the time the FIRST arm spent, so '
                                     'its remaining allowance is `10.0 s - first_arm_cost`',
                      'rule': ORDER_CONFOUND_RULE,
                      'v1_is_order_invariant': (
                          'V1 compares the FIRST step, and no scorer call happens before it, so the '
                          'second arm has spent nothing and no deadline can have fired.  V1 therefore '
                          'cannot be order-sensitive and no V2 order effect can reach it.'),
                      'v2_can_be_order_sensitive': (
                          'V2 compares per-arm completion, which is exactly what the shared deadline '
                          'bounds, so a second-arm `timeout` may be caused by the order rather than '
                          'by the scorer'),
                      'counts': {
                          'ORDER_ATTRIBUTABLE': sum(
                              1 for r in records
                              if r.get('v2_order_attribution') == 'ORDER_ATTRIBUTABLE'),
                          'ORDER_INDETERMINATE': sum(
                              1 for r in records
                              if r.get('v2_order_attribution') == 'ORDER_INDETERMINATE'),
                          'NOT_ORDER_ATTRIBUTABLE': sum(
                              1 for r in records
                              if r.get('v2_order_attribution') == 'NOT_ORDER_ATTRIBUTABLE')},
                      'order_attributable_states': [
                          {'day_id': r['day_id'], 'position': r['position'],
                           'first_arm': ARM_ORDER[0], 'second_arm': ARM_ORDER[1],
                           'per_arm_failure': r.get('per_arm_failure')}
                          for r in records
                          if r.get('v2_order_attribution') == 'ORDER_ATTRIBUTABLE'],
                      'must_not_be_read_as': (
                          'a V2 difference on an ORDER_ATTRIBUTABLE state is NOT evidence that either '
                          'scorer repairs better or worse; it is a consequence of the sequential order '
                          'competing for one pre-registered per-state allowance'),
                      'no_state_is_removed': (
                          'every state affected by the order keeps its record and stays in the '
                          '64-state denominator; nothing is dropped, replaced or re-run, and no '
                          'forbidden retrospective adjustment is applied to any measurement')},
        'all_ledgers_cross_checked': not ledgers_bad,
        'ledgers_not_cross_checked': ledgers_bad,
        'single_factor_failures': identity_bad,
        # every state that did not produce a full comparison is listed with its own row
        'exception_states': exceptions,
        'timeout_states': timeouts,
        'not_comparable_states': [{'day_id': r['day_id'], 'position': r['position']} for r in records
                                  if (r.get('V1') or {}).get('verdict') == 'NOT_COMPARABLE'],
        'empty_pool_states': [{'day_id': r['day_id'], 'position': r['position']} for r in records
                              if (r.get('V1') or {}).get('verdict')
                              == 'NOT_APPLICABLE_EMPTY_MASK'],
        'not_attempted_states': [{'day_id': r['day_id'], 'position': r['position']} for r in records
                                 if (r.get('V1') or {}).get('verdict') == 'NOT_ATTEMPTED'],
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

    `watch` IS part of the view.  Leaving it out was a real defect, not a simplification: the
    credential module derives the code binding (`source_identity_sha256`) FROM `watch`, so a view
    without it made that binding `None` on both sides and the comparison "agreed" -- the credential
    declared the run id, the weights, the data, the contract, the state order and the budgets, and
    NOT ONE BYTE of the code that would execute.
    """
    return {
        'run_id': ident['run_id'],
        'identities': ident['identities'],
        'contracts': ident['contracts'],
        'state_list': ident['state_list'],
        'state_binding_sha256': ident['state_binding_sha256'],
        'dependency_identity_sha256': ident['dependency_identity_sha256'],
        'watch': ident.get('watch') or {},
        'preregistration': ident.get('preregistration'),
        'call_schedule': {'warmup_calls': 1,
                          'calls_per_state': BUDGET['per_arm_per_state_scorer_calls'],
                          'max_forward_calls': BUDGET['per_arm_total_scorer_calls']},
        'budgets': {'caps': {'initialize_compile_s': BUDGET['initialize_compile_s'],
                             'slot_span_s': BUDGET['per_state_wall_clock_s'],
                             'batch_s': BUDGET['batch_wall_clock_s']}},
    }


def authorization_targets_view(ident):
    return {'targets_sha256': ident['forward_targets_sha256']}


def make_l1_ready_probe(out_dir):
    """The V1.22-accepted READY probe, reused VERBATIM in semantics for the L1 worker.

    BLOCKER B.  The L1 supervisor used to call `run_contained` with NO `ready_probe`, so it fell back
    to the V1.11 default (`READY.json` exists) -- a fixed name the V1.18+ worker can NEVER publish,
    because replacing a fixed destination under a reader is the `WinError 5` failure the V1.11 run
    died of.  The supervisor therefore never saw initialization complete, and the honest READY
    record the worker publishes was ignored.  This probe is bound to ONE run directory and accepts
    the record only when it is retrievable through the verified reader AND is a well-formed readiness
    statement, so an empty or truncated file cannot forge READY -- and it is the SAME convention the
    V1.22 F path already proved, not a second, weaker one.
    """
    import d3_l1_v118_persistence as _persistence
    reader = _persistence.VerifiedReader(Path(out_dir))

    def probe(_out):
        record = reader.latest('L1_READY')
        if record is None:
            return {'ready': False, 'probe': 'immutable L1_READY record', 'reason': 'absent',
                    'run_dir': str(out_dir)}
        problems = []
        if record.get('kind') != 'initialization_and_compilation_finished':
            problems.append('kind is %r' % record.get('kind'))
        if not record.get('run_id'):
            problems.append('run_id absent')
        if record.get('n_warmup_calls') != 1:
            problems.append('n_warmup_calls is %r' % record.get('n_warmup_calls'))
        if not record.get('warmup_scores_sha256'):
            problems.append('warmup_scores_sha256 absent')
        if not isinstance(record.get('mono'), (int, float)):
            problems.append('mono absent')
        return {'ready': not problems, 'probe': 'immutable L1_READY record',
                'published_name': record.get('published_name'), 'run_id': record.get('run_id'),
                'n_records': reader.count('L1_READY'), 'problems': problems,
                'run_dir': str(out_dir)}
    return probe


def live_identity_view(ident):
    """`(view, drift_rows)`: the frozen identity RE-DERIVED from the bytes on disk right now.

    BLOCKER A, the half that is easy to miss.  `validate_credential` compares the credential's
    declared fields against the identity it is HANDED.  Handing it `L1_IDENTITY.json` -- a record
    written at freeze time -- only proves the credential agrees with a FILE, not that the code the
    credential authorises still exists.  With that alone, a module edited after the freeze still
    validated and the credential was still consumed: the freeze bound the record, not the run.

    So the supervisor hands over this view instead: the same identity with every watched digest
    recomputed from disk.  A drifted module then makes the live view disagree with the credential's
    declarations, `validate_credential` returns not-ok, and `authorize` refuses BEFORE spending the
    one-shot credential.  `drift_rows` is reported either way so the in-process view and the
    worker's own startup check can never disagree silently.
    """
    live = dict(ident)
    live['watch'] = {}
    drift_rows = []
    for path, recorded in (ident.get('watch') or {}).items():
        actual = sha(path)
        live['watch'][path] = actual
        if actual != recorded:
            drift_rows.append({'path': path, 'recorded': recorded, 'actual': actual})
    if not live['watch']:
        drift_rows.append({'path': None, 'recorded': None, 'actual': None,
                           'reason': 'the frozen identity carries an EMPTY watch set, so nothing '
                                     'about the executed code is bound'})
    return live, drift_rows


def refuse_authorization(out_dir, persistence, *, stage, drift_rows, extra=None):
    """Refuse and leave a durable record naming exactly which watched paths drifted."""
    record = {'granted': False, 'stage': stage, 'n_drifted': len(drift_rows),
              'drifted': [{'path': r.get('path'), 'recorded': r.get('recorded'),
                           'actual': r.get('actual'), 'reason': r.get('reason')}
                          for r in drift_rows]}
    record.update(extra or {})
    try:
        persistence.write_once(Path(out_dir) / 'L1_AUTHORIZATION_REFUSED.json', record)
        persistence.write_once(Path(out_dir) / 'L1_IDENTITY_DRIFT.json', record)
    except Exception:                                                      # noqa: BLE001
        pass
    raise SystemExit('AUTHORIZATION REFUSED (%s): %d watched path(s) drifted since the freeze -- a '
                     'credential must never be consumed for code that no longer exists: %s'
                     % (stage, len(drift_rows),
                        [r.get('path') for r in drift_rows][:6]))


def worker(out_dir, *, synthetic_run=False):
    """The guarded worker.  Refuses before loading ANYTHING unless the execution context is valid.

    The order is deliberate and load-bearing: execution context -> frozen identity -> LIVE identity
    drift.  Only after all three pass is the F scorer bundle constructed.  The identity closure keeps
    no model import, no checkpoint read and no backend initialisation, so a drifted tree is refused in
    milliseconds instead of after half a minute of model loading -- and the refusal cannot be confused
    with a model-loading failure.
    """
    auth, persistence, protocol, real_arm, _ = load_helpers()
    import d3_l1_v120_execution_context as execctx
    sys.path.insert(0, str(EXT / 'scripts/tests'))
    from d3_l1_v112_phase_journal import PhaseJournal
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
    # identity closure before any scoring.  The SAME live re-derivation the supervisor authorises
    # over is used here, so the in-process view and the worker's own startup check cannot disagree.
    _, drift_rows = live_identity_view(ident)
    if drift_rows:
        persistence.write_once(out_dir / 'L1_IDENTITY_DRIFT.json',
                               {'n_drifted': len(drift_rows), 'drifted': drift_rows[:8],
                                'stage': 'worker_startup',
                                'basis': 'live re-derivation of every watched path',
                                'model_loaded': False,
                                'note': 'the drift is caught BEFORE the F scorer bundle is built: no '
                                        'checkpoint was read and no backend was initialised'})
        print('IDENTITY DRIFT: %d path(s)' % len(drift_rows), file=sys.stderr)
        return 1
    scorer_bundle = None
    if not synthetic_run:
        nv_max, m_max = real_arm.frozen_jit_shapes()
        scorer_bundle = real_arm.load_real_f_scorer(nv_max=nv_max, m_max=m_max)

    # BLOCKER B: the per-state PHASE JOURNAL.  `run_contained` derives the current stage from these
    # records, so without them the supervisor cannot tell "a state is running" from "the worker is
    # stuck in initialization" and cannot enforce the per-state wall clock at all.  The journal's
    # `(day_id, position)` domain is exactly the frozen 64-state list (days 0..15 x positions 1..4),
    # so the V1.16 `timeline_check` adjudicates these records without modification.
    phases = out_dir / 'phases'
    journal = PhaseJournal(phases)
    started = time.monotonic()
    publish = persistence.publish_new
    worker_record = {'schema': 'd3_l1_l1_worker', 'run_id': ident['run_id'], 'pid': os.getpid(),
                     'argv': sys.argv, 'execution_context_nonce': context.get('nonce'),
                     'synthetic_run': bool(synthetic_run)}
    publish(phases, 'identity_start', {'stage': 'identity_start', 'run_id': ident['run_id'],
                                       'paths': ident['watch']})

    states = ident['state_list']['states']
    records, errors = [], []
    first = states[0]
    warm_report = None
    try:
        # ---- initialization + compilation, then READY -------------------------------------
        journal.publish('initialize')
        warm_report = run_one_state(int(first['day_id']), int(first['position']),
                                    synthetic=synthetic_run, scorer_bundle=scorer_bundle,
                                    warmup_only=True)
        publish(out_dir, 'L1_WARMUP', dict(warm_report, run_id=ident['run_id']))
        journal.publish('compile')
        publish(out_dir, 'L1_READY',
                {'run_id': ident['run_id'], 'mono': time.monotonic(),
                 'initialize_and_compile_s': time.monotonic() - started,
                 'kind': 'initialization_and_compilation_finished',
                 'worker_pid': os.getpid(),
                 'warmup_scores_sha256': warm_report['scores_sha256'],
                 'n_warmup_calls': 1,
                 'warmup_state': [int(first['day_id']), int(first['position'])]})
    except BaseException as exc:                                           # noqa: BLE001
        import traceback
        publish(out_dir, 'L1_INIT_FAILURE',
                {'run_id': ident['run_id'], 'stage': 'initialization_and_compilation',
                 'exception_type': type(exc).__name__, 'error': str(exc),
                 'traceback': traceback.format_exc(),
                 'note': 'no READY record was published; the supervisor must treat this as an '
                         'initialization failure, not as a completed round'})
        # every pre-registered state still gets a record: the denominator may not shrink
        records = [blank_state_record(int(s['day_id']), int(s['position']),
                                      state_hash=s.get('state_hash'),
                                      status='NOT_ATTEMPTED_INIT_FAILURE',
                                      synthetic=synthetic_run,
                                      failure_phase='initialization_and_compilation')
                   for s in states]
        errors.append({'stage': 'initialization_and_compilation', 'error': str(exc)})
        _publish_round(out_dir, persistence, records, errors, synthetic=synthetic_run,
                       expected_states=len(states))
        return 1

    # ---- the pre-registered states, frozen order, EVERY state keeps a record ----------------
    for s in states:
        day, pos = int(s['day_id']), int(s['position'])
        journal.publish('slot', day_id=day, position=pos)
        t0 = time.monotonic()
        try:
            rec = run_one_state(day, pos, synthetic=synthetic_run, scorer_bundle=scorer_bundle,
                                wall_origin_mono=t0)
            status = 'COMPLETED'
        except BaseException as exc:                                       # noqa: BLE001
            import traceback
            rec = blank_state_record(
                day, pos, state_hash=s.get('state_hash'), status='FAILED_EXCEPTION',
                synthetic=synthetic_run, exception_type=type(exc).__name__, error=str(exc),
                traceback=traceback.format_exc(), failure_phase='state_comparison',
                wall_s=time.monotonic() - t0)
            rec['ledger_ok'] = False
            rec['ledger_problems'] = [{'reason': 'the state raised before its ledger could be '
                                                 'cross-checked', 'exception': type(exc).__name__}]
            status = 'FAILED_EXCEPTION'
            errors.append({'row': [day, pos], 'error': str(exc)})
        # TIMEOUT is a REPORTED outcome, not a deletion: the arms' gates already degraded fail-closed
        # to the original plan, so the record has a verdict and the state keeps its place.
        if any('timeout' in str(v) for v in (rec.get('per_arm_failure') or {}).values()):
            rec['status'] = 'TIMEOUT'
        records.append(rec)
        # the finish record is published for EVERY state, including a failed one, so the supervisor's
        # `timeline_check` sees a matched open/close pair rather than an orphan slot
        journal.publish('slot_finished', day_id=day, position=pos, status=rec.get('status'))
    return _publish_round(out_dir, persistence, records, errors, synthetic=synthetic_run,
                          expected_states=len(states), ident=ident)


def _publish_round(out_dir, persistence, records, errors, *, synthetic, expected_states, ident=None):
    """Publish the immutable per-state records, the round adjudication and the trailing identity.

    The records are written BEFORE the adjudication and the adjudication is computed from exactly
    those records, so a state that finished before a deadline or a kill is never lost: the durable
    JSONL is the input to the verdict, not a summary of it.
    """
    if ident is None:
        ident = load_frozen(out_dir)
    raw = '\n'.join(json.dumps(r, sort_keys=True, default=repr) for r in records).encode('utf-8')
    persistence.write_once(out_dir / 'L1_STATE_RECORDS.jsonl', raw)
    # BLOCKER C: NO FILTER.  The previous version adjudicated over
    # `[r for r in records if r.get('status') != 'FAILED_EXCEPTION']`, so a crashed state silently
    # left the denominator and the round still reported a verdict over the remainder.
    verdict = adjudicate(records, synthetic=synthetic, expected_states=expected_states)
    verdict['n_failed_exception'] = len(errors)
    verdict['failed_exceptions'] = errors[:8]
    persistence.write_once(out_dir / 'L1_ADJUDICATION.json', verdict)
    try:
        tail = {'stage': 'identity_end', 'run_id': ident['run_id'],
                'paths': {p: sha(p) for p in (ident.get('watch') or {})},
                'worker_ended_mono': time.monotonic()}
        sys.path.insert(0, str(EXT / 'scripts/tests'))
        from d3_l1_v112_phase_journal import PhaseJournal
        # The trailing identity goes to the journal directory under its OWN prefix.  It must NOT be
        # named `phase_*.json`: `d3_l1_v116_forward_control.timeline_check` opens EVERY file matching
        # that glob and requires each one to carry a `sequence` and a finite `mono`, so a record
        # published through a different mechanism under that prefix corrupts the timeline audit.
        persistence.publish_new(out_dir / 'phases', 'IDENTITY_END', tail)
        journal = PhaseJournal(out_dir / 'phases')
        journal.publish('completed', run_id=ident['run_id'], status=verdict['status'])
        publish_meta = {'n_journal_phases': len(sorted((out_dir / 'phases')
                                                       .glob('phase_*.json'))),
                        'journal_stages': sorted({json.loads(p.read_text(encoding='utf-8'))
                                                  .get('stage')
                                                  for p in (out_dir / 'phases')
                                                  .glob('phase_*.json')})}
        persistence.write_once(out_dir / 'L1_PHASE_JOURNAL_SUMMARY.json', publish_meta)
    except Exception as exc:                                               # noqa: BLE001
        persistence.write_once(out_dir / 'L1_TAIL_FAILURE.json', {'error': repr(exc)})
    print(json.dumps({k: v for k, v in verdict.items() if k != 'failed_exceptions'},
                     indent=1, default=repr))
    return 0 if verdict['status'] == 'ADJUDICATED' and not errors else 1


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
    # BLOCKER A: authorise over the LIVE identity, never over the frozen record alone.  The record
    # says what the code WAS; only a re-derivation from disk can say what the credential would
    # actually be buying.  This refuses BEFORE `authorize`, so no one-shot credential is spent.
    live, drift_rows = live_identity_view(ident)
    if drift_rows:
        refuse_authorization(out_dir, persistence, stage='pre_consumption_identity_drift',
                             drift_rows=drift_rows,
                             extra={'preregistration': str(preregistration),
                                    'credential_path': str(authorization),
                                    'consumption_attempted': False,
                                    'note': 'the drift is detected BEFORE consumption, so the '
                                            'one-shot credential is NOT spent'})
    view = authorization_identity_view(live)
    targets_view = authorization_targets_view(ident)
    ok, detail = auth.validate_credential(
        credential, identity=view, forward_targets=targets_view, output_dir=out_dir,
        preregistration_sha256=sha(preregistration))
    if not ok:
        persistence.write_once(out_dir / 'L1_AUTHORIZATION_REFUSED.json',
                               {'granted': False, 'problems': detail['problems'],
                                'live_identity_used': True, 'n_drifted': len(drift_rows)})
        raise SystemExit('AUTHORIZATION REFUSED: %s' % json.dumps(detail['problems']))
    granted = auth.authorize(
        credential, identity=view, forward_targets=targets_view, output_dir=out_dir,
        preregistration_path=Path(preregistration), run_dir=out_dir,
        consumption_root=consumption_root)
    # the credential is now spent, so the freeze is frozen for real: refuse if the entry point itself
    # changed between the check above and this point (a TOCTOU window, closed rather than assumed)
    post, post_drift = live_identity_view(ident)
    if post_drift:
        persistence.write_once(out_dir / 'L1_POST_CONSUMPTION_DRIFT.json',
                               {'n_drifted': len(post_drift), 'drifted': post_drift[:8],
                                'consumed': True,
                                'note': 'the drift appeared AFTER the credential was consumed; the '
                                        'run is stopped and the fact is recorded rather than hidden'})
        raise SystemExit('IDENTITY DRIFT AFTER CONSUMPTION: %s'
                         % [r['path'] for r in post_drift][:6])
    published = execctx.publish_execution_context(out_dir, authorization_result=granted)
    env = dict(os.environ, JAX_PLATFORMS='cpu', PYTHONIOENCODING='utf-8')
    env.update(published['env'])
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--out-dir', str(out_dir)]
    if synthetic_run:
        command.append('--synthetic-run')
    # the containment identity and the trailing tail publication are the L1 watch set
    identity = dict(ident['watch'])

    def tail_publisher(end_identity):
        """Publish the supervisor's trailing identity record to a FRESH destination name.

        The V1.22 convention: the fixed name `CONTROL_IDENTITY_END.json` is never replaced while a
        reader might hold it open.  This is the same tail the F path proved.
        """
        return persistence.publish_new(out_dir, 'CONTROL_IDENTITY_END', end_identity)

    result = control.run_contained(
        command, out_dir, identity, init_cap=float(BUDGET['initialize_compile_s']),
        slot_cap=float(BUDGET['per_state_wall_clock_s']),
        batch_cap=float(BUDGET['batch_wall_clock_s']), env=env,
        tail_publisher=tail_publisher,
        # BLOCKER B: without this the supervisor fell back to the V1.11 default probe and waited for a
        # `READY.json` the V1.18+ worker can never publish.
        ready_probe=make_l1_ready_probe(out_dir))
    persistence.write_once(out_dir / 'L1_SUPERVISION.json',
                           {k: v for k, v in result.items() if k != 'end_identity'})
    print(json.dumps({'reason': result.get('reason'), 'ok': result.get('ok'),
                      'returncode': result.get('returncode'),
                      'tree_active_after': result.get('tree_active_after'),
                      'ready_state': result.get('ready_state')}, indent=1, default=repr))
    return 0 if result.get('ok') else 1


def synthetic_run(out_dir, *, n_states=3):
    """BOTH arms in-process with a synthetic scorer, for acceptance.  Never the real comparison.

    The denominator is DECLARED here and passed explicitly, so the acceptance harness cannot inherit
    the production 64-state contract by accident: the three-state scope is visible in the record
    instead of being an unstated exception to it.
    """
    out_dir = Path(out_dir)          # main() hands over a string; every path join needs a Path
    ident = load_frozen(out_dir)
    _, persistence, _, _, _ = load_helpers()
    states = ident['state_list']['states'][:int(n_states)]
    records = []
    for s in states:
        day, pos = int(s['day_id']), int(s['position'])
        t0 = time.monotonic()
        try:
            records.append(run_one_state(day, pos, synthetic=True, wall_origin_mono=t0))
        except BaseException as exc:                                       # noqa: BLE001
            import traceback
            records.append(blank_state_record(day, pos, state_hash=s.get('state_hash'),
                                              status='FAILED_EXCEPTION', synthetic=True,
                                              exception_type=type(exc).__name__, error=str(exc),
                                              traceback=traceback.format_exc(),
                                              failure_phase='state_comparison',
                                              wall_s=time.monotonic() - t0))
    verdict = adjudicate(records, synthetic=True, expected_states=len(states))
    persistence.write_once(out_dir / 'L1_SYNTHETIC_ACCEPTANCE.json',
                           {'verdict': verdict, 'records': records,
                            'expected_states': len(states),
                            'scope': 'SYNTHETIC ACCEPTANCE ONLY: %d state(s), synthetic scorer, '
                                     'no scientific reading permitted' % len(states)})
    print(json.dumps(verdict, indent=1, default=repr))
    return 0 if verdict['status'] == 'ADJUDICATED' else 1


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
