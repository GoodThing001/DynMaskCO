#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L1 NEXT-STEP PACKAGE: the F-forward verdict, the L1 readiness check, and the signable L1 package.

Built only AFTER the single authorised real-F forward completed.  It does three things, each from
artifacts rather than from prose:

  1  restates the F scoring-interface qualification verdict (from the adjudication record) and what
     it does NOT establish
  2  checks the L1 preconditions MECHANICALLY, one by one, against the frozen assets -- so an unmet
     precondition is reported as unmet rather than assumed
  3  produces the L1 execution package that would be signed: the shared-input rules, the two arms,
     the three verdicts, the resource/identity freeze and the exact open questions

It never runs the F-vs-D-Geo comparison: that is a separate authorisation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
for _p in ('scripts', 'scripts/simulation', 'scripts/tests', 'scripts/data'):
    sys.path.insert(0, str(EXT / _p))

import d3_l1_v118_persistence as persistence                                 # noqa: E402
import d3_l1_v121_dependency_closure as closure                              # noqa: E402

EV = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
V122 = EV / 'v122_evidence'
L1 = EV / 'l1_gates'
RUN_DIR = V122 / 'run_v122'


def sha(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding='utf-8')) if p.is_file() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    args = ap.parse_args()

    adj = load(V122 / 'REAL_F_FORWARD_ADJUDICATION_V122.json') or {}
    verifier = load(V122 / 'INDEPENDENT_READONLY_VERIFIER_EXECUTED.json') or {}
    l1_acc = load(L1 / 'A1_D3_L1_TWO_ARM_PROTOCOL_ACCEPTANCE.json') or {}
    l1_audit = load(L1 / 'A1_D3_L1_TWO_ARM_INDEPENDENT_AUDIT.json') or {}
    ident = persistence.VerifiedReader(RUN_DIR).latest('RUN_IDENTITY') or {}
    fwd = persistence.VerifiedReader(RUN_DIR).latest('FORWARD_TARGETS') or {}
    readiness = persistence.VerifiedReader(RUN_DIR).latest('READINESS') or {}

    # ---- 1 the F verdict ------------------------------------------------------------------
    f_verdict = {
        'adjudication_file': str((V122 / 'REAL_F_FORWARD_ADJUDICATION_V122.json')
                                 .relative_to(REPO)),
        'adjudication_sha256': sha(V122 / 'REAL_F_FORWARD_ADJUDICATION_V122.json'),
        'verdict': adj.get('verdict'), 'n_pass': adj.get('n_pass'), 'n_fail': adj.get('n_fail'),
        'independent_verifier': {'file': str((V122 / 'INDEPENDENT_READONLY_VERIFIER_EXECUTED.json')
                                             .relative_to(REPO)),
                                 'sha256': sha(V122 / 'INDEPENDENT_READONLY_VERIFIER_EXECUTED.json'),
                                 'verdict': verifier.get('verdict'),
                                 'row_state': (verifier.get('records') or {}).get('row_state')},
        'execution': adj.get('execution_summary'),
        'what_it_does_NOT_establish': (adj.get('claim_limits') or {})
        .get('what_this_does_NOT_establish'),
        'run_id': ident.get('run_id'), 'labels': ident.get('labels'),
    }

    # ---- 2 L1 preconditions, checked mechanically ------------------------------------------
    watch = {p: {'frozen': v, 'current': sha(p), 'ok': sha(p) == v}
             for p, v in ((persistence.VerifiedReader(RUN_DIR).latest('IDENTITY_TARGETS') or {})
                          .get('paths') or {}).items()}
    closure_info = closure.build_watch_set()
    cc_ok, cc = closure.cross_check_watch_set(
        list((persistence.VerifiedReader(RUN_DIR).latest('IDENTITY_TARGETS') or {}).get('paths')
             or {}))
    pre = {
        'F_forward_passed': {'ok': adj.get('verdict') ==
                             'REAL_F_FORWARD_ENGINEERING_PASS_ADJUDICATED'
                             and adj.get('n_fail') == 0
                             and verifier.get('verdict') == 'VERIFIED',
                             'detail': {'adjudication': adj.get('verdict'),
                                        'verifier': verifier.get('verdict'),
                                        'rows_independently_ok':
                                            ((verifier.get('records') or {}).get('row_state') or {})
                                            .get('n_independently_ok')}},
        'runtime_watch_intact': {'ok': bool(watch) and all(v['ok'] for v in watch.values()),
                                 'detail': {'n_paths': len(watch),
                                            'n_drifted': sum(1 for v in watch.values()
                                                             if not v['ok']),
                                            'covers_closure': bool(cc_ok)}},
        'states_frozen_and_scored': {'ok': fwd.get('n_rows') == 64,
                                    'detail': {'n_targets': fwd.get('n_rows'),
                                               'targets_sha256': fwd.get('targets_sha256')}},
        'contract_and_profile_identity': {
            'ok': bool((ident.get('contracts') or {}).get('matches'))
            and bool((ident.get('identities') or {}).get('profile_file', {}).get('matches')),
            'detail': {'effective_contract': (ident.get('contracts') or {})
                       .get('effective_contract_hash'),
                       'profile_hash': (ident.get('identities') or {})
                       .get('profile_file', {}).get('profile_hash')}},
        'two_arm_protocol_acceptance': {'ok': l1_acc.get('verdict')
                                        == 'L1_TWO_ARM_PROTOCOL_ENGINEERING_ACCEPTED'
                                        and l1_acc.get('n_fail') == 0,
                                        'detail': {'verdict': l1_acc.get('verdict'),
                                                   'n_pass': l1_acc.get('n_pass'),
                                                   'n_fail': l1_acc.get('n_fail'),
                                                   'protocol_sha256':
                                                       l1_acc.get('protocol_sha256')}},
        'independent_l1_audit': {'ok': l1_audit.get('verdict')
                                 == 'L1_TWO_ARM_PROTOCOL_INDEPENDENTLY_AUDITED'
                                 and l1_audit.get('n_fail') == 0,
                                 'detail': {'verdict': l1_audit.get('verdict'),
                                            'n_pass': l1_audit.get('n_pass'),
                                            'n_fail': l1_audit.get('n_fail')}},
        'backbone_identity_ruling': {
            'ok': True,
            'detail': {'label': (ident.get('labels') or {}).get('backbone'),
                       'note': 'the label is UNVERIFIED by ruling and is carried into every L1 '
                               'conclusion; no ruling change is pending'}},
        'c0_budget_for_certification': {
            'ok': None, 'detail': {
                'status': 'AWAITING_USER_BUDGET_RULING',
                'blocking_l1': False,
                'why_not_blocking': 'L1 reports three SEPARATE verdicts and its V3 '
                                    '(certified executable plan) is injectable: it can be reported '
                                    'as NOT_RUN with a named certifier identity, so no formal C0 '
                                    'budget is required to answer the L1 question. A formal budget '
                                    'only becomes necessary if the signer wants V3 to mean the real '
                                    'C0 certification.',
                'if_signer_wants_numeric_V3': 'a C0 budget ruling is required first'}},
    }
    all_ok = all(v['ok'] is True for v in pre.values())
    blocked = [k for k, v in pre.items() if v['ok'] is not True]

    # ---- 3 the signable L1 package ----------------------------------------------------------
    package = {
        'schema': 'd3_l1_l1_execution_signing_package',
        '1_f_forward_verdict': f_verdict,
        '2_l1_preconditions': {'all_satisfied': all_ok, 'not_satisfied': blocked,
                               'checks': pre},
        '3_shared_input_rules': {
            'state': 'the SAME 64 frozen states (V1.9 ledger publication order), or a separate '
                     'pre-registered fresh set for L2',
            'p0_incumbent': 'built by the same V1.10 materialisation path from the shared snapshot',
            'mask': 'B-semantics pool built BEFORE sampling (build_pool_B), identical for both arms',
            'ordered_legal': 'the environment enumerator order, verbatim, shared by both arms; '
                             'first-step legal fingerprint must match on both arms',
            'eps': 'eps_rel = 0.0, eps_abs = 0.0 (no tolerance band: one factor only)',
            'tie_break': 'legal_order_first (F\'s historical first-maximum argmax)',
            'contract': 'the same effective contract hash for both arms',
            'information_boundary': 'strict-online; the same visible set for both arms',
            'declared_varying_fields': ['scorer_id'],
        },
        '4_arms': {
            'F': 'restored MaskCO logits, run_id d3l1-real-f-forward-20261015-v122, label '
                 'F_RECONSTRUCTED_WITH_CURRENT_BACKBONE / HISTORICAL_BACKBONE_UNVERIFIED',
            'D_Geo': 'registered geo_score = -incremental_distance (no model, no training)',
        },
        '5_verdicts': {
            'V1_first_step_divergence': 'DIVERGENT / IDENTICAL / NOT_COMPARABLE / '
                                        'NOT_APPLICABLE_EMPTY_MASK',
            'V2_full_repair_completed': 'per arm, excluding certification',
            'V3_certified_executable_plan': 'per arm; NOT_RUN when no certifier is supplied -- an '
                                            'absent check is neither True nor False',
        },
        '6_resource_and_identity_freeze': {
            'status': 'TO_BE_FROZEN_AT_SIGNING',
            'basis': 'the L1 protocol records per-arm wall clock and the six budget counters '
                     'separately; the caps are set from MEASURED cost, never widened afterwards',
            'measured_input': {'F_per_slot_span_max_s':
                               (adj.get('execution_summary') or {}).get('max_span_s'),
                               'F_worker_wall_s': (adj.get('execution_summary') or {})
                               .get('worker_wall_s'),
                               'note': 'the real F cost is now measured on the 64 frozen states, so '
                                       'the L1 caps can be derived from real numbers instead of the '
                                       'V1.9 collection timings'},
            'identity': {'F_run_id': ident.get('run_id'),
                         'watch_paths': len(watch),
                         'watch_covers_closure': bool(cc_ok),
                         'protocol_sha256': l1_acc.get('protocol_sha256'),
                         'closure_module_sha256': sha(EXT / 'scripts/data/'
                                                      'd3_l1_v121_dependency_closure.py')},
        },
        '7_open_questions_for_the_signer': [
            {'id': 'L1-Q1', 'question': 'Should L1 V3 be reported as NOT_RUN (synthetic certifier, '
                                        'name recorded), or should the signer first rule the C0 '
                                        'budget so V3 means the real C0 certification?',
             'recommended': 'NOT_RUN for the L1 mechanism question; a numeric V3 belongs to L2',
             'blocking': False},
            {'id': 'L1-Q2', 'question': 'Which states? The same 64 real dev_check states (fast, but '
                                        'previously used for development) or a fresh pre-registered '
                                        'set?',
             'recommended': 'the same 64 for L1 (a mechanism question), with L2 on fresh states',
             'blocking': False},
            {'id': 'L1-Q3', 'question': 'Per-arm wall-clock cap: derived from the measured F slot '
                                        'spans with a stated multiplier?',
             'recommended': 'freeze at signing from the measured distribution, never widened later',
             'blocking': False},
        ],
        '8_what_this_package_does_NOT_authorise': [
            'starting the F-vs-D-Geo comparison (a separate, independent authorisation)',
            'training / E2', 'L2', 'L3', 'the 40-day batch', 'the final test',
            'any utility, significance or algorithmic-gain claim'],
        'not_run': ['the L1 F-vs-D-Geo comparison', 'training', 'L2', 'L3', 'the 40-day batch',
                    'the final test'],
        'stable_online_gain': False, 'science_ready': False,
        'tool_sha256': sha(Path(__file__)),
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(package, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    print(json.dumps({'F_verdict': f_verdict['verdict'],
                      'F_adjudication_pass': [adj.get('n_pass'), adj.get('n_fail')],
                      'L1_preconditions_all_satisfied': all_ok,
                      'L1_not_satisfied': blocked,
                      'checks': {k: v['ok'] for k, v in pre.items()}},
                     indent=1, default=repr))
    print('WROTE', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
