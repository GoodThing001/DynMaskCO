#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V1.21 FORMAL EXECUTION SIGNING PACKAGE: the single artifact a signature is given against.

This tool assembles, from the run directory and the repository, everything a signer must be able to
check WITHOUT trusting a status document:

  1  the final git commit SHA and whether the worktree is clean
  2  the run id, and the pre-registration it is bound to, with the pre-registration's SHA
  3  the frozen identities: F checkpoint, backbone, data, profile, base + effective contract,
     dependency-module digest, state binding and forward-targets digest
  4  the 64 states in LEDGER PUBLICATION ORDER, with each state's hash and the order digest
  5  the call schedule (1 warm-up + 64 x 2 = 129 maximum forwards)
  6  the frozen budgets (initialize/compile 60 s, slot 10 s, batch 180 s)
  7  the mechanical evidence that every engineering gate passed: the readiness verdict re-read from
     the record, the acceptance/verifier verdicts, the watch closure cross-check and the per-module
     tamper counterexamples
  8  the independent proof that the real F forward count is still ZERO
  9  the authorization template, its content digest, and the explicit statement that its human
     approval fields are EMPTY -- a signature is given by answering in chat, and the credential is
     built only AFTER that answer

It also re-verifies the template against the frozen identity on the spot, so a template that had
drifted from the freeze would be reported rather than offered for signature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
for _p in ('scripts', 'scripts/simulation', 'scripts/tests', 'scripts/data'):
    sys.path.insert(0, str(EXT / _p))

import d3_l1_v118_authorization as AUTH                                       # noqa: E402
import d3_l1_v118_persistence as persistence                                 # noqa: E402

RUNNER = EXT / 'scripts/simulation/d3_l1_v118_forward_runner.py'
EV = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
RUN_ID = os.environ.get('V_SIGN_RUN_ID', 'd3l1-real-f-forward-20261015-v122')
RUN_DIR = EV / os.environ.get('V_SIGN_RUN_DIR', 'v122_evidence/run_v122')
PREREG = EV / os.environ.get(
    'V_SIGN_PREREG', 'A1_D3_L1_REAL_F_FORWARD_EXECUTION_PREREGISTRATION_V1_22.md')
TEMPLATE = EV / os.environ.get('V_SIGN_TEMPLATE', 'v122_evidence/AUTHORIZATION_TEMPLATE_V122.json')
STATE_LEDGER = EV / 'v19_evidence/formal_collection/STATE_LEDGER.jsonl'
V121_ACC = EV / 'v122_evidence/D3_V121_FREEZE_CLOSURE_ACCEPTANCE.json'
V121_ZERO = EV / 'v122_evidence/ZERO_REAL_F_WORK_AUDIT.json'
V120_BYPASS = EV / 'v120_evidence/D3_V120_BYPASS_ACCEPTANCE.json'
V119_BLOCKERS = EV / 'v119_evidence/D3_V119_BLOCKER_ACCEPTANCE.json'
V118_E2E = EV / 'v122_evidence/SYNTHETIC_END_TO_END.json'
V121_VERIFIER = EV / 'v122_evidence/INDEPENDENT_READONLY_VERIFIER.json'


def sha(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    proc = subprocess.run(['git'] + list(args), cwd=str(REPO), capture_output=True, text=True)
    return proc.stdout.strip()


def load(path):
    p = Path(path)
    return json.loads(p.read_text(encoding='utf-8')) if p.is_file() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    args = ap.parse_args()

    ident_rec = persistence.VerifiedReader(RUN_DIR).latest('RUN_IDENTITY') or {}
    fwd = persistence.VerifiedReader(RUN_DIR).latest('FORWARD_TARGETS') or {}
    idt = persistence.VerifiedReader(RUN_DIR).latest('IDENTITY_TARGETS') or {}
    readiness_rec = persistence.VerifiedReader(RUN_DIR).latest('READINESS') or {}
    readiness = readiness_rec.get('readiness') or {}
    checks = readiness.get('checks') or {}

    # ---- 1 git -------------------------------------------------------------------------
    status = git('status', '--short')
    dirty = [ln for ln in status.splitlines()
             if ln.strip() and 'CC_Compare/MAPT_code' not in ln]
    git_block = {'head': git('rev-parse', 'HEAD'), 'origin_main': git('rev-parse', 'origin/main'),
                 'branch_sync': git('rev-parse', 'HEAD') == git('rev-parse', 'origin/main'),
                 'worktree_clean_except_known_gitlink': not dirty,
                 'dirty_entries': dirty[:8]}

    # ---- 2 run id + pre-registration ----------------------------------------------------
    prereg_block = {'path': str(PREREG.relative_to(REPO)), 'sha256': sha(PREREG),
                    'supersedes': 'A1_D3_L1_REAL_F_FORWARD_EXECUTION_PREREGISTRATION_V1_20.md',
                    'bindings': {}}

    # ---- 3 frozen identities ------------------------------------------------------------
    ids = ident_rec.get('identities') or {}
    identities = {k: {'path': v.get('path'), 'sha256': v.get('sha256'),
                      'registered': v.get('registered'), 'matches': v.get('matches'),
                      'label': v.get('label')}
                  for k, v in ids.items()}
    identities['contracts'] = ident_rec.get('contracts')
    identities['dependency_identity_sha256'] = ident_rec.get('dependency_identity_sha256')
    identities['state_binding_sha256'] = ident_rec.get('state_binding_sha256')
    identities['forward_targets_sha256'] = fwd.get('targets_sha256')

    # ---- 4 the 64 states in ledger publication order -------------------------------------
    states = []
    for line in STATE_LEDGER.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get('event') == 'PUBLISHED':
            states.append({'index': len(states), 'day_id': r['day_id'], 'position': r['position'],
                           'state_hash': r['state_hash']})
    order = [(s['day_id'], s['position']) for s in states]
    state_block = {'n_states': len(states), 'order_source': 'STATE_LEDGER.jsonl PUBLISHED order',
                   'order_is_canonical': order == sorted(order),
                   'order_keys_unique': len(set(order)) == len(order),
                   'all_have_state_hash': all(s['state_hash'] for s in states),
                   'states': states,
                   'order_digest_sha256': hashlib.sha256(json.dumps(
                       [[s['day_id'], s['position'], s['state_hash']] for s in states],
                       sort_keys=True).encode()).hexdigest(),
                   'ledger_sha256': sha(STATE_LEDGER)}

    # ---- 5 / 6 call schedule and budgets -------------------------------------------------
    schedule = ident_rec.get('call_schedule')
    budgets = ident_rec.get('budgets')

    # ---- 7 mechanical gate evidence ------------------------------------------------------
    acc = load(V121_ACC) or {}
    zero = load(V121_ZERO) or {}
    bypass = load(V120_BYPASS) or {}
    blockers = load(V119_BLOCKERS) or {}
    e2e = load(V118_E2E) or {}
    verifier = load(V121_VERIFIER) or {}
    gates = {
        'V1.21_freeze_closure': {'file': str(V121_ACC.relative_to(REPO)),
                                 'sha256': sha(V121_ACC), 'verdict': acc.get('verdict'),
                                 'n_pass': acc.get('n_pass'), 'n_fail': acc.get('n_fail'),
                                 'n_watch_paths': acc.get('n_watch_paths')},
        'V1.20_execution_entry_bypass': {'file': str(V120_BYPASS.relative_to(REPO)),
                                         'sha256': sha(V120_BYPASS),
                                         'verdict': bypass.get('verdict'),
                                         'n_pass': bypass.get('n_pass'),
                                         'n_fail': bypass.get('n_fail')},
        'V1.19_three_blockers': {'file': str(V119_BLOCKERS.relative_to(REPO)),
                                 'sha256': sha(V119_BLOCKERS),
                                 'verdict': blockers.get('verdict'),
                                 'n_pass': blockers.get('n_pass'), 'n_fail': blockers.get('n_fail')},
        'V1.18_synthetic_end_to_end': {'file': str(V118_E2E.relative_to(REPO)),
                                       'sha256': sha(V118_E2E), 'verdict': e2e.get('verdict'),
                                       'n_pass': e2e.get('n_pass'), 'n_fail': e2e.get('n_fail')},
        'independent_read_only_verifier': {'file': str(V121_VERIFIER.relative_to(REPO)),
                                           'sha256': sha(V121_VERIFIER),
                                           'verdict': verifier.get('verdict'),
                                           'n_pass': verifier.get('n_pass'),
                                           'n_fail': verifier.get('n_fail')},
        'readiness_record': {'run_dir': str(RUN_DIR.relative_to(REPO)),
                             'verdict': readiness_rec.get('verdict'),
                             'missing_before_signature': readiness.get('missing_before_signature'),
                             'checks': checks},
    }

    # ---- 8 zero real-F work ---------------------------------------------------------------
    zero_block = {'file': str(V121_ZERO.relative_to(REPO)), 'sha256': sha(V121_ZERO),
                  'verdict': zero.get('verdict'), 'n_pass': zero.get('n_pass'),
                  'n_fail': zero.get('n_fail'),
                  'absent_counts': (zero.get('records') or {}).get('absent_counts'),
                  'consumption_for_this_run': (zero.get('records') or {}).get(
                      'consumption_for_this_run')}

    # ---- 9 the authorization template, re-verified against the freeze ---------------------
    tmpl = load(TEMPLATE) or {}
    ok, detail = None, None
    if tmpl:
        ok, detail = AUTH.validate_credential(tmpl, identity=ident_rec, forward_targets=fwd,
                                             output_dir=RUN_DIR,
                                             preregistration_sha256=sha(PREREG),
                                             require_human=False)
    human_empty = [f for f in AUTH.HUMAN_FIELDS if not str(tmpl.get(f) or '').strip()]
    template_block = {
        'path': str(TEMPLATE.relative_to(REPO)), 'sha256': sha(TEMPLATE),
        'credential_sha256': tmpl.get('credential_sha256'),
        'run_id': tmpl.get('run_id'),
        'preregistration_sha256': tmpl.get('preregistration_sha256'),
        'bindings_match_frozen_identity': bool(ok),
        'binding_detail': detail,
        'human_approval_fields': {f: ('EMPTY' if not str(tmpl.get(f) or '').strip()
                                      else 'FILLED') for f in AUTH.HUMAN_FIELDS},
        'human_approval_fields_empty': human_empty,
        'is_not_an_authorization': True,
        'note': 'the credential is created ONLY after the user answers with an explicit approval of '
                'this run_id and this pre-registration SHA; filling a field in this file is not an '
                'approval',
    }

    package = {
        'schema': 'd3_l1_v121_formal_execution_signing_package',
        'generated_utc_git_head': git_block['head'],
        '1_git': git_block,
        '2_run': {'run_id': RUN_ID, 'run_dir': str(RUN_DIR.relative_to(REPO)),
                  'pre_registration': prereg_block},
        '3_frozen_identities': identities,
        '4_states': state_block,
        '5_call_schedule': schedule,
        '6_frozen_budgets': budgets,
        '7_engineering_gates': gates,
        '8_zero_real_f_proof': zero_block,
        '9_authorization_template': template_block,
        'what_a_signature_authorises': [
            'ONE real-F scoring-interface pass over the 64 frozen states, at most 129 forwards',
            'nothing else'],
        'what_a_signature_does_NOT_authorise': [
            'the F-vs-D-Geo L1 comparison (a separate, independent authorisation)',
            'training / E2', 'L2 / L3', 'the 40-day batch', 'the final test',
            'installing or writing any plan into a live state',
            'any utility, significance or algorithmic-gain claim'],
        'consumption_rule': 'PASS, FAIL, timeout or a mid-batch kill all consume the single '
                            'opportunity: no re-run, no filling missing states, and never splicing '
                            "V1.11's 42 missing records",
        'blocking_for_this_stage': 'none: the C0 formal budget is NOT required for a scoring-'
                                   'interface pass, because no plan is installed',
        'tool_sha256': sha(Path(__file__)),
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(package, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    print(json.dumps({
        'git_head': git_block['head'],
        'run_id': RUN_ID,
        'readiness': readiness_rec.get('verdict'),
        'missing_before_signature': readiness.get('missing_before_signature'),
        'runtime_watch_closed': checks.get('runtime_watch_closed'),
        'runtime_watch_covers_closure': checks.get('runtime_watch_covers_closure'),
        'real_F_forward_executed': checks.get('real_F_forward_executed'),
        'n_states': state_block['n_states'],
        'order_digest_sha256': state_block['order_digest_sha256'],
        'preregistration_sha256': prereg_block['sha256'],
        'template_sha256': template_block['sha256'],
        'template_bindings_match': template_block['bindings_match_frozen_identity'],
        'template_human_fields_empty': template_block['human_approval_fields_empty'],
        'max_forward_calls': (schedule or {}).get('max_forward_calls'),
        'budgets': (budgets or {}).get('caps'),
        'zero_real_f_verdict': zero_block['verdict'],
        'gates': {k: (v.get('verdict'), v.get('n_fail'))
                  for k, v in gates.items() if isinstance(v, dict)},
    }, indent=1, default=repr))
    print('WROTE', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
