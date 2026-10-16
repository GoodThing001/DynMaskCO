#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L1-v2 EXECUTION CREDENTIAL: build, validate, and refuse to proceed on any mismatch.

A HUMAN approval is the SOURCE of this credential, not something this tool invents.  The approval is
the signed L1-v2 authorisation message from the project owner (`GoodThing001`), which names the git
HEAD, the run id, the execution pre-registration, its SHA-256, the source identity digest, the signing
package and the frozen identity size, and states the authorised scope.  `approved_by` and
`approval_statement` transcribe that approval; they are not generated here.

This tool does three things:
  1. `--build`   seal the credential for the frozen run directory
  2. `--validate` compare it field by field against the LIVE identity and the pre-registration file,
                 and print every binding; a single mismatch is reported, not smoothed over
  3. `--show`    print the credential's bindings and its own content digest

It does NOT consume.  Consumption happens exactly once, inside `--supervise`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
for _p in ('scripts', 'scripts/simulation', 'scripts/tests', 'scripts/data'):
    sys.path.insert(0, str(EXT / _p))

import d3_l1_v118_authorization as auth                                        # noqa: E402

EV = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
PREREG = EV / 'A1_D3_L1_L1_FORMAL_EXECUTION_PREREGISTRATION_V2.md'
RUN_DIR = EV / 'l1_v2_execution_run'
CRED = EV / 'l1_gates/L1_V2_EXECUTION_AUTHORIZATION.json'
RUN_ID = 'd3l1-l1-two-arm-20261016-l1v2'
AUTHORISED_HEAD = '558fb5c68b4759dbbf9819f6802707d1d5e2bc8b'
AUTHORISED_PREREG_SHA = 'e02b98587682a42f9142104e6798b6e8ef4db4135e9ee0dc0eaf2aa749e449ba'
AUTHORISED_SOURCE_IDENTITY = ('6139249946497ac1dde4848cda7531df85446667075fb50c012607c275d05d0c')
AUTHORISED_PACKAGE_SHA = '1f74d7dcb04334847bda6ed00a96236b6aa14180587c3b01c104b76ccfb17bc1'

APPROVED_BY = 'GoodThing001 <1948744323@qq.com> (project owner; explicit written authorisation)'
APPROVED_UTC = '2026-10-16T03:23:38Z'
APPROVAL_STATEMENT = (
    'I have reviewed the L1-v2 final signing package and I expressly approve this single formal '
    'F-D-Geo mechanism comparison, bound to: git HEAD 558fb5c68b4759dbbf9819f6802707d1d5e2bc8b; '
    'run_id d3l1-l1-two-arm-20261016-l1v2; execution pre-registration '
    'A1_D3_L1_L1_FORMAL_EXECUTION_PREREGISTRATION_V2.md with SHA-256 '
    'e02b98587682a42f9142104e6798b6e8ef4db4135e9ee0dc0eaf2aa749e449ba; source_identity_sha256 '
    '6139249946497ac1dde4848cda7531df85446667075fb50c012607c275d05d0c; signing package '
    'L1_FORMAL_EXECUTION_SIGNING_PACKAGE_V3.json; frozen identity 38 source files + 4 assets, 42 '
    'paths in total.  Authorised scope: the 64 frozen development states in their original '
    'publication order; F using the V1.22-verified real MaskCO scoring model and D-Geo the registered '
    '-incremental_distance; both arms sharing the initial state, the B pool, the ordered legal list, '
    'the strict-online visible information, the contract and the resource rules; scorer_id as the sole '
    'declared varying factor; the frozen budget (initialize 60 s, per-state 10 s shared by both arms, '
    'batch 180 s, at most 2 scorer calls per arm per state and 128 scorer calls per arm per batch); a '
    'frozen execution order F then D-Geo; V1, V2 and V3 adjudicated separately with V3 fixed at '
    'NOT_RUN, no plan installed and no formal C0 certification performed.  Execution only through the '
    'lawful --supervise --authorization --preregistration entry, with the credential matching this '
    'identity exactly and consumed exactly once.  PASS, FAIL, timeout and force-kill all consume this '
    'opportunity; no re-run, no top-up, no state filtering, no budget relaxation and no splicing of '
    'historical records.  All raw records, timeouts, exceptions, receipts and identity audits are '
    'retained, followed immediately by independent read-only review and formal adjudication.  NOT '
    'authorised: E2, L2, L3, the 40-day experiment, the final test, plan installation, or any other '
    'formal experiment.')


def load_frozen_entry():
    spec = importlib.util.spec_from_file_location(
        'l1_entry_for_cred', str(EXT / 'scripts/simulation/d3_l1_l1_formal_run.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules['l1_entry_for_cred'] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    p = Path(path)
    if not p.is_file():
        return None
    import hashlib
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build():
    entry = load_frozen_entry()
    ident = entry.load_frozen(RUN_DIR)
    if ident['run_id'] != RUN_ID:
        raise SystemExit('REFUSING: the frozen run directory carries run_id %r, not %r'
                         % (ident['run_id'], RUN_ID))
    view = entry.authorization_identity_view(ident)
    targets = entry.authorization_targets_view(ident)
    cred = auth.build_template(view, targets, preregistration_sha256=sha(PREREG),
                               output_dir=RUN_DIR)
    cred['approved_by'] = APPROVED_BY
    cred['approved_utc'] = APPROVED_UTC
    cred['approval_statement'] = APPROVAL_STATEMENT
    sealed = auth.seal_template(cred)
    CRED.parent.mkdir(parents=True, exist_ok=True)
    if CRED.exists():
        raise SystemExit('REFUSING to overwrite an existing credential: %s' % CRED)
    CRED.write_text(json.dumps(sealed, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    return sealed


def validate():
    entry = load_frozen_entry()
    ident = entry.load_frozen(RUN_DIR)
    live, drift = entry.live_identity_view(ident)
    view = entry.authorization_identity_view(live)
    targets = entry.authorization_targets_view(ident)
    cred = auth.load_credential(CRED)
    ok, detail = auth.validate_credential(cred, identity=view, forward_targets=targets,
                                          output_dir=RUN_DIR,
                                          preregistration_sha256=sha(PREREG))
    rows = detail.get('bindings') or {}
    mismatches = {k: v for k, v in rows.items() if not v['ok']}
    # `live` is the raw identity; the credential's view of it is what carries `budgets`/`call_schedule`
    return ok, detail, mismatches, drift, {'identity': live, 'view': view}, entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args()
    if args.build:
        sealed = build()
        print(json.dumps({'credential': str(CRED.relative_to(REPO)),
                          'run_id': sealed['run_id'],
                          'credential_sha256': sealed['credential_sha256'],
                          'source_identity_sha256': sealed['source_identity_sha256'],
                          'approved_by': sealed['approved_by'],
                          'approved_utc': sealed['approved_utc'],
                          'human_fields_filled': all(sealed.get(f) for f in auth.HUMAN_FIELDS)},
                         indent=1, ensure_ascii=False))
    if args.show:
        cred = auth.load_credential(CRED)
        print(json.dumps({k: cred.get(k) for k in auth.BINDING_FIELDS}, indent=1, ensure_ascii=False))
    if args.validate:
        ok, detail, mismatches, drift, live, entry = validate()
        ident, view = live['identity'], live['view']
        print(json.dumps({
            'run_id': ident['run_id'],
            'validated': bool(ok),
            'n_bindings': len(detail.get('bindings') or {}),
            'n_mismatched': len(mismatches),
            'mismatched': mismatches,
            'problems': detail.get('problems'),
            'live_drift': len(drift),
            'content_digest_ok': detail['content_digest']['ok'],
            'source_identity_sha256': (detail.get('bindings') or {})
            .get('source_identity_sha256', {}).get('declared'),
            'preregistration_version': (detail.get('bindings') or {})
            .get('preregistration_version', {}).get('declared'),
            'preregistration_sha256': (detail.get('bindings') or {})
            .get('preregistration_sha256', {}).get('declared'),
            # the authorised values, checked against the credential's own declarations
            'matches_the_authorisation': {
                'source_identity_sha256': (detail.get('bindings') or {})
                .get('source_identity_sha256', {}).get('declared') == AUTHORISED_SOURCE_IDENTITY,
                'preregistration_sha256': (detail.get('bindings') or {})
                .get('preregistration_sha256', {}).get('declared') == AUTHORISED_PREREG_SHA,
                'run_id': ident['run_id'] == RUN_ID,
                'n_watch': len(ident['watch']) == 42,
                'n_states': view['state_list']['n_states'] == 64,
                'budget': view['budgets']['caps'] == {'initialize_compile_s': 60.0,
                                                      'slot_span_s': 10.0, 'batch_s': 180.0}},
        }, indent=1, ensure_ascii=False, default=repr))
        return 0 if ok else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
