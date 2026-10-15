# -*- coding: utf-8 -*-
"""Create the V1.121 authorization credential from the USER'S EXPLICIT APPROVAL.

The approval is not invented here: the user answered, in this session, an explicit approval request
that named run_id `d3l1-real-f-forward-20261015-v121` and pre-registration SHA
`b9e1d2588591b437d02f1134b3f13e1ffede7cc5de3f6e56837dbfb891df9871`, and selected
"approve this run_id and pre-registration SHA, execute this one real-F forward".

This script only TRANSCRIBES that decision into the credential format, with the user's own words as
the approval statement, and then validates the credential against the frozen identity before
sealing it. If validation fails, nothing is written.
"""
import importlib.util
import json
import sys
from pathlib import Path

EXT = Path(r'D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem')
REPO = EXT.parent
EV = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
RUN_DIR = EV / 'v121_evidence/run_v121'
PREREG = EV / 'A1_D3_L1_REAL_F_FORWARD_EXECUTION_PREREGISTRATION_V1_21.md'
OUT = EV / 'v121_evidence/AUTHORIZATION_V121_APPROVED.json'
for p in ('scripts', 'scripts/simulation', 'scripts/tests'):
    sys.path.insert(0, str(EXT / p))

import d3_l1_v118_authorization as AUTH


def load_runner():
    spec = importlib.util.spec_from_file_location(
        'auth_runner', str(EXT / 'scripts/simulation/d3_l1_v118_forward_runner.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules['auth_runner'] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()
frozen = runner.load_frozen(RUN_DIR)
ident, fwd = frozen['identity'], frozen['forward']
template = AUTH.build_template(ident, fwd, preregistration_sha256=runner.sha256_file(PREREG),
                               output_dir=RUN_DIR)

APPROVED_BY = 'USER (explicit approval in the DynMaskCO-CC session, 2026-10-15)'
APPROVED_UTC = '2026-10-15T00:00:00Z'
STATEMENT = (
    "The user was asked: 'approve run_id d3l1-real-f-forward-20261015-v121 and pre-registration SHA "
    "b9e1d2588591b437d02f1134b3f13e1ffede7cc5de3f6e56837dbfb891df9871 to execute this one real-F "
    "forward (64 states, at most 129 forwards, frozen budgets 60/10/180 s)?' and selected the "
    "approval option. This authorises ONE scoring-interface pass over the 64 frozen states and "
    "nothing else: no F-vs-D-Geo comparison, no training, no L2/L3, no 40-day batch, no final test, "
    "no install. PASS/FAIL/timeout/kill all consume the single opportunity.")

credential = dict(template, approved_by=APPROVED_BY, approved_utc=APPROVED_UTC,
                  approval_statement=STATEMENT)
credential = AUTH.seal_template(credential)

ok, detail = AUTH.validate_credential(credential, identity=ident, forward_targets=fwd,
                                      output_dir=RUN_DIR,
                                      preregistration_sha256=runner.sha256_file(PREREG))
print('validated against the frozen identity:', ok)
if not ok:
    print(json.dumps(detail['problems'], indent=1))
    raise SystemExit('REFUSING to write a credential that does not match the frozen identity')
pre = AUTH.validate_pre_registration(PREREG, credential['preregistration_sha256'])
print('pre-registration binding:', pre['ok'], pre['actual'])
if not pre['ok']:
    raise SystemExit('REFUSING: the pre-registration digest does not match')

OUT.write_text(json.dumps(credential, indent=2, sort_keys=True, default=repr), encoding='utf-8')
print(json.dumps({'written': str(OUT), 'run_id': credential['run_id'],
                  'credential_sha256': credential['credential_sha256'],
                  'approved_by': credential['approved_by'],
                  'human_fields_filled': [f for f in AUTH.HUMAN_FIELDS
                                          if str(credential.get(f) or '').strip()]}, indent=1))
