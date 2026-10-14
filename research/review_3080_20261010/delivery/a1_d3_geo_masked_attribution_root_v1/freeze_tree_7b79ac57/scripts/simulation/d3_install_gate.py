#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B2-3: PRE-INSTALL credential re-verification.

Authorised scope: re-verify the three identities before any write-back.  This module DOES NOT
INSTALL, does not touch `_commit` / `_plan_and_commit`, and does not claim state-level rollback.

The invariant (user ruling):
    H(S_cert) == H(S_install)   AND   H(C_cert) == H(C_install)   AND   H(P_cert) == H(P_install)
Any inequality => REFUSE.

And, explicitly, identity matching is NOT sufficient (user ruling): the three equalities do NOT
replace
  * the decision-deadline check (`deadline_ok`),
  * the completeness of certification (`certification_complete`), or
  * the ACTUAL install-result check -- which this module cannot express at all, because the real
    commit happens inside `StrictOnlineEnv._plan_and_commit` after the replanner returns, is
    invisible to the adapter, and `_commit` performs no validation (C5).  That check belongs to B3.
`allow=True` therefore means only "nothing known REFUSES the install", never "the install
happened" and never "the state was committed".
"""
from __future__ import annotations

from d3_credential import credential_identity_check, validate_credential

DECISION_ALLOW = 'ALLOW_PENDING_B3_INSTALL_RESULT'
DECISION_REFUSE = 'REFUSE'

# Reasons that only this layer can produce; recorded so a refusal is always attributable.
EXTRA_REFUSALS = ('deadline_exceeded', 'certification_incomplete', 'credential_structurally_invalid')


def pre_install_reverify(cred_cert, cred_install, *, deadline_ok, certification_complete,
                         require_anchor='post_prepare'):
    """Decide whether an install may proceed, given the certification and install-time credentials.

    `certification_complete` must come from the composed certifier and must be its `installable`
    value (i.e. all executable checks passed AND no blocking capability gap stands), not merely
    its `ok` value.
    """
    refusals, detail = [], {}

    ok_cert, why_cert = validate_credential(cred_cert, require_anchor=require_anchor)
    ok_inst, why_inst = validate_credential(cred_install, require_anchor=require_anchor)
    detail['cert_structural'] = {'ok': ok_cert, 'reasons': why_cert}
    detail['install_structural'] = {'ok': ok_inst, 'reasons': why_inst}
    if not ok_cert or not ok_inst:
        refusals.append('credential_structurally_invalid')

    identity_ok, identity_reasons, identity_detail = credential_identity_check(cred_cert,
                                                                              cred_install)
    detail['identity'] = identity_detail
    refusals.extend(identity_reasons)

    if deadline_ok is False:
        refusals.append('deadline_exceeded')
    if certification_complete is not True:
        refusals.append('certification_incomplete')

    return {
        'gate_schema': 'pre_install_reverify_v1',
        'decision': DECISION_ALLOW if not refusals else DECISION_REFUSE,
        'allow': not refusals,
        'refusals': sorted(set(refusals)),
        'detail': detail,
        'identity_matching_is_not_sufficient': True,
        'checks_this_layer_cannot_perform': [
            'actual install result (the commit happens inside _plan_and_commit, after the '
            'replanner returns, and _commit validates nothing) -- B3',
            'state-level rollback (no replanner keeps a pre/post STATE pair)',
            'durable receipt (not implemented)',
        ],
        'note': ('allow=True means only that nothing known refuses the install; it is NOT evidence '
                 'that the real environment installed or committed anything'),
    }
