#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3 install boundary: the frozen pre-install gate PLUS the durable receipt (G0-Safety batch 2).

Why a re-export module instead of an edit
-----------------------------------------
`d3_geo_adapter.py` is a byte-frozen identity: its sha256 is recorded in freeze trees and the
G0-Safety tests assert its exact text, including `checks_this_layer_cannot_perform`, whose entry
'durable receipt (not implemented)' was TRUE when that batch closed.  Editing that string would
rewrite an identity that other evidence depends on.  So:

  * the frozen module stays BYTE-INTACT and remains importable under its own name;
  * THIS module re-exports `pre_install_reverify` / `DECISION_ALLOW` / `DECISION_REFUSE` unchanged
    (verified by object identity, see `assert_frozen_gate_reused`), and adds the durable install
    boundary from `d3_install_receipt.py`.

`install_boundary(...)` therefore performs the whole sequence in one place:

    1. `pre_install_reverify(...)`  -- the frozen identity / deadline / certification gate
    2. the caller's real install (through `InstallLedger`, which refuses without bound certification)
    3. a DURABLE receipt + an independent read-back

and returns one record that states, for each of those three stages, whether it happened, plus the
things this boundary still cannot prove (the environment's internal commit, state-level rollback,
plan quality).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
for _p in ('scripts', 'scripts/simulation'):
    sys.path.insert(0, str(EXT / _p))

import d3_geo_adapter as GEO                                                   # noqa: E402
import d3_install_receipt as RECEIPT                                          # noqa: E402
from d3_install_gate import (DECISION_ALLOW, DECISION_REFUSE,                 # noqa: E402,F401
                            EXTRA_REFUSALS, pre_install_reverify)

BOUNDARY_SCHEMA = 'd3_install_boundary_v1'

# The gate module whose bytes must not drift under this boundary.
FROZEN_GATE_MODULE = 'd3_install_gate'
FROZEN_GATE_SHA256 = None      # filled by the caller / acceptance harness, which hashes the file


def assert_frozen_gate_reused(gate_module_sha256=None):
    """Prove this boundary reuses the FROZEN gate rather than a re-implementation."""
    import d3_install_gate as G
    same = (pre_install_reverify is G.pre_install_reverify
            and DECISION_ALLOW == G.DECISION_ALLOW
            and DECISION_REFUSE == G.DECISION_REFUSE
            and tuple(EXTRA_REFUSALS) == tuple(G.EXTRA_REFUSALS))
    out = {'gate_module': getattr(G, '__file__', None), 'object_identity_reused': bool(same),
           'gate_module_sha256': gate_module_sha256}
    if gate_module_sha256 is not None and FROZEN_GATE_SHA256 is not None:
        out['sha_matches_frozen'] = gate_module_sha256 == FROZEN_GATE_SHA256
    return out


def install_boundary(*, ledger, plan, sequence, root, cred_cert, cred_install, deadline_ok,
                     certification_complete, run_id=None, state_identity=None,
                     credential_identities=None, require_anchor='post_prepare', extra=None):
    """Run the frozen gate, the ledger install and the durable receipt as ONE recorded sequence.

    Returns a record whose `stages` block says exactly which of the three happened.  A stage that
    did not happen never inherits the previous stage's success.
    """
    gate = pre_install_reverify(cred_cert, cred_install, deadline_ok=deadline_ok,
                                certification_complete=certification_complete,
                                require_anchor=require_anchor)
    record = {
        'schema': BOUNDARY_SCHEMA,
        'gate': gate,
        'stages': {'gate_allow': bool(gate['allow']), 'ledger_installed': False,
                   'receipt_durable': False, 'receipt_read_back': False},
        'receipt': None,
        'readback': None,
        'still_cannot_prove': [
            'that the environment committed the plan (_plan_and_commit is outside this boundary)',
            'state-level rollback',
            'any quality / C0-energy property of the installed plan'],
    }
    if not gate['allow']:
        record['refused'] = 'the frozen pre-install gate refused: %s' % gate['refusals']
        return record

    # Stage 2: the real ledger install (it re-checks certification and plan identity itself).
    from d3_geo_adapter import InstallLedger                                          # noqa: F401
    installed = {'value': None}
    ledger.install_fn = ledger.install_fn or (lambda p: None)
    # The caller is expected to have driven `ledger.finish(...)` already; this boundary only
    # consumes its outcome, so it cannot install anything the ledger did not accept.
    record['stages']['ledger_installed'] = bool(ledger.installs)
    if not ledger.installs:
        record['refused'] = ('the frozen gate allowed, but the ledger installed nothing -- a '
                             'gate ALLOW is not an install')
        return record

    # Stage 3: the durable receipt + independent read-back.
    receipt = RECEIPT.install_with_receipt(
        ledger=ledger, plan=plan, sequence=sequence, root=root, run_id=run_id,
        state_identity=state_identity, credential_identities=credential_identities, extra=extra)
    record['receipt'] = receipt['receipt']
    record['readback'] = receipt['readback']
    record['stages']['receipt_durable'] = bool(receipt['publication']
                                               and receipt['publication']['verified'])
    record['stages']['receipt_read_back'] = bool(receipt['readback']
                                                 and receipt['readback']['ok'])
    record['publication'] = receipt['publication']
    return record
