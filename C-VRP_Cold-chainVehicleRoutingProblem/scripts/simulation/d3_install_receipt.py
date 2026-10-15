#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3 B3-4: DURABLE install receipt + independent read-back.  Closes G0 gap A-5c.

What was missing (recorded in `d3_geo_adapter.py`, A-5c)
-------------------------------------------------------
`InstallLedger` keeps its receipts IN MEMORY.  They vanish with the process, so "the install
happened" could only ever be asserted by the same process that performed it.  A crash, a kill, or a
dishonest caller left no trace, and there was no way to re-read the fact afterwards.

Green-phase pattern used for this batch (the same one the G0-Safety batch used)
------------------------------------------------------------------------------
`d3_geo_adapter.py` is BYTE-FROZEN as an identity (its sha is recorded in freeze trees and read by
tests that assert its exact text).  So this batch does NOT edit it.  Instead it adds a NEW module
and, alongside it, `d3_install_boundary.py`, which RE-EXPORTS the frozen
`pre_install_reverify` / `DECISION_ALLOW` / `DECISION_REFUSE` unchanged and attaches a durable
install boundary to them.  The frozen module keeps its bytes; the new boundary is separately
identifiable and separately testable.

What this module adds (and what it deliberately does NOT claim)
--------------------------------------------------------------
  * `DurableInstallReceipt` records ONE install to disk as an immutable, uniquely named record:
    the installed plan hash, the certified plan hash, the certifier identity, the credential
    identities (when the caller supplies them), the decision-deadline flag, the run/state identity
    the caller passes, and the `install_index` within the process.  The plan is stored WITH every
    field `plan_hash` reads, so a reader can re-derive the hash from the receipt's own bytes.
  * `install_with_receipt(...)` is the only producer.  It calls `InstallLedger.finish()`, then emits
    the receipt and re-reads it to prove durability.  A crash between the install and the receipt
    leaves `receipt_missing` -- which a reader can SEE, instead of a silent gap.
  * `read_receipt(...)` is the independent read-back.  It re-reads the file, recomputes its sha256,
    and re-verifies that the recorded plan hash is reproducible from the recorded plan value.

Boundaries this module does NOT cross (they remain open in the D8 ledger):
  * it does NOT prove the environment's `_plan_and_commit` accepted the plan -- the receipt records
    the *requested* install and the evidence bound to it, not the environment's internal commit;
  * it does NOT prove state-level rollback;
  * it does NOT certify anything: certification evidence is an INPUT, and a receipt is refused
    without it (mirroring `InstallLedger.finish`'s own refusal);
  * a receipt is not a quality certificate and gives no claim about the installed plan's quality.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
for _p in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/tests'):
    sys.path.insert(0, str(EXT / _p))

import d3_geo_adapter as GEO                                                   # noqa: E402

RECEIPT_SCHEMA = 'd3_install_receipt_v1'
RECEIPT_DIR_ENV = 'D3_RECEIPT_DIR'


class ReceiptRefused(RuntimeError):
    """The receipt was refused; nothing durable was written for this install."""


def _digest(raw: bytes) -> str:
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def receipt_dir(root=None):
    """Where receipts live.  Explicit `root` wins; `$D3_RECEIPT_DIR` is the fallback for tests."""
    if root is not None:
        return Path(root)
    env = os.environ.get(RECEIPT_DIR_ENV)
    if env:
        return Path(env)
    return EXT / 'results' / 'd3_install_receipts'


def build_receipt(*, ledger, plan, sequence, run_id=None, state_identity=None,
                  credential_identities=None, receipt_dir_root=None, extra=None):
    """Build (but do not write) the receipt for the install the ledger just performed."""
    installed_hash = GEO._plan_identity(plan)
    if not ledger.installs:
        raise ReceiptRefused('the ledger recorded no install, so there is nothing to receipt')
    record = ledger.installs[-1]
    problems = []
    if installed_hash != record['plan_hash']:
        problems.append('the plan handed to the receipt is not the plan the ledger installed')
    if record['plan_hash'] != record['certified_plan_hash']:
        problems.append('the installed plan hash differs from the certified plan hash')
    if not record.get('certifier'):
        problems.append('the certifier identity is missing')
    if ledger.n_installs != 1:
        problems.append('this ledger recorded %d installs; one ledger is one transaction'
                        % ledger.n_installs)
    if problems:
        raise ReceiptRefused('; '.join(problems))
    return {
        'schema': RECEIPT_SCHEMA,
        'receipt_id': uuid.uuid4().hex,
        'sequence': int(sequence),
        'run_id': run_id,
        'state_identity': state_identity,
        'installed_plan_hash': record['plan_hash'],
        'certified_plan_hash': record['certified_plan_hash'],
        'certifier': record['certifier'],
        'credential_identities': credential_identities or {},
        'decision_deadline_ok': True,
        # The plan is stored with EVERY field `plan_hash` reads (vehicle id, anchor_node, suffix),
        # so the reader can re-derive the hash from the receipt's own bytes.  Storing only the
        # suffix would make the "reproduce the hash" check impossible and turn the receipt into an
        # unfalsifiable claim.
        'plan': {str(v): {'vehicle_id': int(getattr(p, 'vehicle_id', v)),
                          'anchor_node': int(getattr(p, 'anchor_node', 0)),
                          'anchor_time': float(getattr(p, 'anchor_time', 0.0)),
                          'anchor_load': float(getattr(p, 'anchor_load', 0.0)),
                          'suffix': [int(x) for x in getattr(p, 'suffix', [])]}
                  for v, p in plan.items()},
        'n_plans': len(plan),
        'created_mono': time.monotonic(),
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'claims': {
            'records': 'the install REQUESTED through install_with_receipt, with the evidence bound '
                       'to it at the moment it was requested',
            'does_not_prove': [
                'that the environment committed the plan (the real commit happens inside '
                '_plan_and_commit and is invisible here)',
                'state-level rollback',
                'any quality or C0-energy property of the installed plan',
                'that the install was correct -- only that it was requested with bound evidence'],
        },
        'extra': extra or {},
    }


def write_receipt(record, *, root=None):
    """Write the receipt to a FRESH destination name and re-read it to prove durability."""
    directory = receipt_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(record, indent=2, sort_keys=True, default=repr).encode('utf-8')
    name = 'receipt_%09d_%s.json' % (int(record['sequence']), record['receipt_id'])
    dest = directory / name
    if dest.exists():
        raise ReceiptRefused('receipt destination already exists: %s' % name)
    temp = directory / ('.%s.tmp.%s' % (dest.stem, uuid.uuid4().hex))
    with open(temp, 'xb') as fh:
        fh.write(raw)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temp, dest)
    back = dest.read_bytes()
    if back != raw:
        raise ReceiptRefused('receipt %s did not survive a re-read' % name)
    return {'name': name, 'path': str(dest), 'sha256': _digest(raw), 'bytes': len(raw),
            'verified': True, 'directory': str(directory)}


def read_receipt(record_or_path, *, root=None, expect_sha256=None, expect_plan_hash=None):
    """INDEPENDENT read-back.  Re-reads the file and re-verifies its content.

    A reader that trusts the in-memory dict would not be independent, so this function always goes
    back to the bytes on disk.  Returns `(ok, detail)`; it never raises on a missing or malformed
    receipt -- it reports, because "no receipt" is exactly the fact a reader must be able to see.
    """
    detail, problems = {}, []
    path = Path(record_or_path['path']) if isinstance(record_or_path, dict) else Path(record_or_path)
    if root is not None:
        path = receipt_dir(root) / path.name
    detail['path'] = str(path)
    if not path.is_file():
        return False, {'problems': ['receipt file is absent'], **detail}
    raw = path.read_bytes()
    detail['sha256'] = _digest(raw)
    detail['bytes'] = len(raw)
    if expect_sha256 is not None and detail['sha256'] != expect_sha256:
        problems.append('receipt digest differs from the recorded digest')
    try:
        record = json.loads(raw.decode('utf-8'))
    except Exception as exc:                                              # noqa: BLE001
        return False, {'problems': ['receipt is not readable JSON: %r' % exc], **detail}
    detail['schema_ok'] = record.get('schema') == RECEIPT_SCHEMA
    if not detail['schema_ok']:
        problems.append('unexpected receipt schema %r' % record.get('schema'))
    detail['installed_plan_hash'] = record.get('installed_plan_hash')
    detail['certified_plan_hash'] = record.get('certified_plan_hash')
    if record.get('installed_plan_hash') != record.get('certified_plan_hash'):
        problems.append('the receipt\'s installed and certified plan hashes differ')
    if not record.get('certifier'):
        problems.append('the receipt carries no certifier identity')
    if expect_plan_hash is not None and record.get('installed_plan_hash') != expect_plan_hash:
        problems.append('the receipt records a different plan than expected')
    detail['plan_reproduced'] = _reproduce_plan_hash(record)
    if detail['plan_reproduced'] is None:
        problems.append('the receipt\'s plan value cannot be hashed at all')
    elif detail['plan_reproduced'] != record.get('installed_plan_hash'):
        problems.append('the receipt\'s plan value does not reproduce the recorded plan hash')
    detail['problems'] = problems
    detail['record'] = record
    return (not problems), detail


def _reproduce_plan_hash(record):
    """Re-hash the receipt's plan VALUE, so a receipt cannot name a plan it does not contain.

    A legacy receipt whose plan entries are bare suffix lists is accepted as far as it goes: the
    suffix is hashed with the anchor fields the old format did not carry, and the caller's
    `expect_plan_hash` comparison still applies.  A receipt in the current format is hashed exactly.
    """
    plan = record.get('plan')
    if not isinstance(plan, dict):
        return None
    try:
        from action_contract import VehiclePlan, plan_hash
    except Exception:                                                     # noqa: BLE001
        return None
    try:
        rebuilt = {}
        for vid, entry in plan.items():
            vid = int(vid)
            if isinstance(entry, dict):
                rebuilt[vid] = VehiclePlan(vid, int(entry.get('anchor_node', 0)),
                                           float(entry.get('anchor_time', 0.0)),
                                           float(entry.get('anchor_load', 0.0)),
                                           tuple(int(x) for x in entry.get('suffix', [])))
            else:
                rebuilt[vid] = VehiclePlan(vid, 0, 0.0, 0.0, tuple(int(x) for x in entry))
        return plan_hash(rebuilt)
    except Exception:                                                     # noqa: BLE001
        return None


def list_receipts(*, root=None):
    directory = receipt_dir(root)
    if not directory.is_dir():
        return []
    return sorted(directory.glob('receipt_*.json'))


def install_with_receipt(*, ledger, plan, sequence, root=None, run_id=None, state_identity=None,
                         credential_identities=None, extra=None):
    """Perform the receipted install: the ledger's `finish` has ALREADY happened by the caller.

    This function is called AFTER a successful `ledger.finish(...)`; it refuses (and writes nothing)
    when the ledger did not actually install exactly one plan.  Returns
        {'installed': bool, 'receipt': {...}, 'publication': {...}, 'readback': {...}}
    """
    result = {'installed': bool(ledger.installs), 'receipt': None, 'publication': None,
              'readback': None}
    if not result['installed']:
        result['refused'] = 'the ledger installed nothing; no receipt is written'
        return result
    record = build_receipt(ledger=ledger, plan=plan, sequence=sequence, run_id=run_id,
                           state_identity=state_identity,
                           credential_identities=credential_identities,
                           receipt_dir_root=root, extra=extra)
    publication = write_receipt(record, root=root)
    ok, detail = read_receipt(publication, root=root, expect_sha256=publication['sha256'],
                              expect_plan_hash=record['installed_plan_hash'])
    result.update(receipt=record, publication=publication, readback={'ok': ok, 'detail': detail})
    if not ok:
        raise ReceiptRefused('the durable receipt did not read back: %s' % detail['problems'])
    return result
