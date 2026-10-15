#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3 B3-4 acceptance: durable install receipt + INDEPENDENT read-back (closes G0 gap A-5c).

The whole point is that the reader must be able to see the fact AFTER the process that produced it
is gone, and must be able to tell "no receipt" apart from "a receipt that says something else".
Every scenario runs the real `InstallLedger` gate; none of them installs into a live environment.

Scenarios (fixed in advance, one attempt each, failures kept):
  R1 receipted install   a certified, identity-bound install writes a receipt; a SEPARATE process
                         (a real subprocess) re-reads it and confirms the digest and plan hash
  R2 no install          a refused install (no certifier) writes NO receipt; the reader reports
                         "absent" -- a visible gap, not a silent pass
  R3 cert mismatch       a plan that differs from the certified one is refused by the gate and
                         writes NO receipt
  R4 tampered receipt    after a successful install the receipt bytes are modified; the reader's
                         digest check fails and the plan-hash reproduction fails
  R5 rolled-back state   after a successful install the plan value inside the receipt is altered;
                         the reader detects that the recorded hash no longer matches the plan value
  R6 sequence identity   two installs produce two DISTINCT receipts, both readable, neither
                         overwriting the other
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
for _p in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/tests'):
    sys.path.insert(0, str(EXT / _p))

import d3_geo_adapter as GEO                                                   # noqa: E402
import d3_install_receipt as R                                                # noqa: E402

EV = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
CASES = []


def ck(name, ok, detail=None):
    CASES.append((name, bool(ok), detail))
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)[:250]))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan(seed=1):
    """A small full-fleet plan; only its identity matters here."""
    from action_contract import VehiclePlan
    return {0: VehiclePlan(0, 0, 0.0, 0.0, (1, 2 + seed)),
            1: VehiclePlan(1, 0, 0.0, 0.0, (3,)),
            2: VehiclePlan(2, 0, 0.0, 0.0, ())}


def do_install(*, root, sequence, plan=None, certify=True):
    """Drive a REAL `InstallLedger` transaction and then the receipted install."""
    from d3_geo_adapter import InstallLedger
    plan = plan or make_plan()
    installed = {}
    ledger = InstallLedger(lambda p: installed.update(plan=p))
    certified_plan = plan
    certified_hash = GEO._plan_identity(certified_plan)
    ok = ledger.finish(failure_reason=None if certify else 'cert_fail',
                       plan=plan, original_plan=make_plan(0),
                       certified=True if certify else None,
                       certified_plan_hash=certified_hash if certify else None,
                       certifier='tests.d3_acceptance.fake_certifier' if certify else None,
                       deadline_ok=True)
    receipt = None
    if ok:
        receipt = R.install_with_receipt(
            ledger=ledger, plan=plan, sequence=sequence, root=root, run_id='d3-receipt-acceptance',
            state_identity={'state_hash': 'deadbeef' * 8},
            credential_identities={'cert': 'a' * 64, 'install': 'a' * 64})
    return {'ledger': ledger, 'installed': ok, 'receipt': receipt, 'plan': plan}


def r1_receipted_install(root):
    d = Path(tempfile.mkdtemp(prefix='r1_', dir=str(root)))
    out = do_install(root=d, sequence=0)
    if not out['installed'] or not out['receipt']:
        return False, {'reason': 'the receipted install did not happen', 'installed': out['installed']}
    pub = out['receipt']['publication']
    # a SEPARATE process re-reads the receipt: the producer is not the verifier
    script = ('import sys, json; sys.path.insert(0, r"%s"); sys.path.insert(0, r"%s");'
              ' import d3_install_receipt as R;'
              ' ok, detail = R.read_receipt(r"%s", expect_sha256=r"%s",'
              ' expect_plan_hash=r"%s");'
              ' print(json.dumps({"ok": ok, "problems": detail.get("problems"),'
              ' "installed_plan_hash": detail.get("installed_plan_hash"),'
              ' "plan_reproduced": detail.get("plan_reproduced"),'
              ' "schema_ok": detail.get("schema_ok")}, default=str))'
              % (EXT / 'scripts/simulation', EXT / 'scripts/coldchain', pub['path'],
                 pub['sha256'], out['receipt']['receipt']['installed_plan_hash']))
    proc = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    parsed = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else {}
    inproc_ok, inproc_detail = R.read_receipt(pub, root=d, expect_sha256=pub['sha256'])
    det = {'receipt_name': pub['name'], 'receipt_sha256': pub['sha256'],
           'producer_readback_ok': out['receipt']['readback']['ok'],
           'separate_process_ok': parsed.get('ok'), 'separate_process_stdout': parsed,
           'in_process_ok': inproc_ok, 'n_receipts': len(R.list_receipts(root=d)),
           'install_index_recorded': out['ledger'].n_installs}
    ok = (out['receipt']['readback']['ok'] and parsed.get('ok') is True and inproc_ok
          and parsed.get('schema_ok') is True
          and parsed.get('plan_reproduced') == out['receipt']['receipt']['installed_plan_hash']
          and det['n_receipts'] == 1)
    return ok, det


def r2_no_install_no_receipt(root):
    d = Path(tempfile.mkdtemp(prefix='r2_', dir=str(root)))
    out = do_install(root=d, sequence=0, certify=False)
    receipts = R.list_receipts(root=d)
    det = {'installed': out['installed'], 'n_receipts': len(receipts),
           'ledger_refusals': out['ledger'].n_refusals,
           'ledger_events': out['ledger'].events}
    ok, read_detail = R.read_receipt(d / 'receipt_000000000_nonexistent.json')
    det['reader_on_absent'] = {'ok': ok, 'problems': read_detail['problems']}
    return (out['installed'] is False and not receipts and ok is False
            and 'absent' in read_detail['problems'][0]), det


def r3_cert_mismatch(root):
    """The gate refuses when the installed plan is not the certified one -> no receipt is written."""
    from d3_geo_adapter import InstallLedger
    d = Path(tempfile.mkdtemp(prefix='r3_', dir=str(root)))
    plan, other = make_plan(1), make_plan(7)
    installed = {}
    ledger = InstallLedger(lambda p: installed.update(plan=p))
    ok = ledger.finish(failure_reason=None, plan=plan, original_plan=make_plan(0),
                       certified=True, certified_plan_hash=GEO._plan_identity(other),
                       certifier='tests.d3_acceptance.fake_certifier', deadline_ok=True)
    # The gate refused, so the ledger installed nothing: the receipt producer must decline with an
    # attributable reason instead of writing a receipt for an install that never happened.
    outcome = R.install_with_receipt(ledger=ledger, plan=plan, sequence=0, root=d)
    det = {'gate_installed': ok, 'gate_refusal_reason': ledger.events[-1].get('reason'),
           'receipt_producer_installed': outcome['installed'],
           'receipt_producer_refused': outcome.get('refused'),
           'n_receipts': len(R.list_receipts(root=d)), 'installs_in_env': len(installed)}
    return (ok is False and outcome['installed'] is False and not R.list_receipts(root=d)
            and not installed and ledger.n_installs == 0
            and ledger.events[-1].get('reason') == 'certified_plan_identity_mismatch'), det


def r4_tampered_receipt(root):
    d = Path(tempfile.mkdtemp(prefix='r4_', dir=str(root)))
    out = do_install(root=d, sequence=0)
    pub = out['receipt']['publication']
    path = Path(pub['path'])
    raw = json.loads(path.read_text(encoding='utf-8'))
    raw['installed_plan_hash'] = 'f' * len(raw['installed_plan_hash'])
    path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding='utf-8')
    ok, detail = R.read_receipt(pub, root=d, expect_sha256=pub['sha256'])
    det = {'reader_ok': ok, 'problems': detail['problems'],
           'new_sha256': detail.get('sha256'), 'original_sha256': pub['sha256']}
    return (ok is False and any('digest differs' in p for p in detail['problems'])
            and any('do not reproduce' in p or 'differ' in p for p in detail['problems'])), det


def r5_plan_value_altered(root):
    """The receipt survives a byte-level re-read but its plan VALUE no longer hashes to the record."""
    d = Path(tempfile.mkdtemp(prefix='r5_', dir=str(root)))
    out = do_install(root=d, sequence=0)
    pub = out['receipt']['publication']
    path = Path(pub['path'])
    raw = json.loads(path.read_text(encoding='utf-8'))
    first_vid = sorted(raw['plan'])[0]
    raw['plan'][first_vid]['suffix'] = list(raw['plan'][first_vid]['suffix']) + [99]
    path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding='utf-8')
    new_sha = sha(path)
    ok, detail = R.read_receipt(path, expect_sha256=new_sha, expect_plan_hash=None)
    det = {'reader_ok': ok, 'problems': detail.get('problems'),
           'plan_reproduced': detail.get('plan_reproduced'),
           'recorded': detail.get('installed_plan_hash')}
    return (ok is False
            and any('does not reproduce' in p for p in detail['problems'])), det


def r6_sequence_identity(root):
    d = Path(tempfile.mkdtemp(prefix='r6_', dir=str(root)))
    a = do_install(root=d, sequence=0, plan=make_plan(1))
    b = do_install(root=d, sequence=1, plan=make_plan(2))
    receipts = R.list_receipts(root=d)
    ok_a, det_a = R.read_receipt(a['receipt']['publication'], root=d,
                                 expect_sha256=a['receipt']['publication']['sha256'])
    ok_b, det_b = R.read_receipt(b['receipt']['publication'], root=d,
                                 expect_sha256=b['receipt']['publication']['sha256'])
    det = {'n_receipts': len(receipts), 'names': [p.name for p in receipts],
           'distinct_names': len({p.name for p in receipts}) == len(receipts),
           'a_ok': ok_a, 'b_ok': ok_b,
           'hashes_differ': a['receipt']['receipt']['installed_plan_hash']
           != b['receipt']['receipt']['installed_plan_hash'],
           'a_problems': det_a.get('problems'), 'b_problems': det_b.get('problems')}
    return (len(receipts) == 2 and det['distinct_names'] and ok_a and ok_b
            and det['hashes_differ']), det


def r7_install_boundary(root):
    """The full boundary: frozen gate -> ledger install -> durable receipt, on ONE record.

    Also proves the boundary REUSES the frozen gate by object identity (it must not re-implement the
    pre-install check), and that a gate ALLOW without a real install is never reported as an install.
    """
    import d3_install_boundary as B
    import d3_credential as CRED
    from d3_geo_adapter import InstallLedger
    d = Path(tempfile.mkdtemp(prefix='r7_', dir=str(root)))
    ident = B.assert_frozen_gate_reused(
        gate_module_sha256=sha(EXT / 'scripts/simulation/d3_install_gate.py'))
    # Build a STRUCTURALLY COMPLETE credential from the frozen module's own required-field list, so
    # the gate's ALLOW path is exercised by real credentials rather than by a hand-written dict that
    # happens to omit a field.
    def cred(**over):
        base = {f: None for f in CRED.REQUIRED_FIELDS}
        base.update({'credential_schema': CRED.CREDENTIAL_SCHEMA,
                     'state_hash': 'a' * 64, 'state_hash_scope': CRED.STATE_HASH_SCOPE,
                     'state_capture_anchor': CRED.REQUIRED_STATE_CAPTURE_ANCHOR,
                     'raw_contract_reachable': True,
                     'contract_hash': 'b' * 64, 'effective_contract_hash': 'c' * 64,
                     'profile_hash': 'd' * 64, 'plan_hash_legacy': 'e' * 16,
                     'plan_digest_anchor_v1': 'f' * 64})
        base.update(over)
        return base

    cred_cert = cred()
    cred_install = cred()
    plan, other = make_plan(1), make_plan(2)

    # (a) gate ALLOW but the ledger installed nothing -> the boundary must NOT claim an install
    empty_ledger = InstallLedger(lambda p: None)
    no_install = B.install_boundary(ledger=empty_ledger, plan=plan, sequence=0, root=d,
                                    cred_cert=cred_cert, cred_install=cred_install, deadline_ok=True,
                                    certification_complete=True,
                                    credential_identities={'cert': 'x' * 64})

    # (b) the full good path
    installed = {}
    ledger = InstallLedger(lambda p: installed.update(plan=p))
    gate_hash = GEO._plan_identity(plan)
    ok = ledger.finish(failure_reason=None, plan=plan, original_plan=make_plan(0), certified=True,
                       certified_plan_hash=gate_hash,
                       certifier='tests.d3_acceptance.fake_certifier', deadline_ok=True)
    full = B.install_boundary(ledger=ledger, plan=plan, sequence=1, root=d,
                              cred_cert=cred_cert, cred_install=cred_install, deadline_ok=True,
                              certification_complete=True,
                              credential_identities={'cert': 'x' * 64, 'install': 'x' * 64})

    # (c) the gate REFUSES -> no install, no receipt
    refusing = B.install_boundary(ledger=InstallLedger(lambda p: None), plan=other, sequence=2,
                                  root=d, cred_cert=cred_cert, cred_install=cred_install, deadline_ok=False,
                                  certification_complete=True)
    det = {'gate_object_identity_reused': ident['object_identity_reused'],
           'gate_module': ident['gate_module'],
           'no_install_stages': no_install['stages'],
           'no_install_refused': no_install.get('refused'),
           'full_stages': full['stages'], 'full_gate_decision': full['gate']['decision'],
           'ledger_installed': ok,
           'refusing_stages': refusing['stages'], 'refusing_refusals': refusing['gate']['refusals'],
           'n_receipts': len(R.list_receipts(root=d))}
    return (ident['object_identity_reused']
            and no_install['stages']['ledger_installed'] is False
            and no_install['stages']['receipt_durable'] is False
            and 'gate ALLOW is not an install' in (no_install.get('refused') or '')
            and ok is True and full['stages'] == {'gate_allow': True, 'ledger_installed': True,
                                                  'receipt_durable': True, 'receipt_read_back': True}
            and refusing['stages']['gate_allow'] is False
            and refusing['gate']['refusals'] == ['deadline_exceeded']
            and det['n_receipts'] == 1), det


SCENARIOS = (
    ('R1 a receipted install is re-read by a SEPARATE process and reproduces its plan hash',
     r1_receipted_install),
    ('R2 a refused install writes NO receipt and the reader reports the gap', r2_no_install_no_receipt),
    ('R3 a plan that is not the certified one is refused and writes no receipt', r3_cert_mismatch),
    ('R4 tampered receipt bytes are detected by the digest check', r4_tampered_receipt),
    ('R5 an altered plan value no longer reproduces the recorded plan hash', r5_plan_value_altered),
    ('R6 two installs produce two distinct, independently readable receipts', r6_sequence_identity),
    ('R7 the full boundary reuses the FROZEN gate and an ALLOW alone is never an install',
     r7_install_boundary),
)


def main():
    outdir = Path(sys.argv[sys.argv.index('--outdir') + 1]) if '--outdir' in sys.argv else EV
    root = Path(tempfile.mkdtemp(prefix='d3_receipt_acc_'))
    rows = []
    try:
        for name, fn in SCENARIOS:
            started = time.monotonic()
            try:
                ok, detail = fn(root)
            except Exception as exc:                                      # noqa: BLE001
                import traceback
                ok, detail = False, {'exception': '%s: %s' % (type(exc).__name__, exc),
                                     'traceback': traceback.format_exc()}
            rows.append({'name': name, 'ok': bool(ok), 'wall_s': time.monotonic() - started,
                         'detail': detail})
            print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                                     json.dumps(detail, default=repr, sort_keys=True)[:250]))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    n_fail = sum(1 for r in rows if not r['ok'])
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        'schema': 'd3_install_receipt_acceptance',
        'closes_gap': 'A-5c no durable P_after_install receipt (in-memory receipts only)',
        'still_open': ['the environment\'s own commit is not observable from here '
                       '(_plan_and_commit)',
                       'state-level rollback is not proven by a receipt',
                       'no quality / C0-energy property of the installed plan is claimed'],
        'rows': rows, 'n_rows': len(rows), 'n_pass': len(rows) - n_fail, 'n_fail': n_fail,
        'tool_sha256': sha(__file__),
        'receipt_module_sha256': sha(EXT / 'scripts/simulation/d3_install_receipt.py'),
        'geo_adapter_sha256': sha(EXT / 'scripts/simulation/d3_geo_adapter.py'),
        'no_live_install': True, 'science_ready': False, 'stable_online_gain': False,
        'verdict': ('DURABLE_INSTALL_RECEIPT_ACCEPTED' if n_fail == 0
                    else 'DURABLE_INSTALL_RECEIPT_BLOCKED'),
    }
    (outdir / 'D3_INSTALL_RECEIPT_ACCEPTANCE.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    print('\n=== D3 install receipt acceptance: %d rows, %d PASS, %d FAIL ==='
          % (len(rows), len(rows) - n_fail, n_fail))
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
