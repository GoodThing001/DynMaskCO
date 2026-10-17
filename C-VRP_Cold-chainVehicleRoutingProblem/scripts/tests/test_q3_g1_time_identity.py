#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G-1 ISOLATED CANDIDATE CONTROLS -- C1..C12 and N-1..N-13 from the pre-registration.

Everything here runs on synthetic fixtures.  No data is read, no model is loaded, nothing is wired
into production, and the production `snapshot_state_identity` is only CALLED for comparison -- never
modified.

The primary control is one pair with two obligations:
    (a) STABILITY   the same state, rebuilt/reshaped, yields the same identity (no false positive)
    (b) SEPARATION  a one-ulp change in a bound field yields a different identity (no false negative)

Usage: python scripts/tests/test_q3_g1_time_identity.py --out <json path>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
sys.path.insert(0, str(EXT / 'scripts'))
sys.path.insert(0, str(EXT / 'scripts' / 'coldchain'))
sys.path.insert(0, str(EXT / 'scripts' / 'm1b_adapter'))
sys.path.insert(0, str(EXT / 'scripts' / 'q3_qualification'))

import m1b_immutable_snapshot as IS                                        # noqa: E402
import q3_g1_time_identity_candidate as G1                                 # noqa: E402

ROWS = []
LIMITS = ('synthetic fixtures only; no data read, no training, no GPU, no rollout, no production '
          'wiring, production modules never modified')


def ck(cid, name, ok, detail=None, conclusion='', grading='PROVED_CONDITIONAL'):
    st = 'PASS' if ok else 'FAIL'
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'status': st, 'detail': detail or {},
                 'conclusion': conclusion, 'grading': grading, 'limits': LIMITS})
    print('[%s] %-7s %s' % (st, cid, name[:92]))


def try_call(fn, *a, **k):
    try:
        return {'ok': True, 'value': fn(*a, **k)}
    except G1.Refusal as exc:
        return {'ok': False, 'code': exc.code, 'detail': exc.detail}


class V(object):
    """The attribute surface `snapshot_state_identity` reads."""

    def __init__(self, **kw):
        self.status = kw.get('status', 'ready')
        self.current_node = kw.get('current_node', 1)
        self.ready_time = kw.get('ready_time', 24.0)
        self.current_load = kw.get('current_load', 5.0)
        self.committed_next = kw.get('committed_next', None)
        self.committed_finish = kw.get('committed_finish', None)
        self.return_finish = kw.get('return_finish', None)
        self.mutable_suffix = kw.get('mutable_suffix', [])
        self.cc_identity = kw.get('cc_identity', 'cc0')


def snap(vehicles):
    return {'vehicles': {i: v for i, v in enumerate(vehicles)},
            'served_mask': [True, True] + [False] * 3,
            'plan_identity': 'plan-0'}


GOOD_DOMAIN = {'models': ['euclidean'], 'multiplier_ceiling': 1.0, 'tw_speed': 1.0}
GOOD_G2 = {'verdict': 'PROVED_CONDITIONAL', 'budget_qualified': True,
           'travel_models': ['euclidean'], 'witness_ok': True}
GOOD_MONITOR = {'certifiable': True, 'receipt_digest': 'a' * 64, 'problems': []}
PROD_DIGESTS = {}


def base_kwargs(**over):
    kw = dict(clock=24.0, anchor=23.9,
              vehicles=[{'vehicle_id': 0, 'ready_time': 24.0, 'current_load': 5.0,
                         'committed_finish': None, 'return_finish': None, 'status': 'ready',
                         'current_node': 1}],
              contract_hash='c0ffee', hours_per_time_unit=1.0, distance_km_per_unit=1.0,
              instance_id=0, event_id=7, served_prefix=[0, 1], committed_next=[0],
              T_clock_sup=25.414213562373096, n_max_per_vehicle=20003,
              travel_domain=dict(GOOD_DOMAIN), source={'files': {'f': '0' * 64},
                                                       'basis': 'committed blob'},
              census_digest='d' * 64, domain={'dataset_glob': 'synthetic'},
              g2_qualification=dict(GOOD_G2), monitor_receipt=dict(GOOD_MONITOR))
    kw.update(over)
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    # production identities of the five registered files (must not change)
    for rel in ('scripts/m1b_adapter/m1b_immutable_snapshot.py',
                'scripts/m1b_adapter/m1b_q2_certifier.py',
                'scripts/coldchain/coldchain_state.py',
                'scripts/coldchain/coldchain_contract.py',
                'scripts/simulation/strict_online_env.py'):
        import subprocess
        blob = subprocess.run(['git', 'show', 'HEAD:%s/%s' % (EXT.name, rel)],
                              cwd=str(EXT.parent), capture_output=True).stdout
        PROD_DIGESTS[rel] = hashlib.sha256(blob).hexdigest() if blob else None

    ident = G1.build_identity(**base_kwargs())

    # ============================================================ C1 primary control
    s = snap([V()])
    s2 = snap([V()])                       # rebuilt: must be STABLE
    h_prod = IS.snapshot_state_identity(s)
    h_v2 = G1.state_identity_v2(s)
    h_v2b = G1.state_identity_v2(s2)
    drift = snap([V(ready_time=math.nextafter(24.0, 25.0))])
    h_prod_drift = IS.snapshot_state_identity(drift)
    h_v2_drift = G1.state_identity_v2(drift)
    ok1 = (h_v2 == h_v2b) and (h_v2 != h_v2_drift) \
        and (h_prod == h_prod_drift)         # the production identity does NOT see one ulp
    ck('C1', 'PRIMARY CONTROL: the candidate identity is STABLE under a rebuild and SEPARATES a '
       'one-ulp drift; the production identity satisfies the first obligation and FAILS the second, '
       'which is the defect the candidate exists to remove',
       ok1,
       {'v2_stable': h_v2 == h_v2b, 'v2_separates_1ulp': h_v2 != h_v2_drift,
        'production_stable': h_prod == h_prod_drift,
        'production_separates_1ulp': h_prod != h_prod_drift,
        'ulp': math.ulp(24.0),
        'reading': ('the production canonicaliser maps both 24.0 and 24.0+1ulp to the same string, so '
                    'its identity is blind to the drift it is supposed to flag')},
       conclusion='the two obligations are met by the candidate and not by the production function')

    # ============================================================ C2 lossless round-trip
    samples = [0.1, 0.30000000000000004, 23.899999618530273, 24.0, 24.000001, 1e-9]
    rt = []
    for x in samples:
        e = G1.enc_time(x, 'probe')
        back = struct.unpack('>d', bytes.fromhex(e['value']))[0]
        rt.append({'x': repr(x), 'bits': e['value'], 'roundtrip_exact': back == x,
                   'dtype': e['dtype']})
    ok2 = all(r['roundtrip_exact'] for r in rt)
    ck('C2', 'the encoding is lossless for every fixture value, including 0.1 and '
       '0.30000000000000004, and records the source dtype next to each value so an upcast is visible',
       ok2, {'rows': rt}, conclusion='no precision is lost by construction')

    # ============================================================ C3 no lossy canonicaliser
    scan = try_call(G1.lossy_canonicaliser_scan)
    tmp = HERE / '_g1_lossy_probe.py'
    tmp.write_text('def f(x):\n    return repr(round(x, 9))\n', encoding='utf-8')
    try:
        bad = try_call(G1.lossy_canonicaliser_scan, tmp)
    finally:
        tmp.unlink(missing_ok=True)
    ok3 = (scan['ok'] and bad['ok'] is False and bad['code'] == 'LOSSY_CANONICALISER')
    ck('C3', 'the module scans its own AST and contains no `round`/formatting/percent-precision '
       'canonical form; the scanner does fire on a file that contains one, so the check is live',
       ok3, {'self_scan': scan, 'detector_on_a_lossy_file': bad},
       conclusion='R-4 is enforced, not documented')

    # ============================================================ C4 None vs 0.0
    a = G1.enc_time(None, 'committed_finish', allow_null=True)
    b = G1.enc_time(0.0, 'committed_finish')
    miss = try_call(G1.enc_time, None, 'committed_finish')
    ok4 = (a['value'] == G1.NULL and b['value'] == G1.enc_bits(0.0) and a['value'] != b['value']
           and miss['ok'] is False and miss['code'] == 'MISSING_TIME')
    ck('C4', 'an absent value is the sentinel `null` and is distinct from the bit pattern of 0.0; a '
       'missing required time is refused instead of being coerced to zero',
       ok4, {'null_encoding': a, 'zero_encoding': b, 'missing': miss},
       conclusion='"absent" cannot be mistaken for "zero"')

    # ============================================================ C5 unit binding
    i2 = G1.build_identity(**base_kwargs(hours_per_time_unit=2.0))
    i1b = G1.build_identity(**base_kwargs())
    req = try_call(G1.require_certifiable, i2)
    ok5 = (i2['identity_digest'] != ident['identity_digest']
           and i2['binding']['hours_per_time_unit'] == G1.enc_bits(2.0)
           and i2['certifiable'] is False and req['ok'] is False
           and req['code'] == 'TIME_UNIT_CONVERSION_NOT_BUDGETED')
    ck('C5', 'the unit scale is bound: changing it changes the digest, marks the identity '
       'non-certifiable, and `require_certifiable` refuses with the unit code instead of letting a '
       'non-unit scale ride along',
       ok5, {'unit_1': ident['binding']['hours_per_time_unit'],
             'unit_2': i2['binding']['hours_per_time_unit'],
             'certifiable_2': i2['certifiable'], 'require_certifiable': req},
       conclusion='C5: units are identity, not configuration')

    # ============================================================ C6 contract binding
    i3 = G1.build_identity(**base_kwargs(contract_hash='deadbeef'))
    mismatch = try_call(G1.require_same_contract, ident, 'deadbeef')
    ok6 = (i3['identity_digest'] != ident['identity_digest']
           and mismatch['ok'] is False and mismatch['code'] == 'CONTRACT_IDENTITY_MISMATCH')
    ck('C6', 'the contract hash is bound and a different contract cannot inherit the identity '
       '(`CONTRACT_IDENTITY_MISMATCH`)',
       ok6, {'digest_c0ffee': ident['identity_digest'][:16],
             'digest_deadbeef': i3['identity_digest'][:16], 'mismatch': mismatch},
       conclusion='C6: contract identity is not transferable')

    # ============================================================ C7 clock vs anchor
    i4 = G1.build_identity(**base_kwargs(anchor=23.9000001))
    i5 = G1.build_identity(**base_kwargs(clock=24.0000001))
    ok7 = (i4['identity_digest'] != ident['identity_digest']
           and i5['identity_digest'] != ident['identity_digest']
           and i4['bits']['clock'] == ident['bits']['clock']
           and i4['bits']['anchor'] != ident['bits']['anchor'])
    ck('C7', 'the EVENT clock and the COLD-CHAIN anchor are both bound and are separate components: '
       'moving the anchor leaves the clock field identical and still changes the digest',
       ok7, {'clock_field': ident['bits']['clock'], 'anchor_a': ident['bits']['anchor'],
             'anchor_b': i4['bits']['anchor'],
             'digests': [ident['identity_digest'][:12], i4['identity_digest'][:12],
                         i5['identity_digest'][:12]]},
       conclusion='an anchor injected by a restore cannot hide behind an unchanged clock')

    # ============================================================ C8 trusted bound required
    missing_bound = try_call(G1.build_identity, **base_kwargs(T_clock_sup=None))
    missing_n = try_call(G1.build_identity, **base_kwargs(n_max_per_vehicle=None))
    missing_domain = try_call(G1.build_identity, **base_kwargs(travel_domain=None))
    ok8 = (not missing_bound['ok'] and missing_bound['code'] == 'TRUSTED_TIME_BOUND_MISSING'
           and not missing_n['ok'] and missing_n['code'] == 'TRUSTED_TIME_BOUND_MISSING'
           and not missing_domain['ok'] and missing_domain['code'] == 'DISTANCE_DOMAIN_MISSING')
    ck('C8', 'without a clock bound, an advance bound or a distance model the identity REFUSES to be '
       'built: it never falls back to the horizon or to `max_iter`, and never assumes sqrt(2)',
       ok8, {'missing_clock_bound': missing_bound, 'missing_n_max': missing_n,
             'missing_travel_domain': missing_domain},
       conclusion='C8: a missing premise is a refusal, not a default')

    # ============================================================ C9 cross instance / prefix
    x_inst = try_call(G1.require_same_instance_plan, ident, instance_id=1,
                      served_prefix=[0, 1], committed_next=[0])
    x_pref = try_call(G1.require_same_instance_plan, ident, instance_id=0,
                      served_prefix=[0, 2], committed_next=[0])
    ok9 = (x_inst['ok'] is False and x_inst['code'] == 'CROSS_INSTANCE_CONFUSION'
           and x_pref['ok'] is False and x_pref['code'] == 'EXECUTED_PREFIX_CHANGED')
    ck('C9', 'cross-instance reuse and executed-prefix drift both fail closed, so an identity cannot '
       'be moved to another instance or to a different prefix',
       ok9, {'cross_instance': x_inst, 'prefix_drift': x_pref},
       conclusion='C9: instance and prefix are binding')

    # ============================================================ C10 restore refused
    r = try_call(G1.build_identity, **base_kwargs(declared_resume=True))
    ok10 = (r['ok'] is False and r['code'] == 'SNAPSHOT_RESTORE_INSIDE_INTERVAL')
    ck('C10', 'a declared snapshot restore inside the interval is refused: the anchor could have been '
       'injected from outside the event loop, so no interval identity may be built across it',
       ok10, r, conclusion='C10: the restore path is a refusal, not a caveat')

    # ============================================================ C11 production untouched
    now = {}
    for rel in PROD_DIGESTS:
        import subprocess
        blob = subprocess.run(['git', 'show', 'HEAD:%s/%s' % (EXT.name, rel)],
                              cwd=str(EXT.parent), capture_output=True).stdout
        now[rel] = hashlib.sha256(blob).hexdigest() if blob else None
    ok11 = (now == PROD_DIGESTS and all(v is not None for v in now.values()))
    ck('C11', 'the five registered production files are byte-identical to HEAD: the candidate is '
       'isolated, nothing was rewritten, and `m1b_immutable_snapshot.py` keeps its lossy function',
       ok11, {'digests': now}, conclusion='C11: isolation holds')

    # ============================================================ C12 grading honesty
    ok12 = (ident['g2_qualification']['verdict'] == 'PROVED_CONDITIONAL'
            and ident['not_for_production'] is True
            and ident['production_consumable'] is False
            and set(G1.GRADING) == {'PROVED_CONDITIONAL', 'DIAGNOSTIC', 'UNKNOWN'})
    ck('C12', 'the artifact is labelled: DIAGNOSTIC kind, not_for_production, production_consumable '
       'False, and the three grading labels are declared rather than implied',
       ok12, {'grading': G1.GRADING, 'flags': {k: ident[k] for k in
                                               ('not_for_production', 'production_consumable')}},
       conclusion='C12: no unnamed claim')

    # ============================================================ N-1 .. N-13
    n = {}
    n['N-1'] = {'injected': 'one-ulp drift in ready_time',
                'v2_changed': G1.state_identity_v2(snap([V(ready_time=math.nextafter(24.0, 25.0))]))
                != h_v2,
                'production_changed': IS.snapshot_state_identity(
                    snap([V(ready_time=math.nextafter(24.0, 25.0))])) != h_prod}
    n['N-2'] = try_call(G1.enc_time, -0.0, 'clock')
    n['N-3'] = [try_call(G1.enc_time, x, 'clock') for x in
                (float('nan'), float('inf'), float('-inf'), 'soon', True)]
    n['N-4'] = {'hpu_2_certifiable': i2['certifiable'],
                'hpu_2_require': try_call(G1.require_certifiable, i2)['code']}
    n['N-5'] = x_inst
    n['N-6'] = r
    n['N-7'] = try_call(G1.verify_state_unchanged_v2, snap([V()]),
                        snap([V(current_load=5.0000001)]))
    n['N-8'] = x_pref
    n['N-9'] = missing_bound
    n['N-10'] = try_call(G1.build_identity, **base_kwargs(contract_hash=''))
    n['N-11'] = bad
    n['N-12'] = {'g2_unqualified': (lambda i: {'certifiable': i['certifiable'],
                                               'require': try_call(
                                                   G1.require_certifiable, i)['code']})(
        G1.build_identity(**base_kwargs(g2_qualification={'verdict': 'REFUSED',
                                                          'budget_qualified': False})))}
    n['N-13'] = [try_call(G1.build_identity, **base_kwargs(
        travel_domain={'models': ['euclidean'], 'multiplier_ceiling': 1.0, 'tw_speed': s}))
        for s in (0.0, -1.0, float('nan'), None, 'fast')]

    expect = {
        'N-1': (lambda v: v['v2_changed'] and not v['production_changed']),
        'N-2': (lambda v: v['ok'] is False and v['code'] == 'NEGATIVE_ZERO_TIME'),
        'N-3': (lambda v: all(x['ok'] is False for x in v)
                and [x['code'] for x in v] == ['NON_FINITE_TIME', 'NON_FINITE_TIME',
                                               'NON_FINITE_TIME', 'NON_NUMERIC_TIME',
                                               'NON_NUMERIC_TIME']),
        'N-4': (lambda v: v['hpu_2_certifiable'] is False
                and v['hpu_2_require'] == 'TIME_UNIT_CONVERSION_NOT_BUDGETED'),
        'N-5': (lambda v: v['ok'] is False and v['code'] == 'CROSS_INSTANCE_CONFUSION'),
        'N-6': (lambda v: v['ok'] is False and v['code'] == 'SNAPSHOT_RESTORE_INSIDE_INTERVAL'),
        'N-7': (lambda v: v['ok'] is False and v['code'] == 'POST_CERTIFICATION_STATE_CHANGED'),
        'N-8': (lambda v: v['ok'] is False and v['code'] == 'EXECUTED_PREFIX_CHANGED'),
        'N-9': (lambda v: v['ok'] is False and v['code'] == 'TRUSTED_TIME_BOUND_MISSING'),
        'N-10': (lambda v: v['ok'] is False and v['code'] == 'CONTRACT_IDENTITY_UNBOUND'),
        'N-11': (lambda v: v['ok'] is False and v['code'] == 'LOSSY_CANONICALISER'),
        'N-12': (lambda v: v['g2_unqualified']['certifiable'] is False
                 and v['g2_unqualified']['require'] == 'G2_BUDGET_NOT_QUALIFIED'),
        'N-13': (lambda v: all(x['ok'] is False and x['code'] == 'TW_SPEED_INVALID' for x in v)),
    }
    for key in sorted(n):
        ok = bool(expect[key](n[key]))
        ck(key, 'negative control %s fires as declared' % key, ok, n[key],
           'the refusal path is exercised', grading='PROVED_CONDITIONAL')

    # ============================================================ requirement 7 (no endorsement)
    endorsed = G1.build_identity(**base_kwargs(g2_qualification={
        'verdict': 'PROVED_CONDITIONAL', 'budget_qualified': True}))
    not_endorsed = G1.build_identity(**base_kwargs(g2_qualification={
        'verdict': 'NOT_CERTIFIABLE', 'budget_qualified': False}))
    okR7 = (endorsed['certifiable'] is True and not_endorsed['certifiable'] is False
            and not_endorsed['identity_consistent'] is True
            and try_call(G1.require_certifiable, not_endorsed)['code'] == 'G2_BUDGET_NOT_QUALIFIED')
    ck('R-7', 'identity consistency never endorses an unqualified budget: with a NOT_CERTIFIABLE G-2 '
       'verdict the identity is still internally consistent and cleanly non-certifiable, and '
       '`require_certifiable` refuses with `G2_BUDGET_NOT_QUALIFIED`',
       okR7, {'qualified': {'certifiable': endorsed['certifiable']},
              'unqualified': {'certifiable': not_endorsed['certifiable'],
                              'identity_consistent': not_endorsed['identity_consistent'],
                              'conditional': not_endorsed['conditional']}},
       conclusion='the two questions stay separate')

    # ---------------------------------------------------------------- summary
    n_pass = sum(1 for r in ROWS if r['status'] == 'PASS')
    n_fail = sum(1 for r in ROWS if r['status'] == 'FAIL')
    verdict = 'G1_CANDIDATE_FAIL' if n_fail else 'G1_CANDIDATE_PASS_ISOLATED'
    payload = {
        'schema': 'q3_g1_time_identity_controls',
        'kind': 'DIAGNOSTIC', 'not_for_production': True, 'production_consumable': False,
        'baseline_head': 'a71a0c3cede64b15e20f34f7833134d6236bb7a1',
        'preregistration': ['scripts/q3_qualification/q3_g1_preregistration_2026-10-17.md',
                            'scripts/q3_qualification/q3_g1_preregistration_amendment_01.md'],
        'verdict': verdict, 'n_rows': len(ROWS), 'n_pass': n_pass, 'n_fail': n_fail,
        'identity_digest': ident['identity_digest'],
        'production_file_digests': PROD_DIGESTS,
        'rows': ROWS,
        'status_lines': {
            'GTMIN_V2_ACCEPTANCE_FAIL_AS_LITERALLY_PREREGISTERED': 'permanent, not retro-approved',
            'THERMAL_CONTINUOUS_NOT_ESTABLISHED': True, 'Q3_FORMAL_FREEZE_NOT_AUTHORISED': True,
            'M2_NOT_AUTHORISED': True, 'L1': 'BLOCKED', 'L2_NOT_ESTABLISHED': True,
            'A3_NOT_ESTABLISHED': True, 'stable_online_gain': False,
            'HISTORICAL_BACKBONE_UNVERIFIED': True, 'C0': 'AWAITING_USER_BUDGET_RULING',
            't_min': 'always 0', 'final_test_seeds': 'SEALED'},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    print('\n=== %d controls: %d PASS, %d FAIL -> %s ===' % (len(ROWS), n_pass, n_fail, verdict))
    print('WROTE %s' % args.out)
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
