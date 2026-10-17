#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G-1 PREPARATION -- feasibility probe for the time-identity ENCODING only (DIAGNOSTIC).

The G-1 roadmap says: kill `round(x, 9)` in `snapshot_state_identity`.  Before writing any G-1
candidate this probe answers one narrow question:

    is there a lossless, deterministic, platform-independent encoding of the quantities the identity
    must bind -- and how bad is the current canonicaliser?

It is NOT the G-1 identity object, it changes nothing, and it touches no production module.  Every
result here is about the ENCODING, not about certification.

Usage: python scripts/tests/test_q3_g1_encoding_probe.py --out <json path>
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

ROWS = []
LIMITS = ('pure CPU arithmetic on synthetic values; no data, no model, no rollout, no production '
          'module imported or modified')


def ck(cid, name, ok, detail=None, conclusion=''):
    st = 'PASS' if ok else 'FAIL'
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'status': st, 'detail': detail or {},
                 'conclusion': conclusion, 'limits': LIMITS})
    print('[%s] %-6s %s' % (st, cid, name[:94]))


# ---------------------------------------------------------------------------------------------
# candidate canonicalisers
# ---------------------------------------------------------------------------------------------
def enc_bits(x):
    """Bit-exact, endian-fixed, total on -0.0 / +0.0 / nan / inf.  The proposed canonical form."""
    return struct.pack('>d', float(x)).hex()


def enc_hex(x):
    """Human-readable, still lossless for finite values (loses the NaN payload)."""
    return float(x).hex()


def enc_round9(x):
    """The CURRENT canonicaliser in `m1b_immutable_snapshot.snapshot_state_identity`."""
    return repr(round(float(x), 9))


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    # ============================================================ P-01 round(x,9) is not injective
    t0 = 24.0
    # the smallest collapse: two doubles whose 9-decimal rounding is identical
    a, b = t0 + 1e-12, t0 + 2e-12
    ulp_at_24 = math.ulp(t0)
    window = 1e-9
    window_ulps = window / ulp_at_24
    collapse = [t for t in (a, b) if enc_round9(t) == enc_round9(t0)]
    ok1 = (a != b and enc_round9(a) == enc_round9(b) == enc_round9(t0)
           and enc_bits(a) != enc_bits(b) and len(collapse) == 2)
    ck('P-01', 'the current canonicaliser is LOSSY: two distinct doubles around 24.0 (1e-12 and 2e-12 '
       'apart) both round to 24.0, while the bit encoding separates them -- so a state identity built '
       'on `round(x,9)` cannot see a drift smaller than 1e-9',
       ok1,
       {'a': a, 'b': b, 'round9_a': enc_round9(a), 'round9_b': enc_round9(b),
        'bits_a': enc_bits(a), 'bits_b': enc_bits(b)},
       conclusion='round(x,9) is not a canonical form, it is a filter')

    # ============================================================ P-02 how much drift is invisible
    ok2 = (window_ulps > 1e5)
    ck('P-02', 'quantified: at the registered clock bound the 1e-9 window is about 2.8e5 ulps wide, so '
       'the current identity is blind to a drift five orders of magnitude coarser than one ulp -- the '
       'exact alarm the identity exists to raise',
       ok2,
       {'ulp_at_24': ulp_at_24, 'round_window': window, 'round_window_in_ulps': window_ulps,
        'exact': '%d ulps' % int(window_ulps)},
       conclusion='the canonicaliser must be bit-level, not decimal')

    # ============================================================ P-03 bit encoding round-trips
    samples = [0.0, 1e-9, 0.1, 23.899999618530273, 24.0, 24.000001, 1e6, -0.0, 0.30000000000000004]
    rt_ok = all(struct.unpack('>d', bytes.fromhex(enc_bits(x)))[0] == x
                or (x == 0.0 and struct.unpack('>d', bytes.fromhex(enc_bits(x)))[0] == 0.0)
                for x in samples)
    hex_ok = all(float.fromhex(enc_hex(x)) == x for x in samples if math.isfinite(x))
    ok3 = rt_ok and hex_ok
    ck('P-03', 'the proposed encoding round-trips EXACTLY: `struct.pack(">d").hex()` for every sample '
       'including 0.1 and 0.30000000000000004, and `float.hex()` in parallel so a human-readable echo '
       'is available without a second source of truth',
       ok3,
       {'encoded': {repr(x): enc_bits(x) for x in samples},
        'struct_roundtrip': rt_ok, 'float_hex_roundtrip': hex_ok},
       conclusion='lossless encoding exists and is trivial')

    # ============================================================ P-04 signed zero
    # Correction of my own first draft: I expected `round(x,9)` to collapse -0.0 into 0.0.  It does
    # NOT -- `repr(round(-0.0, 9))` is '-0.0'.  Signed zero is therefore not a defect of the current
    # canonicaliser; it is a POLICY question for the bit-level one, because -0.0 == 0.0 numerically
    # while the two bit patterns differ, so a bit-hashing identity would report "two states" for one
    # instant.  The rule is: a time must be finite, >= 0 and NOT negative zero.
    ok4 = (enc_bits(0.0) != enc_bits(-0.0) and enc_hex(0.0) != enc_hex(-0.0)
           and enc_round9(0.0) != enc_round9(-0.0)
           and (0.0 == -0.0)
           and math.copysign(1.0, -0.0) < 0)
    ck('P-04', 'signed zero: the bit encoding distinguishes it, the current canonicaliser also keeps '
       'the sign, and the two compare EQUAL numerically -- so the identity must carry an explicit '
       'policy (refuse -0.0 as a timestamp) instead of relying on either hashing or comparison to '
       'notice it',
       ok4,
       {'bits_plus': enc_bits(0.0), 'bits_minus': enc_bits(-0.0),
        'hex_plus': enc_hex(0.0), 'hex_minus': enc_hex(-0.0),
        'round9_plus': enc_round9(0.0), 'round9_minus': enc_round9(-0.0),
        'numeric_equal': True,
        'corrected_claim': ('my first draft asserted the current canonicaliser collapses -0.0; '
                            'measured, it does not.  The remaining argument is the policy one.'),
        'rule': 'a time must be finite and >= 0; -0.0 is REFUSED as a time, encoded as a bit value'},
       conclusion='the encoding can tell, so the policy can refuse')

    # ============================================================ P-05 NaN / inf
    specials = [float('nan'), float('inf'), float('-inf')]
    detected = [x for x in specials if not math.isfinite(x)]
    ok5 = (len(detected) == 3
           and all(enc_bits(x) for x in specials)
           and len({enc_bits(x) for x in specials}) == 3)
    ck('P-05', 'NaN and both infinities are DETECTABLE before encoding (so they can be refused as '
       'times) and are still distinct bit patterns (so the refusal is auditable); the encoding never '
       'silently normalises them',
       ok5,
       {'values': [repr(x) for x in specials],
        'bits': [enc_bits(x) for x in specials],
        'policy': 'REFUSED for time fields; encoded only in the audit record of the refusal'},
       conclusion='no NaN can enter an identity silently')

    # ============================================================ P-06 one ulp changes the digest
    x = 24.0
    y = math.nextafter(x, 25.0)
    ok6 = (x != y and enc_bits(x) != enc_bits(y)
           and digest({'t': enc_bits(x)}) != digest({'t': enc_bits(y)}))
    ck('P-06', 'a one-ulp drift changes the composite digest: this is the control the current identity '
       'cannot pass, because `round(x,9)` maps both values to the same string',
       ok6,
       {'x': x, 'y': y, 'ulp': math.ulp(x),
        'digest_x': digest({'t': enc_bits(x)}), 'digest_y': digest({'t': enc_bits(y)}),
        'round9_equal': enc_round9(x) == enc_round9(y)},
       conclusion='the identity can be made sensitive at the ulp level')

    # ============================================================ P-07 composite binding
    base = {'contract_hash': 'c', 'instance_id': 0, 'clock': enc_bits(24.0),
            'anchor': enc_bits(23.9), 'units': enc_bits(1.0), 'n_max': 20003,
            'T_clock_sup': enc_bits(25.414213562373096)}
    d0 = digest(base)
    mutated = []
    for key, value in (('clock', enc_bits(math.nextafter(24.0, 25.0))),
                       ('anchor', enc_bits(math.nextafter(23.9, 25.0))),
                       ('instance_id', 1), ('units', enc_bits(2.0)),
                       ('contract_hash', 'other'), ('n_max', 20002),
                       ('T_clock_sup', enc_bits(24.0))):
        m = dict(base)
        m[key] = value
        mutated.append({'field': key, 'digest': digest(m)[:16],
                        'changed': digest(m) != d0})
    ok7 = (all(e['changed'] for e in mutated))
    ck('P-07', 'the composite identity binds seven components (contract, instance, clock, anchor, '
       'units, advance budget, clock upper bound) and changing ANY of them changes the digest',
       ok7,
       {'base_digest': d0[:16], 'mutations': mutated},
       conclusion='every field the review asked to bind actually participates')

    # ============================================================ P-08 sample round-trip over the bound
    n = 1000
    lo, hi = 0.0, 25.414213562373096
    ok8 = True
    for i in range(n):
        v = lo + (hi - lo) * i / (n - 1)
        if struct.unpack('>d', bytes.fromhex(enc_bits(v)))[0] != v:
            ok8 = False
            break
    ck('P-08', 'over 1000 samples spanning the whole registered clock range, the encoding is '
       'bit-exactly invertible, so the losslessness claim is not just about hand-picked values',
       ok8, {'n_samples': n, 'range': [lo, hi]},
       conclusion='the encoding is total over the domain it will actually see')

    # ---------------------------------------------------------------- summary
    n_pass = sum(1 for r in ROWS if r['status'] == 'PASS')
    n_fail = sum(1 for r in ROWS if r['status'] == 'FAIL')
    verdict = 'G1_ENCODING_PROBE_FAIL' if n_fail else 'G1_ENCODING_PROBE_PASS'
    payload = {
        'schema': 'q3_g1_encoding_probe',
        'kind': 'DIAGNOSTIC', 'not_for_production': True,
        'baseline_head': 'a71a0c3cede64b15e20f34f7833134d6236bb7a1',
        'target': ('scripts/m1b_adapter/m1b_immutable_snapshot.py:176 snapshot_state_identity, which '
                   'currently canonicalises four fields with round(x, 9)'),
        'proposed_encoding': {
            'primary': 'struct.pack(">d", x).hex()  (bit-exact, endian-fixed, total)',
            'echo': 'float(x).hex()  (lossless for finite values, human-readable)',
            'policy': 'times must be finite, >= 0 and not -0.0; violations are REFUSED, never encoded',
            'rejected': 'round(x, 9) -- lossy, collapses ~2.8e5 ulps at the registered clock bound',
        },
        'verdict': verdict, 'n_rows': len(ROWS), 'n_pass': n_pass, 'n_fail': n_fail,
        'rows': ROWS,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    print('\n=== %d probe rows: %d PASS, %d FAIL -> %s ===' % (len(ROWS), n_pass, n_fail, verdict))
    print('WROTE %s' % args.out)
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
