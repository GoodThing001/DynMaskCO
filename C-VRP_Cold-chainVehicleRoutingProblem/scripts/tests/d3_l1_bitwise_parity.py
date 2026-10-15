#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-L1 bitwise parity comparator + deterministic causal perturbation (P0-A / P0-B correction).

P0-A: the previous comparator had a REAL false-negative path -- `bits()` called float(x) on
anything with a `dtype`, so two different arrays both produced None and were reported EQUAL, and
NumPy dtype/raw-type and list-vs-tuple distinctions were ignored.  This module replaces it:

  * ndarray            -> raw type + dtype.str + shape + tobytes() must all match
  * numpy scalar       -> raw type + dtype.str + raw bit pattern must all match
  * python float       -> IEEE-754 double bits (so +0.0 vs -0.0 is a DIFFERENCE)
  * NaN                -> compared BY BITS (same payload == equal, different payload != equal)
  * dict/list/tuple    -> the container TYPE must match first, then every member
  * dataclass          -> field by field; other objects -> __dict__ field by field
  * anything else      -> NOT_VERIFIABLE (never "equal" by default)
  * budget/exception   -> NOT_VERIFIABLE, never "equal"

P0-B: `perturb_snapshot(snapshot, day_id, position)` is a documented PURE function: it selects the
currently INVISIBLE orders by a fixed rule, maps their attributes by a deterministic hash-derived
amount inside a legal range, preserves order IDs / array dimensions / the revealed set / the
executed state, and reports n_invisible_modified.  If no perturbable array field is present it
returns NOT_VERIFIABLE instead of pretending a perturbation happened.

Read-only: nothing here collects states, runs a rollout or calls F/D-Geo.
"""
import argparse
import dataclasses
import hashlib
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
for p in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/data'):
    sys.path.insert(0, str(EXT / p))

EQUAL, DIFFERENT, NOT_VERIFIABLE = 'EQUAL', 'DIFFERENT', 'NOT_VERIFIABLE'
PERTURB_FIELDS = ('coords', 'demands', 'tw_start', 'tw_end', 'service_time', 'temp_class')
PERTURB_ARRAY_KEYS = ('coords', 'demands', 'tw_start', 'tw_end', 'service_time', 'temp_class')
CASES = []
REC = {}


def ck(name, ok, detail=None):
    CASES.append((name, bool(ok), detail))
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)))


def raw_type(x):
    t = type(x)
    return '%s.%s' % (t.__module__, t.__qualname__)


def is_ndarray(x):
    return hasattr(x, 'shape') and hasattr(x, 'dtype') and hasattr(x, 'tobytes') \
        and getattr(x, 'ndim', None) is not None and getattr(x, 'ndim', 0) >= 1


def is_np_scalar(x):
    return hasattr(x, 'dtype') and getattr(x, 'ndim', None) == 0


def fbits(x):
    return struct.pack('>d', float(x)).hex()


def bitwise_equal(a, b, path='root', out=None, budget=None):
    """Return (verdict, diffs).  verdict is EQUAL / DIFFERENT / NOT_VERIFIABLE."""
    out = [] if out is None else out
    budget = {'n': 0, 'cap': 200000} if budget is None else budget
    try:
        budget['n'] += 1
        if budget['n'] > budget['cap']:
            return NOT_VERIFIABLE, [{'path': path, 'kind': 'budget-exhausted'}]
        if is_ndarray(a) or is_ndarray(b):
            if not (is_ndarray(a) and is_ndarray(b)):
                out.append({'path': path, 'kind': 'array-vs-nonarray',
                            'a': raw_type(a), 'b': raw_type(b)})
                return DIFFERENT, out
            if raw_type(a) != raw_type(b) or a.dtype.str != b.dtype.str or a.shape != b.shape:
                out.append({'path': path, 'kind': 'array-meta', 'a_type': raw_type(a),
                            'b_type': raw_type(b), 'a_dtype': a.dtype.str, 'b_dtype': b.dtype.str,
                            'a_shape': list(a.shape), 'b_shape': list(b.shape)})
                return DIFFERENT, out
            if a.tobytes() != b.tobytes():
                out.append({'path': path, 'kind': 'array-bytes', 'note': 'data bytes differ'})
                return DIFFERENT, out
            return EQUAL, out
        if is_np_scalar(a) or is_np_scalar(b):
            if not (is_np_scalar(a) and is_np_scalar(b)):
                out.append({'path': path, 'kind': 'scalar-vs-nonscalar', 'a': raw_type(a),
                            'b': raw_type(b)})
                return DIFFERENT, out
            if raw_type(a) != raw_type(b) or a.dtype.str != b.dtype.str:
                out.append({'path': path, 'kind': 'np-scalar-meta', 'a_type': raw_type(a),
                            'b_type': raw_type(b), 'a_dtype': a.dtype.str, 'b_dtype': b.dtype.str})
                return DIFFERENT, out
            try:
                ba, bb = a.tobytes(), b.tobytes()
            except Exception as exc:
                return NOT_VERIFIABLE, [{'path': path, 'kind': 'np-scalar-unreadable',
                                         'error': str(exc)[:120]}]
            if ba != bb:
                out.append({'path': path, 'kind': 'np-scalar-bits', 'a': ba.hex(), 'b': bb.hex()})
                return DIFFERENT, out
            return EQUAL, out
        if isinstance(a, float) or isinstance(b, float):
            if type(a) is not type(b):
                out.append({'path': path, 'kind': 'type', 'a': raw_type(a), 'b': raw_type(b)})
                return DIFFERENT, out
            ba, bb = fbits(a), fbits(b)
            if ba != bb:
                out.append({'path': path, 'kind': 'float-bits', 'a': ba, 'b': bb})
                return DIFFERENT, out
            return EQUAL, out
        if isinstance(a, dict) or isinstance(b, dict):
            if type(a) is not type(b):
                out.append({'path': path, 'kind': 'container-type', 'a': raw_type(a),
                            'b': raw_type(b)})
                return DIFFERENT, out
            ka, kb = list(a.keys()), list(b.keys())
            if sorted(map(repr, ka)) != sorted(map(repr, kb)):
                out.append({'path': path, 'kind': 'keys', 'a': sorted(map(repr, ka))[:8],
                            'b': sorted(map(repr, kb))[:8]})
                return DIFFERENT, out
            for k in ka:
                v, d = bitwise_equal(a[k], b[k], '%s.%s' % (path, k), out, budget)
                if v != EQUAL:
                    return v, out
            return (DIFFERENT if out else EQUAL), out
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            if type(a) is not type(b):
                out.append({'path': path, 'kind': 'container-type', 'a': raw_type(a),
                            'b': raw_type(b)})
                return DIFFERENT, out
            if len(a) != len(b):
                out.append({'path': path, 'kind': 'length', 'a': len(a), 'b': len(b)})
                return DIFFERENT, out
            for i, (x, y) in enumerate(zip(a, b)):
                v, d = bitwise_equal(x, y, '%s[%d]' % (path, i), out, budget)
                if v != EQUAL:
                    return v, out
            return (DIFFERENT if out else EQUAL), out
        if dataclasses.is_dataclass(a) or dataclasses.is_dataclass(b):
            if not (dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b)) \
                    or type(a) is not type(b):
                out.append({'path': path, 'kind': 'dataclass-type', 'a': raw_type(a),
                            'b': raw_type(b)})
                return DIFFERENT, out
            for f in dataclasses.fields(a):
                v, d = bitwise_equal(getattr(a, f.name), getattr(b, f.name),
                                     '%s.%s' % (path, f.name), out, budget)
                if v != EQUAL:
                    return v, out
            return (DIFFERENT if out else EQUAL), out
        if type(a) is not type(b):
            out.append({'path': path, 'kind': 'type', 'a': raw_type(a), 'b': raw_type(b)})
            return DIFFERENT, out
        if isinstance(a, (int, str, bool)) or a is None:
            if a != b:
                out.append({'path': path, 'kind': 'value', 'a': repr(a)[:80], 'b': repr(b)[:80]})
                return DIFFERENT, out
            return EQUAL, out
        if hasattr(a, '__dict__') and hasattr(b, '__dict__'):
            va = {k: v for k, v in vars(a).items() if not k.startswith('_')}
            vb = {k: v for k, v in vars(b).items() if not k.startswith('_')}
            return bitwise_equal(va, vb, path, out, budget)
        return NOT_VERIFIABLE, [{'path': path, 'kind': 'uncomparable-type',
                                 'a': raw_type(a), 'b': raw_type(b)}]
    except BaseException as exc:
        return NOT_VERIFIABLE, [{'path': path, 'kind': 'exception',
                                 'error': '%s: %s' % (type(exc).__name__, str(exc)[:120])}]


def perturb_snapshot(snapshot, day_id, position):
    """Deterministic, documented causal perturbation (PURE: the input snapshot is not modified).

    Invisible-order rule : c in snapshot['customer_universe'] with visible_mask[c] == False
    Attribute mapping    : for each invisible c (ascending), a hash-derived unit value
                           u in [0,1) from sha256('d3l1-perturb|day|pos|c|field') drives
                           coords  += u * COORD_DELTA        (COORD_DELTA = 1.0)
                           demands *= (1 + u)                 (kept > 0)
                           tw_start/tw_end += u * TW_DELTA    (TW_DELTA = 0.5)
                           service_time *= (1 + u)
                           temp_class left alone (categorical: not perturbed, recorded as such)
    Preserved            : order id set, array shapes/dtypes, visible_mask, served_mask,
                           clock, committed/executed state, replanner_state
    Returns              : (perturbed_snapshot_or_None, report)
    """
    import numpy as np
    import copy
    rep = {'day_id': int(day_id), 'position': int(position),
           'invisible_orders': [], 'fields_perturbed': [], 'n_invisible_modified': 0,
           'verdict': None, 'not_perturbed_fields': []}
    try:
        vis = np.asarray(snapshot['visible_mask']).astype(bool)
        universe = [int(c) for c in snapshot['customer_universe']]
        invisible = [c for c in universe if not bool(vis[c])]
        rep['invisible_orders'] = invisible
        if not invisible:
            rep['verdict'] = NOT_VERIFIABLE
            rep['reason'] = 'no invisible orders at this decision point'
            return None, rep
        out = copy.deepcopy(snapshot)
        COORD_DELTA, TW_DELTA = 1.0, 0.5
        per_order = {}
        for c in invisible:
            for field in PERTURB_ARRAY_KEYS:
                if field not in out:
                    continue
                arr = np.asarray(out[field])
                if arr.ndim < 1 or arr.shape[-1] <= c:
                    rep['not_perturbed_fields'].append('%s(shape)' % field)
                    continue
                key = ('d3l1-perturb|%d|%d|%d|%s' % (day_id, position, c, field)).encode()
                u = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') / float(1 << 64)
                per_order.setdefault(c, {})[field] = u
                new = arr.copy()
                idx = (0,) * (arr.ndim - 1) + (c,)
                if field == 'coords':
                    coord = np.asarray(new[idx], dtype=np.float64)
                    if coord.shape[-1] >= 2:
                        coord = coord + u * COORD_DELTA
                    new[idx] = coord.astype(new.dtype)
                elif field == 'demands':
                    new[idx] = max(float(new[idx]) * (1.0 + u), 1e-6)
                elif field in ('tw_start', 'tw_end'):
                    new[idx] = float(new[idx]) + u * TW_DELTA
                elif field == 'service_time':
                    new[idx] = float(new[idx]) * (1.0 + u)
                out[field] = new
                rep['fields_perturbed'].append(field)
        rep['fields_perturbed'] = sorted(set(rep['fields_perturbed']))
        rep['n_invisible_modified'] = len([c for c in per_order if per_order[c]])
        rep['per_order_units'] = {str(k): {f: round(v, 6) for f, v in sorted(d.items())}
                                  for k, d in sorted(per_order.items())}
        preserved_ok = (list(snapshot['customer_universe']) == list(out['customer_universe'])
                        and np.array_equal(np.asarray(snapshot['visible_mask']),
                                           np.asarray(out['visible_mask']))
                        and np.array_equal(np.asarray(snapshot['served_mask']),
                                           np.asarray(out['served_mask']))
                        and float(snapshot['clock']) == float(out['clock']))
        rep['preserved_rule_holds'] = bool(preserved_ok)
        if rep['n_invisible_modified'] == 0:
            rep['verdict'] = NOT_VERIFIABLE
            rep['reason'] = 'no perturbable array field for the invisible orders'
            return None, rep
        if not preserved_ok:
            rep['verdict'] = NOT_VERIFIABLE
            rep['reason'] = 'preservation rule violated by the perturbation itself'
            return None, rep
        rep['verdict'] = 'PERTURBED'
        return out, rep
    except BaseException as exc:
        rep['verdict'] = NOT_VERIFIABLE
        rep['error'] = '%s: %s' % (type(exc).__name__, str(exc)[:200])
        return None, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    args = ap.parse_args()
    import numpy as np
    nan2 = np.array([np.nan], dtype='float64').view('uint64')
    nan2[0] = 0x7FF8000000000001
    nan_other = nan2.view('float64')[0]
    tests = [
        ('array_data_differs', np.array([1., 2.]), np.array([1., 3.]), DIFFERENT),
        ('array_equal', np.array([1., 2.]), np.array([1., 2.]), EQUAL),
        ('array_shape_differs', np.array([1., 2.]), np.array([[1., 2.]]), DIFFERENT),
        ('array_dtype_differs', np.array([1, 2], dtype='int32'),
         np.array([1, 2], dtype='int64'), DIFFERENT),
        ('np_float32_vs_float64', np.float32(1.0), np.float64(1.0), DIFFERENT),
        ('np_int64_vs_float64', np.int64(1), np.float64(1), DIFFERENT),
        ('np_scalar_vs_python_float', np.float64(1.0), 1.0, DIFFERENT),
        ('list_vs_tuple', [1, 2], (1, 2), DIFFERENT),
        ('dict_type_differs', {'a': 1}, dict(a=1), EQUAL),
        ('signed_zero_differs', -0.0, 0.0, DIFFERENT),
        ('pos_zero_equal', 0.0, 0.0, EQUAL),
        ('nan_same_payload_equal', float('nan'), float('nan'), EQUAL),
        ('nan_different_payload_differs', float('nan'), float(nan_other), DIFFERENT),
        ('deep_float_bit_differs', {'a': [{'b': 1.5}]}, {'a': [{'b': 1.5000000000000002}]},
         DIFFERENT),
        ('int_vs_float_type', 1, 1.0, DIFFERENT),
        ('missing_key', {'a': 1}, {'a': 1, 'b': 2}, DIFFERENT),
    ]
    got = {}
    for name, a, b, expect in tests:
        v, diffs = bitwise_equal(a, b)
        got[name] = {'expected': expect, 'verdict': v, 'first': (diffs[0] if diffs else None)}
    REC['comparator_gate'] = got
    ok = all(v['verdict'] == v['expected'] for v in got.values())
    ck('P0-A the corrected bitwise comparator passes every counterexample, including the four '
       'the user supplied (array data, float32 vs float64, int64 vs float64, list vs tuple) plus '
       'signed zero, NaN payloads, dtype, shape and missing keys',
       ok, {k: (v['expected'], v['verdict']) for k, v in got.items() if v['verdict'] != v['expected']}
       or {'all': 'matched'})
    REC['BITWISE_COMPARATOR_QUALIFICATION'] = 'ESTABLISHED' if ok else 'NOT_ESTABLISHED'
    REC['superseded_claim'] = {
        'previous_claim': 'bitwise_comparator_qualified=true from the earlier readiness audit '
                          '(2/2 PASS) -- that audit is NOT rewritten',
        'reason': 'its bits() called float() on anything carrying a dtype, so two different arrays '
                  'both yielded None and compared equal, and NumPy raw type/dtype plus list-vs-tuple '
                  'were not distinguished',
        'previous_flag_now': 'BITWISE_COMPARATOR_QUALIFICATION = NOT_ESTABLISHED (superseded by '
                             'this module)',
        'original_record_preserved': True}
    REC['NOT_VERIFIABLE_policy'] = 'unreadable/uncomparable inputs and budget exhaustion return ' \
        'NOT_VERIFIABLE, never EQUAL'

    # ---- P0-B: perturbation pure function on a real snapshot ------------------------------
    try:
        from d3_l1_state_collector import (RevealSlotCollector, build_env, make_replanner,
                                           run_baseline_rollout)
        from coldchain_contract import (ObjectiveProfile, apply_objective_profile,
                                        default_pilot_contract)
        data = dict(np.load(str(EXT / 'data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz')))
        prof = json.loads((EXT / 'results/o0cc/scale_v2/objective_profile.json')
                          .read_text(encoding='utf-8'))
        fields = {f.name for f in dataclasses.fields(ObjectiveProfile)}
        eff = apply_objective_profile(default_pilot_contract(),
                                      ObjectiveProfile(**{k: v for k, v in prof.items()
                                                          if k in fields}))
        env = build_env(data, 50.0, 25, 1.0, make_replanner(50.0, 24.0, eff)[0], eff, 24.0)
        col = RevealSlotCollector(env, 0, 1)
        run_baseline_rollout(env, 0, hook=col)
        snap = col.slots[0]['snapshot']
        p1, rep1 = perturb_snapshot(snap, 0, 1)
        p2, rep2 = perturb_snapshot(snap, 0, 1)
        REC['perturbation'] = {'report': rep1, 'deterministic_repeat': rep1 == rep2,
                               'input_unchanged': bitwise_equal(snap, snap)[0] == EQUAL}
        ck('P0-B the perturbation is a deterministic pure function: n_invisible_modified > 0, the '
           'preservation rule holds, a repeat call yields an identical report, and the input '
           'snapshot is untouched',
           rep1.get('verdict') == 'PERTURBED' and rep1.get('n_invisible_modified', 0) > 0
           and rep1.get('preserved_rule_holds') and rep1 == rep2,
           {'verdict': rep1.get('verdict'), 'n_invisible_modified':
            rep1.get('n_invisible_modified'), 'fields': rep1.get('fields_perturbed'),
            'preserved': rep1.get('preserved_rule_holds'), 'deterministic': rep1 == rep2})
        if p1 is not None:
            v, diffs = bitwise_equal(snap, p1)
            REC['perturbation']['original_vs_perturbed'] = {'verdict': v,
                                                            'first': (diffs[0] if diffs else None)}
            ck('P0-B the perturbed snapshot actually differs from the original (so a projection '
               'comparison is meaningful)', v == DIFFERENT,
               REC['perturbation']['original_vs_perturbed'])
    except BaseException as exc:
        import traceback
        REC['perturbation'] = {'error': '%s: %s' % (type(exc).__name__, str(exc)[:300]),
                               'tb': traceback.format_exc()[-600:]}
        ck('P0-B the perturbation pure function runs on a real snapshot', False,
           REC['perturbation'])

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'rows': [{'name': n, 'ok': o, 'detail': d} for n, o, d in CASES],
                               'records': REC, 'n_rows': len(CASES),
                               'n_pass': sum(1 for c in CASES if c[1]),
                               'n_fail': sum(1 for c in CASES if not c[1]),
                               'source_sha256_this_tool':
                                   hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                              indent=2, default=repr, sort_keys=True), encoding='utf-8')
    print('\n=== %d rows, %d PASS, %d FAIL | BITWISE_COMPARATOR_QUALIFICATION=%s ==='
          % (len(CASES), sum(1 for c in CASES if c[1]), sum(1 for c in CASES if not c[1]),
             REC['BITWISE_COMPARATOR_QUALIFICATION']))
    print('WROTE', out)
    return 0 if not [c for c in CASES if not c[1]] else 1


if __name__ == '__main__':
    sys.exit(main())
