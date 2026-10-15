"""Targeted shared-snapshot regressions. Synthetic C0 / v2 only; never loads VAL/test data."""
import argparse
import json
import pickle
import sys
import traceback
from pathlib import Path

EXT = Path(__file__).resolve().parents[2]
for part in ('scripts/tests', 'scripts/simulation', 'scripts/coldchain', 'scripts/evaluation'):
    sys.path.insert(0, str(EXT / part))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    import test_coldchain_contract as c0
    import test_recourse_snapshot as v2
    import numpy as np
    from d3_l1_bitwise_parity import bitwise_equal
    from recourse_snapshot import _scalar_archive, _restore_scalars, snapshot_state_hash
    rows = []
    for mod, functions in ((c0, sorted(n for n in vars(c0) if n.startswith('test_'))),
                            (v2, ('test_asymmetric_parity', 'test_wait_returning_scenario',
                                  'test_exhaustion_scenario', 'test_clone_and_hash'))):
        for name in functions:
            try:
                getattr(mod, name)()
            except Exception:
                mod.RESULTS.append({'test': name, 'status': 'FAIL', 'detail': traceback.format_exc()})
        rows.extend(dict(r, suite=mod.__name__) for r in mod.RESULTS)
    from types import SimpleNamespace
    import struct
    special_nan = struct.unpack('>d', bytes.fromhex('7ff8000000000001'))[0]
    samples = [None, True, 17, 0., -0., special_nan, np.float32(-0.), np.float64(special_nan),
               np.int64(12), np.bool_(True)]
    for sample in samples:
        original, restored = SimpleNamespace(value=sample), SimpleNamespace(value=None)
        _restore_scalars([restored], _scalar_archive([original]))
        verdict, diffs = bitwise_equal(original.value, restored.value)
        rows.append({'test': 'typed_scalar_' + type(sample).__name__ + '_' + repr(sample),
                     'status': 'PASS' if verdict == 'EQUAL' else 'FAIL', 'detail': diffs})
    base = EXT.parent / ('research/review_3080_20261010/d3_g0_evidence_root_v1/'
                         'a1_d3_l1_prereg_root_v1/v18_evidence')
    before = pickle.loads((base / 'before/TYPED_OBJECTS.pkl').read_bytes())
    after = pickle.loads((base / 'after/TYPED_OBJECTS.pkl').read_bytes())
    for old, new in zip(before['slots'], after['slots']):
        stripped = dict(new['snapshot'])
        stripped.pop('runtime_scalars_v18')
        stripped.pop('runtime_scalar_encoding')
        stripped['state_hash'] = snapshot_state_hash(stripped)
        verdict, diffs = bitwise_equal(old['snapshot'], stripped)
        rows.append({'test': 'legacy_array_projection_unchanged_slot%d' % (len(rows) - 33),
                     'status': 'PASS' if verdict == 'EQUAL' else 'FAIL', 'detail': diffs})
    result = {'classification': 'synthetic_shared_snapshot_and_C0_regressions', 'rows': rows,
              'n_pass': sum(r['status'] == 'PASS' for r in rows),
              'n_fail': sum(r['status'] != 'PASS' for r in rows),
              'excluded': 'VAL and final seeds; existing real-data snapshot tests not invoked',
              'argv': sys.argv}
    with open(args.out, 'x', encoding='utf-8') as fh:
        json.dump(result, fh, indent=2, default=repr, sort_keys=True)
    print('REGRESSION', result['n_pass'], result['n_fail'])
    return int(result['n_fail'] != 0)


if __name__ == '__main__':
    sys.exit(main())
