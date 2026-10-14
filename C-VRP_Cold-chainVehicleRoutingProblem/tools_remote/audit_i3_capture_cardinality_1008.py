"""Independent local census of sealed raw arrays; no simulation or outcome choice."""
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/i3_day12_collection_failure_diag_1008_v1'


def decode(value):
    if isinstance(value, dict) and value.get('tag') == 'array':
        raw = base64.b64decode(value['bytes'], validate=True)
        dtype = np.dtype(value['dtype'])
        shape = tuple(value['shape'])
        assert not dtype.hasobject and len(raw) == np.prod(shape) * dtype.itemsize
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if isinstance(value, dict) and value.get('tag') == 'dict':
        return {key: decode(val) for key, val in value['items']}
    raise ValueError('Only the inert dataset dictionary and array types are decoded')


def run():
    diagnostic = json.loads((OUT / 'BASE_DIAGNOSTIC.json').read_text())
    raw = (OUT / 'base_private_typed.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == diagnostic['private_payload_sha256']
    envelope = json.loads(raw)
    packed = next(value for key, value in envelope['items'] if key == 'dataset')
    dataset = decode(packed)
    h = hashlib.sha256()
    for key in sorted(dataset):
        h.update(key.encode())
        h.update(dataset[key].tobytes())
    assert h.hexdigest() == diagnostic['dataset_sha256']
    meta = {key: {'shape': list(dataset[key].shape), 'dtype': str(dataset[key].dtype)}
            for key in sorted(dataset)}
    meta_sha = hashlib.sha256(json.dumps(meta, sort_keys=True,
                               separators=(',', ':')).encode()).hexdigest()
    assert meta_sha == diagnostic['dataset_meta_sha256']
    counts = np.count_nonzero(dataset['demands'][:, 1:] > 0, axis=1)
    assert dataset['demands'].shape == (200, 201)
    assert counts[12] == diagnostic['decisions_seen'] == 150
    assert diagnostic['original_evaluation']['hard_feasible'] is True
    assert diagnostic['original_evaluation']['failures'] == []
    assert diagnostic['captured_count'] == 3 and diagnostic['expected_count'] == 4
    old = [24, 74, 124, 174]
    rows = []
    for day in range(32):
        count = int(counts[day])
        # The fixed quartiles reproduce v1 exactly at N=200. Integer arithmetic
        # gives ceil(q*N)-1 without floating-point boundary ambiguity.
        proposed = [(numerator * count + 7) // 8 - 1 for numerator in (1, 3, 5, 7)]
        assert len(set(proposed)) == 4 and all(0 <= i < count for i in proposed)
        rows.append({'day_id': day, 'actual_orders': count,
            'old_ordinals': old, 'old_missing_ordinals': [i for i in old if i >= count],
            'proposed_quartile_ordinals': proposed})
    answer = {'qualification': 'independent_structural_diagnosis_only',
        'dataset_sha256': h.hexdigest(), 'dataset_meta_sha256': meta_sha,
        'private_payload_sha256': diagnostic['private_payload_sha256'],
        'slot_cap': 200, 'history_days': 200,
        'actual_count_min_all_history': int(counts.min()),
        'actual_count_max_all_history': int(counts.max()),
        'rows': rows, 'old_missing_days': [r['day_id'] for r in rows if r['old_missing_ordinals']],
        'old_expected_states': 128,
        'old_existing_states': sum(4 - len(r['old_missing_ordinals']) for r in rows),
        'proposed_states': sum(len(r['proposed_quartile_ordinals']) for r in rows),
        'proposed_rule': 'ceil(q*N_day)-1, q=(1/8,3/8,5/8,7/8); N_day counts positive-demand customers',
        'sampling_rule_requires_protocol_amendment': True,
        'proposed_rule_approved': False, 'simulation_rerun_started': False,
        'teacher_outcomes_used_to_choose_rule': False,
        'root_cause': 'A capped variable-order generator cannot guarantee ordinal 174 on every day',
        'original_day12_hard_feasible': True, 'original_outputs_modified': False,
        'online_gain_established': False}
    with (OUT / 'INDEPENDENT_CARDINALITY_AUDIT.json').open('x', encoding='utf-8') as stream:
        json.dump(answer, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in answer.items() if k != 'rows'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
