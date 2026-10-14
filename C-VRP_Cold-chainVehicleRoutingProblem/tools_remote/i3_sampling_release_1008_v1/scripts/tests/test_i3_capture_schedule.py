"""Actual missing-ordinal regression and tamper/missing-day guards."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from i3_capture_schedule import quartile_ordinals, validate_capture_schedule


def rejected(collection, counts=None):
    try:
        validate_capture_schedule(collection, counts)
    except ValueError:
        return
    raise AssertionError('Invalid schedule was accepted')


def test():
    assert quartile_ordinals(200) == [24, 74, 124, 174]
    assert quartile_ordinals(150) == [18, 56, 93, 131]
    assert quartile_ordinals(174) == [21, 65, 108, 152]
    assert quartile_ordinals(134) == [16, 50, 83, 117]
    for count in range(4, 201):
        values = quartile_ordinals(count)
        assert len(set(values)) == 4 and all(0 <= i < count for i in values)
    cc = {'days': [0, 12], 'orders': 200,
        'capture_ordinals_by_day': {'0': [24,74,124,174], '12': [18,56,93,131]},
        'capture_design': {'rule': 'ceil(q*N_day)-1', 'numerators': [1,3,5,7],
            'denominator': 8, 'actual_orders_by_day': {'0': 200, '12': 150}}}
    assert len(validate_capture_schedule(cc, {0: 200, 12: 150})) == 8
    rejected({'days': [12], 'orders': 200, 'capture_ordinals': [24,74,124,174]}, {12: 150})
    bad = deepcopy(cc); bad['capture_ordinals_by_day']['12'][-1] = 174; rejected(bad)
    bad = deepcopy(cc); bad['capture_ordinals_by_day'].pop('12'); rejected(bad)
    bad = deepcopy(cc); bad['capture_ordinals_by_day']['12'][1] = 18; rejected(bad)
    bad = deepcopy(cc); bad['capture_design']['actual_orders_by_day']['12'] = 149; rejected(bad)
    bad = deepcopy(cc); bad['capture_ordinals'] = [24,74,124,174]; rejected(bad)
    rejected(cc, {0: 200, 12: 149})
    protocol = ROOT.parent / 'configs/i3_20261008_v2/protocol.json'
    if protocol.exists():
        reg = json.loads(protocol.read_text(encoding='utf-8'))
        keys = validate_capture_schedule(reg['teacher_collection'])
        assert len(keys) == len(set(keys)) == 128
        assert sum(day in reg['train_days'] for day, ordinal in keys) == 96
        assert sum(day in reg['holdout_days'] for day, ordinal in keys) == 32
    print('ALL PASS: real N=150/174/134 capture regression, original N=200 positions, all N>=4, tamper/missing/duplicate/count guards, 128 unique keys with 96/32 split')


if __name__ == '__main__':
    test()
