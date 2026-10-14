"""Checks distinguish a teacher-objective holdout from an unseen history pool."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from i3_pool_lineage import pool_lineage, verified_pool_days


def rejected(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Invalid pool lineage accepted')


def test():
    ds = {'coords': np.arange(24, dtype=np.float32).reshape(3, 4, 2),
          'demands': np.ones((3, 4), dtype=np.int32)}
    full = pool_lineage(ds, train_seed=20260925, excluded_days=[])
    assert verified_pool_days(ds, full) == [0, 1, 2]
    selected = pool_lineage(ds, train_seed=20260925, excluded_days=[2])
    assert verified_pool_days(ds, selected) == [0, 1]
    assert full['selected_data_sha256'] != selected['selected_data_sha256']
    bad = deepcopy(selected)
    bad['excluded_days'] = []
    rejected(lambda: verified_pool_days(ds, bad))
    altered = deepcopy(ds)
    altered['demands'] = altered['demands'].astype(np.int64)
    rejected(lambda: verified_pool_days(altered, selected))
    for days in ([3], [-1], [0, 0], [0, 1, 2]):
        rejected(lambda: pool_lineage(ds, train_seed=20260925, excluded_days=days))
    rejected(lambda: pool_lineage(ds, train_seed=20260927, excluded_days=[]))
    print('ALL PASS: exact included/excluded decoding days, data bytes/shape/dtype, no empty/duplicate/final-seed pool')


if __name__ == '__main__':
    test()
