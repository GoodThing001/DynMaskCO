"""Explicit training/deployment historical decoding pool, independent of labels."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'evaluation'))
from run_identity import dataset_identity, dataset_meta_sha256


def pool_lineage(history_data, *, train_seed, excluded_days):
    if train_seed != 20260925 or not history_data:
        raise ValueError('Decoding pool must come from the registered training seed')
    sizes = {len(value) for value in history_data.values()}
    if len(sizes) != 1:
        raise ValueError('Historical dataset batch sizes differ')
    size = sizes.pop()
    if (not isinstance(excluded_days, list) or len(set(excluded_days)) != len(excluded_days) or
            any(type(d) is not int or not 0 <= d < size for d in excluded_days)):
        raise ValueError('Explicit unique excluded training days required')
    included = [d for d in range(size) if d not in excluded_days]
    if not included:
        raise ValueError('Empty decoder history pool')
    selected = {k: np.asarray(v)[included] for k, v in history_data.items()}
    return {'schema': 'i3_decoder_pool_v1', 'train_seed': train_seed,
        'history_days': size, 'excluded_days': sorted(excluded_days),
        'included_days': included, 'history_data_sha256': dataset_identity(history_data),
        'history_data_meta_sha256': dataset_meta_sha256(history_data),
        'selected_data_sha256': dataset_identity(selected),
        'selected_data_meta_sha256': dataset_meta_sha256(selected)}


def verified_pool_days(history_data, expected):
    actual = pool_lineage(history_data, train_seed=expected['train_seed'],
                          excluded_days=expected['excluded_days'])
    if actual != expected:
        raise ValueError('Training/deployment decoding pool lineage mismatch')
    return actual['included_days']
