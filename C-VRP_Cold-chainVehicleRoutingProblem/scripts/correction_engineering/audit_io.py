"""Lossless local event records, with explicit integrity and completion checks."""
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time


def encode(value):
    if is_dataclass(value):
        return {'__dataclass__': type(value).__module__ + '.' + type(value).__name__,
                'fields': {f.name: encode(getattr(value, f.name)) for f in fields(value)}}
    if hasattr(value, 'dtype') and hasattr(value, 'shape'):
        return {'__array__': str(value.dtype), 'shape': list(value.shape),
                'values': encode(value.tolist())}
    if isinstance(value, float) and not math.isfinite(value):
        return {'__float__': repr(value)}
    if isinstance(value, dict):
        return {'__mapping__': [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple, set, frozenset)):
        seq = sorted(value) if isinstance(value, (set, frozenset)) else value
        return {'__sequence__': type(value).__name__, 'values': [encode(v) for v in seq]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError('Unsupported audit value: ' + type(value).__name__)


def decode(value, classes=None):
    if not isinstance(value, dict):
        return value
    if '__float__' in value:
        return float(value['__float__'])
    if '__array__' in value:
        import numpy as np
        return np.array(decode(value['values'], classes), dtype=value['__array__']).reshape(value['shape'])
    if '__mapping__' in value:
        return {decode(k, classes): decode(v, classes) for k, v in value['__mapping__']}
    if '__sequence__' in value:
        constructors = {'list': list, 'tuple': tuple, 'set': set, 'frozenset': frozenset}
        return constructors[value['__sequence__']]([decode(v, classes) for v in value['values']])
    if '__dataclass__' in value:
        cls = (classes or {}).get(value['__dataclass__'])
        if cls is None:
            raise ValueError('Dataclass not explicitly allowed: ' + value['__dataclass__'])
        return cls(**{k: decode(v, classes) for k, v in value['fields'].items()})
    raise ValueError('Unknown audit encoding')


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('xb')
        self.previous = '0' * 64
        self.sequence = 0
        self.io_seconds = 0.0

    def write(self, kind, **payload):
        start = time.perf_counter()
        row = {'sequence': self.sequence, 'previous': self.previous,
               'kind': kind, 'payload': encode(payload)}
        row['sha256'] = hashlib.sha256(canonical(row)).hexdigest()
        self.stream.write(canonical(row) + b'\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.previous = row['sha256']
        self.sequence += 1
        self.io_seconds += time.perf_counter() - start

    def close(self):
        self.stream.close()


def verify(path, expected_days=None):
    previous, seen, closed = '0' * 64, [], False
    with Path(path).open('rb') as stream:
        for index, line in enumerate(stream):
            if not line.endswith(b'\n'):
                raise ValueError('Truncated final journal record')
            row = json.loads(line)
            digest = row.pop('sha256')
            if closed or row['sequence'] != index or row['previous'] != previous:
                raise ValueError('Broken journal order or trailing records')
            if hashlib.sha256(canonical(row)).hexdigest() != digest:
                raise ValueError('Journal content changed')
            previous = digest
            if row['kind'] == 'day_end':
                # Integrity checking must not import arbitrary dataclass types.
                seen.append(dict(row['payload']['__mapping__'])['day_id'])
            closed = row['kind'] == 'run_end'
    if len(seen) != len(set(seen)):
        raise ValueError('Duplicate day completion')
    return {'integrity': True, 'complete': closed and
            (expected_days is None or sorted(seen) == list(range(expected_days))),
            'completed_days': seen, 'last_sha256': previous}
