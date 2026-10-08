"""Typed, sealed private teacher archives; no pickle or executable payloads.

An engineering serialization primitive, not a registered dataset collection
plan. The caller must provide independently sealed context/source identities
and a predeclared list of day/ordinal entries, including invalid teachers.
Load verifies the external archive checksum and expected context before replay.
Only record.visible may be passed to the actor; the archive is teacher-private.
"""
import base64
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from i3_counterfactual_teacher import TeacherRecord, TRAIN_SEED, record_identity
from scenario_saa import ScenarioOrder, VisibleSnapshot, VisibleVehicle
from run_identity import dataset_identity, dataset_meta_sha256

SCHEMA = 'i3_private_teacher_archive_v1'
CLASSES = {cls.__name__: cls for cls in (TeacherRecord, ScenarioOrder, VisibleSnapshot, VisibleVehicle)}
CONTEXT_FIELDS = {'train_seed', 'dataset_sha256', 'dataset_meta_sha256',
                  'contract_hash', 'source_sha256', 'generator_config',
                  'collection_config', 'history_dataset_sha256',
                  'history_dataset_meta_sha256'}


def _pack(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError('Object arrays are not a teacher archive format')
        return {'tag': 'array', 'dtype': value.dtype.str, 'shape': list(value.shape),
                'bytes': base64.b64encode(value.tobytes(order='C')).decode('ascii')}
    if isinstance(value, np.generic):
        return {'tag': 'scalar', 'array': _pack(np.asarray(value))}
    if is_dataclass(value):
        name = type(value).__name__
        if CLASSES.get(name) is not type(value):
            raise ValueError('Unsupported dataclass: ' + name)
        return {'tag': 'dataclass', 'name': name,
                'fields': {f.name: _pack(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, dict):
        return {'tag': 'dict', 'items': [[_pack(k), _pack(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple, set, frozenset)):
        tag = type(value).__name__
        items = [_pack(v) for v in value]
        if tag in ('set', 'frozenset'):
            items.sort(key=lambda v: json.dumps(v, sort_keys=True))
        return {'tag': tag, 'items': items}
    if isinstance(value, float):
        # Snapshots legitimately contain NaN padding/infinite offline deadlines.
        # JSON itself never contains nonstandard NaN/Infinity tokens.
        return {'tag': 'float', 'hex': value.hex()}
    if value is None or type(value) in (str, int, bool):
        return value
    raise ValueError('Unsupported private archive value: ' + type(value).__name__)


def _unpack(value):
    if value is None or type(value) in (str, int, bool):
        return value
    if not isinstance(value, dict):
        raise ValueError('Malformed typed archive value')
    tag = value.get('tag')
    if tag == 'array':
        dtype = np.dtype(value['dtype'])
        shape = value['shape']
        if dtype.hasobject or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError('Invalid archived array dtype/shape')
        raw = base64.b64decode(value['bytes'], validate=True)
        size = 1
        for n in shape:
            size *= n
        if len(raw) != size * dtype.itemsize:
            raise ValueError('Array payload does not match sealed dtype/shape')
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if tag == 'scalar':
        array = _unpack(value['array'])
        if not isinstance(array, np.ndarray) or array.shape != ():
            raise ValueError('Invalid archived scalar')
        return array[()]
    if tag == 'float':
        return float.fromhex(value['hex'])
    if tag == 'dict':
        pairs = [(_unpack(k), _unpack(v)) for k, v in value['items']]
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError('Duplicate dictionary keys in archive')
        return result
    if tag in ('list', 'tuple', 'set', 'frozenset'):
        items = [_unpack(v) for v in value['items']]
        return {'list': list, 'tuple': tuple, 'set': set, 'frozenset': frozenset}[tag](items)
    if tag == 'dataclass':
        cls = CLASSES.get(value['name'])
        if cls is None or set(value['fields']) != {f.name for f in fields(cls)}:
            raise ValueError('Unknown class or dataclass schema mismatch')
        return cls(**{k: _unpack(v) for k, v in value['fields'].items()})
    raise ValueError('Unknown archive tag')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _validate(context, dataset, entries, expected_keys):
    if not CONTEXT_FIELDS <= set(context) or context['train_seed'] != TRAIN_SEED:
        raise ValueError('Archive requires complete registered training provenance')
    if (dataset_identity(dataset) != context['dataset_sha256'] or
            dataset_meta_sha256(dataset) != context['dataset_meta_sha256']):
        raise ValueError('Archive dataset byte/shape/dtype identity mismatch')
    if not context['source_sha256'] or not context['generator_config'] or not context['collection_config']:
        raise ValueError('Empty source/generation/collection provenance')
    keys = [(entry['day_id'], entry['ordinal']) for entry in entries]
    expected_keys = [tuple(key) for key in expected_keys]
    if (len(set(expected_keys)) != len(expected_keys) or len(set(keys)) != len(keys) or
            sorted(keys) != sorted(expected_keys)):
        raise ValueError('Missing/duplicate/unexpected teacher states; no trimming')
    for entry in entries:
        record = entry['record']
        if not isinstance(record, TeacherRecord) or not record.record_hash:
            raise ValueError('Missing sealed private teacher record')
        if (record.record_hash != record_identity(record) or
                record.train_seed != TRAIN_SEED or
                record.dataset_sha256 != context['dataset_sha256'] or
                record.dataset_meta_sha256 != context['dataset_meta_sha256'] or
                int(record.snapshot['instance_id']) != entry['day_id'] or
                record.snapshot.get('coldchain_contract_hash') != context['contract_hash']):
            raise ValueError('Teacher record/context identity mismatch')


def write_archive(path, *, context, dataset, entries, expected_keys):
    """Create a new immutable-by-convention archive; never overwrite an attempt.

    Returned SHA must be kept in an external source/run manifest. Checksums are
    integrity checks, not evidence that the caller's generator/split was honest.
    """
    _validate(context, dataset, entries, expected_keys)
    payload = _pack({'context': context, 'dataset': dataset, 'entries': entries,
                     'expected_keys': [tuple(k) for k in expected_keys]})
    envelope = {'schema': SCHEMA, 'payload': payload,
                'payload_sha256': hashlib.sha256(_canonical(payload)).hexdigest()}
    content = _canonical(envelope)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(content)
    return hashlib.sha256(content).hexdigest()


def load_archive(path, *, archive_sha256, expected_context, expected_keys):
    """Verify externally sealed identities; restore ndarray types and planner IDs."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != archive_sha256:
        raise ValueError('Externally sealed archive checksum mismatch')
    envelope = json.loads(raw)
    if (envelope.get('schema') != SCHEMA or
            hashlib.sha256(_canonical(envelope['payload'])).hexdigest() != envelope['payload_sha256']):
        raise ValueError('Archive schema/payload checksum mismatch')
    restored = _unpack(envelope['payload'])
    if _canonical(_pack(restored['context'])) != _canonical(_pack(expected_context)):
        raise ValueError('Archive context differs from caller-sealed context')
    if sorted(restored['expected_keys']) != sorted(tuple(k) for k in expected_keys):
        raise ValueError('Archive capture plan differs from caller-sealed plan')
    _validate(restored['context'], restored['dataset'], restored['entries'], expected_keys)
    return restored
