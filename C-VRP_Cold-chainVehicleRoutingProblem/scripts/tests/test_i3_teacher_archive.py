"""Meaningful private-archive replay and corruption guards on a tiny fixture."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in ('tests', 'training', 'evaluation', 'simulation', 'coldchain'):
    sys.path.insert(0, str(ROOT / folder))
from test_i3_counterfactual_teacher import fixture, replay, check_raises
from i3_teacher_archive import write_archive, load_archive
from i3_counterfactual_teacher import TRAIN_SEED, teacher_preference
from run_identity import dataset_identity, dataset_meta_sha256


def test_archive():
    ds, contract, factory, policy, records, original = fixture(infeasible=True)
    entries = []
    for ordinal, record in enumerate(records):
        branches = [replay(record, ds, contract, factory, accept) for accept in (True, False)]
        # Save full traces as ordinary data, not arbitrary executable classes.
        for branch in branches:
            if 'traces' in branch:
                branch['traces'] = [asdict(trace) for trace in branch['traces']]
        entries.append({'day_id': 0, 'ordinal': ordinal, 'record': record,
                        'branches': branches,
                        'preference': teacher_preference(*branches, minimum_gap=5)})
    source = ROOT / 'training/i3_counterfactual_teacher.py'
    context = {'train_seed': TRAIN_SEED, 'dataset_sha256': dataset_identity(ds),
        'dataset_meta_sha256': dataset_meta_sha256(ds), 'contract_hash': contract.contract_hash,
        'source_sha256': {'teacher': hashlib.sha256(source.read_bytes()).hexdigest()},
        'generator_config': {'kind': 'tiny controlled fixture, not a scientific dataset'},
        'collection_config': {'ordinals': list(range(4)), 'not_formal': True},
        'history_dataset_sha256': 'fixture history externally defined in test',
        'history_dataset_meta_sha256': 'fixture history externally defined in test'}
    keys = [(0, i) for i in range(4)]
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'private.json'
        digest = write_archive(path, context=context, dataset=ds, entries=entries, expected_keys=keys)
        restored = load_archive(path, archive_sha256=digest, expected_context=context, expected_keys=keys)
        assert sum(not e['preference']['teacher_valid'] for e in restored['entries']) == 1
        for entry in restored['entries']:
            record = entry['record']
            assert record.record_hash == records[entry['ordinal']].record_hash
            assert record.visible == records[entry['ordinal']].visible
            assert isinstance(record.visible.accepted, frozenset)
            assert all(type(k) is int for k in record.snapshot['replanner_state']['i3_replay']['id_map'])
            chosen = replay(record, restored['dataset'], contract, factory,
                            record.candidate in policy._accepted)
            assert chosen['accepted'] == sorted(policy._accepted)
            assert chosen['rejected'] == sorted(policy._rejected)
            np.testing.assert_allclose(chosen['utility'], original['utility'], rtol=0, atol=1e-9)
        try:
            write_archive(path, context=context, dataset=ds, entries=entries, expected_keys=keys)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Do not overwrite an archived attempt')
        altered = deepcopy(context)
        altered['collection_config']['ordinals'].pop()
        check_raises(lambda: load_archive(path, archive_sha256=digest,
                     expected_context=altered, expected_keys=keys))
        check_raises(lambda: load_archive(path, archive_sha256=digest,
                     expected_context=context, expected_keys=keys[:-1]))
        check_raises(lambda: write_archive(Path(temp) / 'trimmed.json', context=context,
                     dataset=ds, entries=entries[:-1], expected_keys=keys))
        duplicate = entries + [entries[0]]
        check_raises(lambda: write_archive(Path(temp) / 'duplicate.json', context=context,
                     dataset=ds, entries=duplicate, expected_keys=keys))
        path.write_bytes(path.read_bytes() + b' ')
        check_raises(lambda: load_archive(path, archive_sha256=digest,
                     expected_context=context, expected_keys=keys))
    print('ALL PASS: typed private archive roundtrip, all-state completeness, invalid teacher retained, exact resumed utility/sets, metadata/external checksum guards and no overwrite')


if __name__ == '__main__':
    test_archive()
