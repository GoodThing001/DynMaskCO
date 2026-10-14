"""Read-only proof on the failed real shard; never repair/promote old records."""
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path('/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem')
OLD = ROOT / 'results/i3_registered_1008_v3'
NEW = ROOT / 'results/i3_registered_1008_v4'
EXT = NEW / 'src/C-VRP_Cold-chainVehicleRoutingProblem'
for folder in ('training', 'evaluation', 'simulation', 'coldchain'):
    sys.path.insert(0, str(EXT / 'scripts' / folder))
from i3_teacher_archive import _canonical, _pack, _unpack
from i3_counterfactual_teacher import record_identity
from recourse_snapshot import snapshot_state_hash
from run_identity import dataset_identity, dataset_meta_sha256
from scenario_saa import make_c0_contract_v2


def old_identity(record):
    return snapshot_state_hash({
        'train_seed': record.train_seed, 'dataset_sha256': record.dataset_sha256,
        'dataset_meta_sha256': record.dataset_meta_sha256,
        'continuation_signature': record.continuation_signature,
        'snapshot_hash': snapshot_state_hash(record.snapshot),
        'visible': asdict(record.visible), 'candidate': record.candidate,
        'pending': record.pending,
        'event_scenarios': [[asdict(o) for o in s] for s in record.event_scenarios]})


path = OLD / 'teacher_day_003/private.json'
raw = path.read_bytes()
envelope = json.loads(raw)
assert hashlib.sha256(_canonical(envelope['payload'])).hexdigest() == envelope['payload_sha256']
restored = _unpack(envelope['payload'])
assert _canonical(_pack(restored)) == _canonical(envelope['payload'])
context = restored['context']
assert dataset_identity(restored['dataset']) == context['dataset_sha256']
assert dataset_meta_sha256(restored['dataset']) == context['dataset_meta_sha256']
assert make_c0_contract_v2().contract_hash == context['contract_hash']
progress = [json.loads(line) for line in (path.parent / 'progress.jsonl').read_text().splitlines()]
spec = importlib.util.spec_from_file_location('teacher_terminal_audit', NEW / 'teacher_data_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
backend = audit.attach_backend(EXT)
rows = []
for entry, public in zip(restored['entries'], progress):
    record = entry['record']
    assert snapshot_state_hash(record.snapshot) == record.snapshot['state_hash']
    # In-memory prospective identity test only. No archive/output from v3 is
    # changed or used by formal training, and the old sealed hash is preserved.
    prospective = replace(record, record_hash=record_identity(record))
    second = _unpack(_pack(prospective))
    assert record_identity(second) == prospective.record_hash
    assert second.visible == record.visible
    assert second.snapshot['state_hash'] == record.snapshot['state_hash']
    reevaluated = audit.check_entry(entry, public, restored['dataset'], make_c0_contract_v2(),
                                    context['collection_config'], backend)
    rows.append({'ordinal': entry['ordinal'], 'legacy_roundtrip_identity_matches': old_identity(record) == record.record_hash,
        'prospective_canonical_roundtrip_matches': True, 'physical_snapshot_matches': True,
        'archived_terminal_metrics_recomputed': True, 'teacher_delta': reevaluated['teacher_delta']})
assert len(rows) == 4 and [row['ordinal'] for row in rows if not row['legacy_roundtrip_identity_matches']] == [174]
assert path.read_bytes() == raw
report = {'purpose': 'engineering fault proof only, no old-shard promotion',
          'failed_archive_sha256': hashlib.sha256(raw).hexdigest(), 'failed_day': 3,
          'typed_payload_and_dataset_bytes_intact': True, 'all_four_original_action_checks_passed': True,
          'rows': rows, 'old_output_modified': False, 'formal_shards_reused': False,
          'scientific_parameters_changed': False, 'online_gain_established': False}
with (NEW / 'ARCHIVE_ROUNDTRIP_FAULT_PROOF.json').open('x', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
print(json.dumps(report, indent=2))
