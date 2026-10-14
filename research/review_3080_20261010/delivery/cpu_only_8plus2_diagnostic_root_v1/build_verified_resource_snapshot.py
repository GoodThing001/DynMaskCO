"""Produce RESOURCE_SNAPSHOT_3080_VERIFIED.json: the verified replacement value for the
`owned_leftovers_checked: false` field left open by the delivery request registration.

This is a NEW artifact.  The original
`../actual_loaded_closure_request_root_v1/RESOURCE_SNAPSHOT.json` is left byte-intact; its
open field is answered here, from a real in-process measurement, with the exclusion list.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
s3 = json.loads((HERE / 'run_r3/DIAG_STATE.json').read_text(encoding='utf-8'))
probe = json.loads((HERE / 'run_r3/SHARD_PROBE.json').read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


pre = s3['preconditions']
out = {
    'supersedes_open_field_in': '../actual_loaded_closure_request_root_v1/RESOURCE_SNAPSHOT.json',
    'superseded_field': 'server.owned_leftovers_checked (was false)',
    'original_file_left_intact_sha256': sha(
        HERE / '../actual_loaded_closure_request_root_v1/RESOURCE_SNAPSHOT.json'),
    'source_run': 'run_r3 (attempt_id run_r3; DIAG_STATE.json sha256 below)',
    'source_run_diag_state_sha256': sha(HERE / 'run_r3/DIAG_STATE.json'),
    'source_run_shard_probe_sha256': sha(HERE / 'run_r3/SHARD_PROBE.json'),
    'measured_at_unix': pre['measured_at_unix'],
    'server': {
        'ram_total_gib': pre['total_ram_gib'],
        'ram_available_gib': pre['available_ram_gib'],
        'ram_gate_gib': pre['ram_gate_gib'],
        'ram_pass': pre['ram_pass'],
        'nproc': pre['cpu_count'],
        'load_average': pre['load_avg'],
        'target_core_index': pre['core_index'],
        'target_core_usage_percent': pre['core_usage_percent'],
        'target_core_idle_pass': pre['core_idle_pass'],
        'owned_leftovers_checked': pre['owned_leftovers_checked'],
        'owned_leftover_processes': pre['owned_leftover_processes'],
        'owned_leftovers_pass': pre['owned_leftovers_pass'],
        'leftovers_excluded_own_ancestry': pre['leftovers_excluded_own_ancestry'],
        'gpu_stage_declared': pre['gpu_stage_declared'],
        'gpu_devices_requested': pre['gpu_devices_requested'],
        'jax_devices': s3['warmup']['devices'],
        'jax_backend': s3['warmup']['backend'],
    },
    'gate_reading': {
        'backbone_ram_gate_gib': pre['ram_gate_gib'],
        'server_ram_pass': pre['ram_pass'],
        'no_owned_remote_leftovers': pre['owned_leftovers_pass'],
        'cpu_affinity_enforced': s3['cpu_lock']['enforced'],
        'cpu_only_backend': s3['warmup']['backend'] == 'cpu',
    },
    'registered_shard_identity_3080': {
        'registered_shards': probe['registered_shard_count'],
        'present_on_host': len(probe['present']),
        'absent_on_host': len(probe['absent']),
        'compute_sources_matching_local': len(probe['sources_dir_probe']),
    },
    'publication_performed': False,
    'stable_online_gain': False,
}
(HERE / 'RESOURCE_SNAPSHOT_3080_VERIFIED.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(out['server'], ensure_ascii=False, indent=2))
print('gate_reading', json.dumps(out['gate_reading'], ensure_ascii=False))
